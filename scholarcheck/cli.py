#!/usr/bin/env python3
"""scholarcheck - verifiable literature grounding from the command line.

Built for one job: **never cite a paper that does not exist.** Every result
comes back with real metadata (DOI, authors, year, venue) pulled live from
public scholarly databases, so a reference can be checked rather than trusted.

Sources
    OpenAlex          primary, stable, no API key required
    Semantic Scholar  citations and TLDRs; often rate-limits without a key
                      (set SCHOLARCHECK_S2KEY, free to obtain)
    Crossref          DOI -> BibTeX
    arXiv             preprints, including theory papers OpenAlex has not indexed

Commands
    verify      is this citation real? -> match confidence, or "likely hallucinated"
    bibtex      DOI / title -> a BibTeX entry (refuses to guess on a weak match)
    search      multi-source search, re-ranked by term overlap
    latest      recent work only - relevance *and* recency, so new papers are not
                buried under highly-cited old ones
    priorart    pull the nearest N real papers for a specific claim, with a
                checklist for judging whether the claim is already taken
    citedby     what cited a given paper (has someone already extended it?)
    journal     live journal metrics, instead of quoting an impact factor from memory
    injournal   recent papers from one journal, to study its actual conventions
    fetch       download the open-access PDF so a claim can be checked in full text

Environment
    SCHOLARCHECK_PROXY   optional proxy, e.g. socks5h://127.0.0.1:1080 (default: direct)
    SCHOLARCHECK_MAILTO  your email; joins OpenAlex's polite pool for better rate limits
    SCHOLARCHECK_S2KEY   optional Semantic Scholar API key

Notes learned the hard way
    * Feed **focused keywords**, not a whole sentence - long claims drag in
      off-topic papers.
    * `search` ranks by relevance and therefore favours highly-cited older work.
      To see what is happening *now*, use `latest`.
    * A weak title match returns nothing rather than a plausible-looking wrong
      entry. Silently citing the wrong paper is worse than citing none.
"""
import os, sys, json, subprocess, argparse, urllib.parse, re, time, datetime

CUR_YEAR = datetime.date.today().year        # resolved at runtime, never hard-coded
PROXY = os.environ.get("SCHOLARCHECK_PROXY", "")     # direct connection by default
MAILTO = os.environ.get("SCHOLARCHECK_MAILTO", "")   # set to join OpenAlex polite pool
S2KEY = os.environ.get("SCHOLARCHECK_S2KEY")
DOI_RE = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
ARXIV_RE = r"\d{4}\.\d{4,5}(v\d+)?"

#: Network failures seen during this run. This matters: without it, a dead
#: connection looks exactly like "no such paper exists", and a citation
#: verifier that reports a network outage as "hallucinated" is worse than
#: useless. Callers check this before concluding anything is fake.
NET_ERRORS = []


def _clean_env():
    """Environment for curl, with every inherited *_proxy variable stripped.

    Proxy behaviour must be decided solely by SCHOLARCHECK_PROXY. Inheriting
    the caller's http_proxy/all_proxy makes the tool behave differently on two
    machines for no visible reason, and mixing an inherited HTTP proxy with an
    explicit SOCKS one fails in ways that are painful to debug.
    """
    return {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}


def _curl(url, accept=None, headers=None, retries=3, timeout=18):
    """Fetch via curl with backoff on 429/503. Returns the body, or None.

    On failure the reason is recorded in NET_ERRORS so the caller can tell a
    genuine "not found" apart from "could not reach the API".
    """
    cmd = ["curl", "-sS", "-L", "--max-time", str(timeout), "-w", "\n__HTTP__%{http_code}"]
    if PROXY:
        cmd += ["-x", PROXY]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    ua = "scholarcheck/0.1 (+https://github.com/GuoCheng24/scholarcheck)"
    if MAILTO:
        ua += " mailto:%s" % MAILTO
    cmd += ["-H", "User-Agent: " + ua, url]
    host = urllib.parse.urlsplit(url).netloc
    last = "unknown error"
    for k in range(retries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               env=_clean_env(), timeout=timeout + 6)
            out, code = r.stdout, ""
            if "__HTTP__" in out:
                out, _, code = out.rpartition("__HTTP__"); code = code.strip()
            if r.returncode == 0 and out.strip() and code in ("200", "201", ""):
                return out
            if r.returncode != 0:
                last = (r.stderr or "").strip().splitlines()[-1] if r.stderr else f"curl exit {r.returncode}"
            elif code:
                last = f"HTTP {code}"
            if code in ("429", "403", "503", "500", "502"):      # rate-limited or temporarily unavailable -> back off
                time.sleep(2.0 * (k + 1)); continue
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (k + 1))
    NET_ERRORS.append(f"{host}: {last}")
    return None


#: Sources whose failure genuinely means "we could not look". Semantic Scholar
#: rate-limits hard without an API key and arXiv only supplements coverage, so
#: neither should turn a real answer into "inconclusive" - otherwise the tool
#: would refuse to flag anything whenever S2 returns 429, which is often.
_PRIMARY = ("openalex", "crossref")


def _net_failed(primary_only=True):
    """True if a source we actually depend on could not be reached."""
    if primary_only:
        return any(any(h in e for h in _PRIMARY) for e in NET_ERRORS)
    return bool(NET_ERRORS)


def _net_hint():
    """A one-line, actionable explanation of what went wrong on the network."""
    crit = [e for e in NET_ERRORS if any(h in e for h in _PRIMARY)]
    uniq = list(dict.fromkeys(crit or NET_ERRORS))[:3]
    hint = "Could not reach: " + "; ".join(uniq)
    # A 429 has its own remedy, and it is not "check your proxy" - the proxy is
    # working, the other end is rate-limiting it. Saying the wrong thing here
    # sends people to debug a network that is fine.
    if any("429" in e for e in uniq):
        hint += "\n  (rate-limited, not unreachable. Already retried with backoff.)"
        if not MAILTO:
            hint += ("\n  Set SCHOLARCHECK_MAILTO=you@example.com to join OpenAlex's polite pool,"
                     "\n  which has far higher limits, then try again.")
        else:
            hint += ("\n  You are in the polite pool already; a shared IP can still be throttled."
                     "\n  Waiting a minute usually clears it.")
    elif PROXY:
        hint += f"\n  (using proxy {PROXY} from SCHOLARCHECK_PROXY - check it is reachable)"
    else:
        hint += "\n  (no proxy set; if your network needs one, set SCHOLARCHECK_PROXY)"
    return hint

def _get_json(url, headers=None):
    t = _curl(url, accept="application/json", headers=headers)
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None

# ---------- text utilities ----------
_STOP = set("the a an of for and or to in on with via using from is are be that this we our by as at "
            "how when what which under over into onto not no".split())
def _tokens(s, minlen=4):
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", (s or "").lower()) if len(w) >= minlen and w not in _STOP]
def _match_ratio(query, title):
    """Fraction of the query's content words covered by the title (used by `verify`)."""
    tq = set(_tokens(query, 3)); tt = set(_tokens(title, 3))
    if not tq or not tt:
        return 0.0
    return len(tq & tt) / len(tq)
def _relevance(query, p):
    """Term hits weighted title x3 + abstract x1; used to re-rank away off-topic results."""
    toks = set(_tokens(query))
    title = (p.get("title") or "").lower(); ab = (p.get("abstract") or "").lower()
    return sum((3 if w in title else 0) + (1 if w in ab else 0) for w in toks)

# ---------- OpenAlex ----------
def _oa_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    s = " ".join(pos[i] for i in sorted(pos))
    return (s[:300] + "…") if len(s) > 300 else s

def _oa_work(w):
    src = (w.get("primary_location") or {}).get("source") or {}
    ids = w.get("ids") or {}
    return {
        "title": w.get("title") or "(no title)",
        "year": w.get("publication_year"),
        "venue": src.get("display_name") or (w.get("type") or ""),
        "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
        "arxiv": None,
        "id": (w.get("id") or "").replace("https://openalex.org/", ""),
        "cited_by": w.get("cited_by_count", 0),
        "authors": [a.get("author", {}).get("display_name") for a in (w.get("authorships") or [])[:6]],
        "abstract": _oa_abstract(w.get("abstract_inverted_index")),
    }

def search_openalex(query, n, since=None):
    q = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={q}&per-page={n}&sort=relevance_score:desc&mailto={MAILTO}"
    if since:
        url += f"&filter=from_publication_date:{since}-01-01"
    d = _get_json(url)
    if not d or "results" not in d:
        return None
    return [_oa_work(w) for w in d["results"]]

def citedby_openalex(oa_id, n):
    url = f"https://api.openalex.org/works?filter=cites:{oa_id}&per-page={n}&sort=cited_by_count:desc&mailto={MAILTO}"
    d = _get_json(url)
    if not d or "results" not in d:
        return None
    return [_oa_work(w) for w in d["results"]]

def journal_lookup(name, n=5):
    """Live journal metrics. OpenAlex 2yr_mean_citedness is an IF-like measure:
    open, current, and not behind the JCR paywall - but not the official IF."""
    q = urllib.parse.quote(name)
    d = _get_json(f"https://api.openalex.org/sources?search={q}&per-page={n}&mailto={MAILTO}")
    if not d or "results" not in d:
        return None
    out = []
    for s in d["results"]:
        ss = s.get("summary_stats") or {}
        out.append({
            "name": s.get("display_name"), "type": s.get("type"),
            "id": (s.get("id") or "").replace("https://openalex.org/", ""),
            "if2yr": ss.get("2yr_mean_citedness"), "h_index": ss.get("h_index"),
            "works": s.get("works_count"), "issn": s.get("issn_l"),
            "publisher": s.get("host_organization_name"), "homepage": s.get("homepage_url"),
        })
    return out

def injournal(name, n, topic=None, since=None):
    """Recent papers from one journal, to study its actual conventions. Returns (name, papers)."""
    js = journal_lookup(name, 1)
    if not js or not js[0].get("id"):
        return None, None
    sid = js[0]["id"]; jname = js[0]["name"]
    yr = int(since) if since else CUR_YEAR - 2
    url = (f"https://api.openalex.org/works?filter=primary_location.source.id:{sid},"
           f"from_publication_date:{yr}-01-01&per-page={n * 3}&sort=publication_date:desc&mailto={MAILTO}")
    if topic:
        url += f"&search={urllib.parse.quote(topic)}"
    d = _get_json(url)
    if not d or "results" not in d:
        return jname, None
    ws = []
    for w in d["results"]:
        p = _oa_work(w)
        p["is_oa"] = bool((w.get("open_access") or {}).get("is_oa"))
        ws.append(p)
    if topic:
        ws.sort(key=lambda p: (_relevance(topic, p), p.get("year") or 0), reverse=True)
    return jname, ws[:n]

def resolve_openalex_id(s):
    """Title / DOI / OpenAlex id -> an OpenAlex work id (used by `citedby`)."""
    s = s.strip()
    if re.fullmatch(r"W\d+", s):
        return s
    m = re.search(DOI_RE, s)
    if m:
        d = _get_json(f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(m.group(0))}&per-page=1&mailto={MAILTO}")
        if d and d.get("results"):
            return d["results"][0]["id"].replace("https://openalex.org/", "")
    best, r = best_match(s)                 # titles go through best_match; too weak -> refuse to resolve
    if not best or r < 0.5:
        return None
    if (best.get("id") or "").startswith("W"):
        return best["id"]
    if best.get("doi"):                     # matched in S2/arXiv -> exchange the DOI for an OpenAlex id
        d = _get_json(f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(best['doi'])}&per-page=1&mailto={MAILTO}")
        if d and d.get("results"):
            return d["results"][0]["id"].replace("https://openalex.org/", "")
    return None

# ---------- Semantic Scholar ----------
def search_s2(query, n):
    q = urllib.parse.quote(query)
    fields = "title,year,venue,authors,externalIds,abstract,citationCount,tldr"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit={n}&fields={fields}"
    d = _get_json(url, headers=({"x-api-key": S2KEY} if S2KEY else None))
    if not d or "data" not in d:
        return None
    out = []
    for p in d["data"]:
        ext = p.get("externalIds") or {}
        out.append({
            "title": p.get("title"), "year": p.get("year"),
            "venue": p.get("venue") or "", "doi": ext.get("DOI"),
            "arxiv": ext.get("ArXiv"), "id": p.get("paperId"),
            "cited_by": p.get("citationCount", 0),
            "authors": [a.get("name") for a in (p.get("authors") or [])[:6]],
            "abstract": (p.get("tldr") or {}).get("text") or (p.get("abstract") or "")[:300],
        })
    return out

# ---------- arXiv ----------
def search_arxiv(query, n):
    import xml.etree.ElementTree as ET
    q = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={n}&sortBy=relevance"
    t = _curl(url)
    if not t:
        return None
    try:
        root = ET.fromstring(t)
    except Exception:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        aid = (e.findtext("a:id", default="", namespaces=ns) or "").split("/abs/")[-1]
        yr = (e.findtext("a:published", default="", namespaces=ns) or "")[:4]
        out.append({
            "title": " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split()),
            "year": int(yr) if yr.isdigit() else None,
            "venue": "arXiv", "doi": None, "arxiv": aid, "id": aid, "cited_by": 0,
            "authors": [a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)][:6],
            "abstract": " ".join((e.findtext("a:summary", default="", namespaces=ns) or "").split())[:300],
        })
    return out

def arxiv_by_id(aid):
    """Look an arXiv id up on arXiv itself. Returns one paper dict, or None.

    This is the authoritative mapping for an arXiv id, and it is used in
    preference to resolving the id through a DOI. Aggregators derive the
    10.48550/arXiv.* DOI second-hand and can attach it to the wrong record -
    observed in the wild, returning an unrelated paper for a valid id, which is
    far more damaging than returning nothing.
    """
    import xml.etree.ElementTree as ET

    aid = aid.split("v")[0]
    t = _curl(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}")
    if not t:
        return None
    try:
        root = ET.fromstring(t)
    except Exception:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    if e is None:
        return None
    title = " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split())
    if not title or title.lower().startswith("error"):
        return None
    yr = (e.findtext("a:published", default="", namespaces=ns) or "")[:4]
    doi = e.findtext("a:doi", default="", namespaces=ns) or None
    return {
        "title": title,
        "year": int(yr) if yr.isdigit() else None,
        "venue": "arXiv", "doi": doi, "arxiv": aid, "id": aid, "cited_by": 0,
        "authors": [a.findtext("a:name", default="", namespaces=ns)
                    for a in e.findall("a:author", ns)][:6],
        "abstract": " ".join((e.findtext("a:summary", default="", namespaces=ns) or "").split())[:300],
    }


#: Recorded when two sources return materially different records for the same
#: identifier. Cross-checking is the whole point of querying more than one.
SOURCE_CONFLICTS = []


# ---------- BibTeX ----------
def _fallback_bibtex(p):
    """Build an entry from metadata when there is no Crossref DOI (arXiv -> @misc with eprint)."""
    names = [x for x in (p.get("authors") or []) if x]
    auth = " and ".join(names) if names else "Unknown"
    y = p.get("year") or "n.d."
    first = (re.sub(r"[^A-Za-z]", "", (names[0].split()[-1] if names else "anon")) or "anon").lower()
    kw = (_tokens(p.get("title", "")) or ["ref"])[0]
    key = f"{first}{y}{kw}"
    title = p.get("title", "")
    if p.get("arxiv"):
        return (f"@misc{{{key},\n  title={{{title}}},\n  author={{{auth}}},\n  year={{{y}}},\n"
                f"  eprint={{{p['arxiv']}}},\n  archivePrefix={{arXiv}},\n  note={{arXiv:{p['arxiv']}}}\n}}")
    return (f"@article{{{key},\n  title={{{title}}},\n  author={{{auth}}},\n  year={{{y}}},\n"
            f"  journal={{{p.get('venue','')}}}\n}}")

# ---------- audit: check every reference in a file ----------

_BIB_START = re.compile(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,", re.S)


def _bib_entries(text):
    """Yield (type, key, body) for every entry, by matching braces.

    The previous regex ended an entry at a closing brace on its own line, so
    the two commonest hand-written shapes - `doi={...}}` and a whole entry on
    one line - matched nothing and were dropped in silence, which made `audit`
    report a clean file it had never read.
    """
    for m in _BIB_START.finditer(text):
        depth, i, n = 1, m.end(), len(text)
        while i < n and depth:
            c = text[i]
            if c == "\\":
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            yield m.group(1), m.group(2), text[m.end():i - 1]


def _bib_field(body, name):
    """One field out of a BibTeX entry body, braces balanced.

    A real parser is overkill and would cost a dependency; this handles the
    shapes bibtex files actually contain, including nested braces in titles.
    """
    m = re.search(r"\b" + name + r"\s*=\s*", body, re.I)
    if not m:
        return ""
    i = m.end()
    while i < len(body) and body[i] in " \t":
        i += 1
    if i >= len(body):
        return ""
    if body[i] == "{":
        depth, j = 0, i
        while j < len(body):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    # Inner braces in a bibtex title only protect capitalisation
                    # ("the {Kakeya} conjecture"); they are not part of the text.
                    v = re.sub(r"[{}]", "", body[i + 1:j])
                    return re.sub(r"\s+", " ", v).strip()
            j += 1
        return ""
    if body[i] == '"':
        j = body.find('"', i + 1)
        return re.sub(r"\s+", " ", body[i + 1:j]).strip() if j > 0 else ""
    j = i
    while j < len(body) and body[j] not in ",\n":
        j += 1
    return body[i:j].strip()


def parse_refs(text):
    """References out of a .bib file, or one identifier per line.

    Returns [{key, title, doi, arxiv}]. Anything without a title or an
    identifier is skipped - there is nothing to check it against.
    """
    out = []
    if "@" in text and re.search(r"@\w+\s*\{", text):
        for _typ, key, body in _bib_entries(text + "\n"):
            doi = _bib_field(body, "doi")
            eprint = _bib_field(body, "eprint")
            title = _bib_field(body, "title")
            arxiv = ""
            if eprint and re.fullmatch(ARXIV_RE + r"(v\d+)?", eprint.strip()):
                arxiv = eprint.strip()
            if not arxiv:
                note = _bib_field(body, "note") + " " + _bib_field(body, "url")
                m = re.search(r"arxiv[:/\s]*(" + ARXIV_RE + r")", note, re.I)
                if m:
                    arxiv = m.group(1)
            if title or doi or arxiv:
                out.append({"key": key, "title": title, "doi": doi, "arxiv": arxiv})
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(DOI_RE, line)
        if m:
            out.append({"key": line[:40], "title": "", "doi": m.group(0), "arxiv": ""})
            continue
        m = re.search(r"(?:arxiv[:\s/]*)(" + ARXIV_RE + r")", line, re.I)
        if m:
            out.append({"key": line[:40], "title": "", "doi": "", "arxiv": m.group(1)})
            continue
        out.append({"key": line[:40], "title": line, "doi": "", "arxiv": ""})
    return out


def doi_registered(doi, timeout=15):
    """Is this DOI registered at all? True / False / None if we could not ask.

    doi.org is the registry itself, so its answer settles the question without
    OpenAlex: a registered DOI redirects to its publisher, an unregistered one
    is a 404. That distinction survives OpenAlex being rate-limited, which on a
    shared CI runner is the common case rather than the exotic one.
    """
    cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
           "--max-time", str(timeout)]
    if PROXY:
        cmd += ["-x", PROXY]
    cmd += ["-H", "Accept: application/x-bibtex", "https://doi.org/" + doi.strip()]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=_clean_env(), timeout=timeout + 6)
        code = (r.stdout or "").strip()
    except Exception:
        return None
    if code == "404":
        return False
    if code and code[0] in "23":
        return True
    NET_ERRORS.append("doi.org: HTTP %s" % (code or "no response"))
    return None


def audit_ref(ref):
    """One reference -> (state, detail). States: OK / SUSPECT / UNCHECKED.

    UNCHECKED is not a failure. A rate-limited run must never be reported as
    a fabricated citation - that is the same error this tool exists to stop,
    and in CI it would fail somebody's build over a busy database.
    """
    NET_ERRORS.clear()
    ident = ref.get("doi") or (("arXiv:" + ref["arxiv"]) if ref.get("arxiv") else "")
    if ident:
        hit = resolve_identifier(ident)
        if hit:
            return "OK", hit.get("title", "")[:70]
        # The aggregators may be down, but for a DOI the registry itself can be
        # asked directly, and its answer is the authoritative one.
        if ref.get("doi"):
            reg = doi_registered(ref["doi"])
            if reg is False:
                return "SUSPECT", "not registered at doi.org: " + ref["doi"]
            if reg is True:
                return "OK", "registered at doi.org (metadata lookup unavailable)"
        if _net_failed():
            return "UNCHECKED", _net_hint().splitlines()[0]
        return "SUSPECT", "identifier resolves to nothing: " + ident
    title = ref.get("title") or ""
    if not title:
        return "UNCHECKED", "no title or identifier to check"
    cands = search_openalex(title, 3) or []
    if not cands or _match_ratio(title, cands[0].get("title", "")) < 0.6:
        cands = cands + (search_s2(title, 3) or [])
    if not cands:
        return ("UNCHECKED", _net_hint().splitlines()[0]) if _net_failed() else \
               ("SUSPECT", "no record in any source")
    best = max(cands, key=lambda p: _match_ratio(title, p.get("title", "")))
    r = _match_ratio(title, best.get("title", ""))
    if r >= 0.75:
        return "OK", best.get("title", "")[:70]
    if _net_failed():
        return "UNCHECKED", "a primary source was unreachable; not judged"
    if r >= 0.45:
        return "OK", "partial (%.0f%%): %s" % (r * 100, best.get("title", "")[:56])
    return "SUSPECT", "closest is only %.0f%%: %s" % (r * 100, best.get("title", "")[:52])


def bibtex(s):
    """Returns a BibTeX string; ('WEAK', candidate, ratio) when the title match is
    too weak to be safe; ('INCONCLUSIVE', candidate, ratio) when a primary source
    could not be reached; or None. It will not hand back a wrong entry."""
    s = s.strip()
    m = re.search(DOI_RE, s)
    doi = m.group(0) if m else None
    meta = None
    if not doi:                             # title input: best_match guards against returning the wrong paper
        NET_ERRORS.clear()
        meta, r = best_match(s)
        # A degraded search is exactly when the wrong entry is most likely: the
        # "best" match is then drawn from whichever sources happened to answer.
        # `verify` already refuses to speak under those conditions; emitting a
        # citation would be a stronger claim than refusing to make a weaker one.
        if _net_failed():
            return ("INCONCLUSIVE", meta, r)
        # Inclusive boundary: half the query's content words is not a confident
        # match by any reading, and an exact 0.50 used to slip through.
        if not meta or r <= 0.5:
            return ("WEAK", meta, r)        # better to return nothing than to silently emit a wrong citation
        doi = meta.get("doi")
    if doi:
        bib = _curl(f"https://doi.org/{doi}", accept="application/x-bibtex")
        if bib and "@" in bib:
            return bib.strip()
    if meta is None and doi:                # Crossref failed -> fall back to OpenAlex metadata
        d = _get_json(f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(doi)}&per-page=1&mailto={MAILTO}")
        if d and d.get("results"):
            meta = _oa_work(d["results"][0])
    return _fallback_bibtex(meta) if meta else None

# ---------- PDF retrieval: from a hit to the actual full text ----------
PDFDIR = os.environ.get("SCHOLAR_PDFDIR", "/tmp/scholar_pdfs")

def _openalex_work_full(doi=None, oa_id=None):
    if doi:
        d = _get_json(f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(doi)}&per-page=1&mailto={MAILTO}")
        return (d.get("results") or [None])[0] if d else None
    if oa_id:
        return _get_json(f"https://api.openalex.org/works/{oa_id}?mailto={MAILTO}")
    return None

def _work_pdf_url(work):
    """Find a downloadable PDF on an OpenAlex work: arXiv first, then OA pdf_url/oa_url."""
    if not work:
        return None, None
    for loc in [work.get("primary_location")] + (work.get("locations") or []):
        if not loc:
            continue
        landing = loc.get("landing_page_url") or ""
        host = ((loc.get("source") or {}).get("display_name") or "")
        if "arxiv" in (landing + host).lower():
            m = re.search(r"(\d{4}\.\d{4,5})", landing)
            if m:
                return f"https://arxiv.org/pdf/{m.group(1)}", "arXiv:" + m.group(1)
        if loc.get("pdf_url"):
            return loc["pdf_url"], "OA-pdf"
    oa = (work.get("open_access") or {}).get("oa_url")
    return (oa, "OA") if oa else (None, None)

def resolve_pdf(s):
    """arXiv id / DOI / title / URL -> (pdf_url, label, stem) or (None, reason, None)."""
    s = s.strip()
    m = re.fullmatch(r"(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?", s, re.I)
    if m:
        aid = m.group(1) + (m.group(2) or "")
        return f"https://arxiv.org/pdf/{aid}", "arXiv:" + aid, aid.replace(".", "_")
    if s.lower().startswith("http"):
        return s, "url", re.sub(r"\W+", "_", s)[-40:]
    doi_m = re.search(DOI_RE, s)
    work, stem = None, "paper"
    if doi_m:
        work = _openalex_work_full(doi=doi_m.group(0)); stem = doi_m.group(0).replace("/", "_")
    else:
        best, r = best_match(s)
        if not best or r < 0.5:
            return None, "title match too weak - use a DOI, an arXiv id, or a more exact title", None
        stem = re.sub(r"\W+", "_", (best.get("title") or "paper"))[:40]
        if best.get("arxiv"):
            aid = best["arxiv"]
            return f"https://arxiv.org/pdf/{aid}", "arXiv:" + aid, aid.replace(".", "_")
        if (best.get("id") or "").startswith("W"):
            work = _openalex_work_full(oa_id=best["id"])
        elif best.get("doi"):
            work = _openalex_work_full(doi=best["doi"])
    url, label = _work_pdf_url(work)
    return (url, label, stem) if url else (None, "no open-access PDF found (likely paywalled - try the publisher or your library)", None)

def download_pdf(url, path, retries=3):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    env = dict(os.environ); env["no_proxy"] = ""; env["NO_PROXY"] = ""
    cands = [url]
    m = re.search(r"PMC(\d+)", url)              # direct PMC links are often blocked -> fall back to the europepmc renderer
    if m:
        cands.append(f"https://europepmc.org/articles/PMC{m.group(1)}?pdf=render")
    UA = "Mozilla/5.0 (X11; Linux x86_64) scholar-ground/1.0"
    for u in cands:
        for k in range(retries):                 # these endpoints are flaky; retry automatically
            try:
                subprocess.run(["curl", "-sL", "--max-time", "60", "-x", PROXY,
                                "-H", f"User-Agent: {UA}", "-o", path, u],
                               env=env, timeout=70, capture_output=True)
                if os.path.getsize(path) > 1000 and open(path, "rb").read(5) == b"%PDF-":
                    return True
            except Exception:
                pass
            time.sleep(1.2 * (k + 1))
    try:                                         # drop non-PDF junk (error pages) instead of leaving a broken file
        if os.path.exists(path) and open(path, "rb").read(5) != b"%PDF-":
            os.remove(path)
    except Exception:
        pass
    return False

# ---------- output ----------
def _locator(p):
    if p.get("doi"):
        return "doi:" + p["doi"]
    if p.get("arxiv"):
        return "arXiv:" + p["arxiv"]
    pid = p.get("id") or ""
    if pid.startswith("W"):
        return "OpenAlex:" + pid
    return "no locator"

def fmt_paper(p, i=None):
    tag = f"[{i}] " if i is not None else ""
    names = [x for x in (p.get("authors") or []) if x]
    au = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    head = f"{tag}{p.get('title') or '(no title)'}  ({p.get('year','?')}, {p.get('venue') or '?'}; cited={p.get('cited_by',0)})  {_locator(p)}"
    body = (f"    {au}\n    {p.get('abstract','')}").rstrip()
    return head + ("\n" + body if body.strip() else "")

def multi_search(query, n, since=None):
    """OpenAlex as the stable base, opportunistically enriched with S2 and arXiv,
    de-duplicated, then re-ranked by term overlap."""
    pool, seen = [], set()
    def add(lst):
        for p in (lst or []):
            k = (p.get("title") or "").lower().strip()[:60]
            if k and k not in seen:
                seen.add(k); pool.append(p)
    add(search_openalex(query, n * 3, since))
    add(search_s2(query, n))
    if len(pool) < n:
        add(search_arxiv(query, n))
    if since:
        yr0 = int(since)
        pool = [p for p in pool if (p.get("year") or 0) >= yr0] or pool
    pool.sort(key=lambda p: (_relevance(query, p) + _year_bonus(p), p.get("cited_by", 0)), reverse=True)
    return pool[:n]

def _year_bonus(p):
    """Recency weighting, so this year's papers are not buried under highly-cited old ones."""
    y = p.get("year") or 0
    if y >= CUR_YEAR: return 2.5
    if y >= CUR_YEAR - 1: return 1.3
    if y >= CUR_YEAR - 2: return 0.5
    return 0.0

def latest(query, n, since=None):
    """Recent related work: relevance ranking plus a recency filter, which avoids the
    noise of sorting by date alone. Defaults to roughly the last 18 months."""
    yr = int(since) if since else CUR_YEAR - 1
    res = (search_openalex(query, n * 2, since=str(yr)) or [])
    s2 = search_s2(query, n) or []
    seen = {(p.get("title") or "").lower()[:60] for p in res}
    for p in s2:
        if (p.get("year") or 0) >= yr and (p.get("title") or "").lower()[:60] not in seen:
            res.append(p)
    res.sort(key=lambda p: (p.get("year") or 0, _relevance(query, p)), reverse=True)
    return res[:n]

def resolve_identifier(s):
    """A DOI or arXiv id resolved exactly, or None if `s` is not an identifier.

    Identifiers must not go through title search. Feeding "arXiv:1906.08253"
    to a title matcher returns whatever paper happens to share those digits and
    then scores it as a mismatch - which reads as "this citation is fake" when
    the truth is that the query was never looked up properly.
    """
    m = re.search(DOI_RE, s)
    if m:
        w = _openalex_work_full(doi=m.group(0))
        return _oa_work(w) if w else None

    m = re.search(r"(?:arxiv[:\s/]*)?(" + ARXIV_RE + r")", s, re.I)
    if m and (re.search(r"arxiv", s, re.I) or re.fullmatch(r"[\d.v]+", s.strip())):
        aid = m.group(1)
        primary = arxiv_by_id(aid)            # authoritative for an arXiv id
        w = _openalex_work_full(doi="10.48550/arXiv." + aid.split("v")[0])
        secondary = _oa_work(w) if w else None
        if primary and secondary:
            # Disagreement means an aggregator has the id attached to the wrong
            # work. Trust arXiv, and say so rather than silently picking one.
            if _match_ratio(primary["title"], secondary["title"]) < 0.5:
                SOURCE_CONFLICTS.append(
                    f"arXiv:{aid} -> arXiv says \"{primary['title'][:60]}\"; "
                    f"the aggregator says \"{secondary['title'][:60]}\"")
            else:
                primary["cited_by"] = secondary.get("cited_by", 0) or 0
                primary["venue"] = secondary.get("venue") or primary["venue"]
        return primary or secondary
    return None


def best_match(query, n=6):
    """Title -> best matching paper, with re-ranking and a coverage ratio, so bibtex
    and citedby never silently resolve to the wrong paper. Returns (paper|None, ratio)."""
    cands = multi_search(query, n) or []
    best = max(cands, key=lambda p: _match_ratio(query, p.get("title", "")), default=None)
    r = _match_ratio(query, best.get("title", "")) if best else 0.0
    if r < 0.6:                              # weak match -> also try arXiv directly (theory papers are often arXiv-only)
        for p in (search_arxiv(query, 5) or []):
            rr = _match_ratio(query, p.get("title", ""))
            if rr > r:
                best, r = p, rr
    return best, r

OCC_TEMPLATE = (
    "—" * 60 + "\n"
    "[Prior-art check] Ask this of every paper below. All 'no' => the claim is still open:\n"
    "  * Does it do *exactly* your specific twist and mechanism? Overlapping on the\n"
    "    broader topic alone does not count as taken.\n"
    "  * Does it cover the dimension your claim is more specific about - which\n"
    "    variable, regime, bound or mechanism?\n"
    "  * If one looks close, run `citedby \"<its DOI/title>\"` to see whether a\n"
    "    follow-up already covers your extension.\n"
    "  ! Never conclude 'already taken' from a title alone. For the closest papers,\n"
    "    run `scholarcheck fetch \"<DOI/arXiv-id/title>\"` and check the full text."
)

def _main():
    # Typing the bare command is how most people first meet a CLI. argparse's
    # default there is an error message, which is a poor greeting; show what the
    # tool does and one runnable line instead.
    if len(sys.argv) == 1:
        print(__doc__.strip() if __doc__ else "scholarcheck")
        print("\nTry one of these:\n"
              '  scholarcheck verify "Attention Is All You Need"\n'
              '  scholarcheck verify "arXiv:2502.17655"\n'
              '  scholarcheck bibtex "10.1109/cvpr.2016.90"\n'
              '  scholarcheck priorart "low-degree polynomial detection lower bound" -n 6\n'
              "\nscholarcheck --help  for every command.")
        return 0

    ap = argparse.ArgumentParser(
        description="Verifiable literature grounding - OpenAlex / Semantic Scholar / Crossref / arXiv",
        epilog='example: scholarcheck priorart "low-degree polynomial detection lower bound" -n 6 --since 2020',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "priorart", "occupancy", "verify", "bibtex", "citedby", "fetch", "latest", "journal", "injournal", "audit"])
    ap.add_argument("query", help="focused keywords / paper title / DOI / arXiv id / URL / journal name / for `audit`, a .bib file or a list of identifiers")
    ap.add_argument("-n", type=int, default=8, help="number of results (default 8)")
    ap.add_argument("--strict", action="store_true",
                    help="audit: also fail when a reference could not be checked")
    ap.add_argument("--since", default=None, help="only papers from year YYYY onwards")
    ap.add_argument("--topic", default=None, help="injournal: topic keywords to filter within the journal")
    ap.add_argument("--json", action="store_true", help="structured JSON output")
    ap.add_argument("-o", "--out", default=None, help="fetch: path to save the PDF")
    a = ap.parse_args()

    if a.cmd in ("search", "priorart", "occupancy"):
        res = multi_search(a.query, a.n, a.since)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2)); return
        if not res:
            if _net_failed():
                print(f"Could not query the sources.\n  {_net_hint()}"); return
            print("No results. Try narrower, more focused keywords - long sentences "
                  "drag in off-topic papers."); return
        print(f"# {len(res)} nearest papers - query: {a.query}\n")
        for i, p in enumerate(res, 1):
            print(fmt_paper(p, i)); print()
        if a.cmd in ("priorart", "occupancy"):
            lat = latest(a.query, 5)                       # always surface the newest work - a prior-art check must not miss it
            if lat:
                print(f"# Most recent related work (>={CUR_YEAR-1}) - check whether someone *just* did your angle:")
                for p in lat: print("  " + fmt_paper(p).replace("\n", "\n  "))
                print()
            print(OCC_TEMPLATE)
        return

    if a.cmd == "injournal":
        jname, res = injournal(a.query, a.n, a.topic, a.since)
        if a.json:
            print(json.dumps(res or [], ensure_ascii=False, indent=2)); return
        if not res:
            print(f"Journal or its recent papers not found: {a.query} (resolved name={jname})"); return
        print(f"# Recent papers in {jname} - study its actual conventions before submitting "
              f"- topic: {a.topic or '(all)'}\n")
        for i, p in enumerate(res, 1):
            oa = "OA" if p.get("is_oa") else "closed"
            print(fmt_paper(p, i) + f"   [{oa}]"); print()
        print('-> Use `scholarcheck fetch "<DOI>" -o out.pdf` to pull one, then read it for\n'
              '   structure, figure style, abstract format, length and how statistics are reported.')
        print('   Note: is_oa includes free-to-read (bronze), which is not always machine-downloadable.')
        print('   Some publishers block automated downloads; fall back to an institutional network.')
        return

    if a.cmd == "journal":
        res = journal_lookup(a.query, a.n)
        if a.json:
            print(json.dumps(res or [], ensure_ascii=False, indent=2)); return
        if not res:
            print(f"Journal not found, or the lookup failed: {a.query}"); return
        print(f"# Journal metrics, fetched live from OpenAlex - query: {a.query}")
        print("  Note: if2yr = OpenAlex 2yr_mean_citedness, an impact-factor-like measure.\n"
              "  It is close to, but not, the official Clarivate IF - use JCR for that, and the\n"
              "  journal's own aims & scope page for scope.\n")
        for p in res:
            ifv = f"{p['if2yr']:.1f}" if p.get('if2yr') is not None else "?"
            print(f"  {p['name']}  [{p.get('type','')}]  {p.get('publisher') or ''}")
            print(f"    IF-proxy(2yr)={ifv}  h-index={p.get('h_index','?')}  works={p.get('works','?')}  ISSN={p.get('issn','?')}  {p.get('homepage') or ''}")
        return

    if a.cmd == "latest":
        res = latest(a.query, a.n, a.since)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2)); return
        if not res:
            if _net_failed():
                print(f"Could not query the sources.\n  {_net_hint()}"); return
            print(f"No matches since {a.since or CUR_YEAR-1}. Try narrower keywords, "
                  "or relax --since."); return
        print(f"# Recent related work (>={a.since or CUR_YEAR-1}, newest first) - query: {a.query}\n")
        for i, p in enumerate(res, 1):
            print(fmt_paper(p, i)); print()
        return

    if a.cmd == "audit":
        try:
            text = open(a.query, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"cannot read {a.query}: {e}")
            return 2
        refs = parse_refs(text)
        if not refs:
            print(f"no references found in {a.query} - expected a .bib file, or one "
                  f"DOI / arXiv id / title per line")
            return 2
        results = [(r, *audit_ref(r)) for r in refs]
        if a.json:
            print(json.dumps([{"key": r["key"], "state": st, "detail": d}
                              for r, st, d in results], ensure_ascii=False, indent=2))
        else:
            width = min(28, max(len(r["key"]) for r, _, _ in results))
            for r, st, d in results:
                mark = {"OK": "ok      ", "SUSPECT": "SUSPECT ", "UNCHECKED": "unchecked"}[st]
                print(f"  {mark} {r['key'][:width].ljust(width)}  {d}")
            n_ok = sum(1 for _, st, _ in results if st == "OK")
            n_sus = sum(1 for _, st, _ in results if st == "SUSPECT")
            n_unk = len(results) - n_ok - n_sus
            print(f"\n{len(results)} references: {n_ok} verified, {n_sus} suspect, "
                  f"{n_unk} unchecked")
            if n_sus:
                print("Suspect entries did not resolve anywhere reachable. Check them by "
                      "hand before submitting.")
            if n_unk and not a.strict:
                print("Unchecked entries are not a verdict - a source was unavailable. "
                      "Use --strict to fail on these too.")
        n_sus = sum(1 for _, st, _ in results if st == "SUSPECT")
        n_unk = sum(1 for _, st, _ in results if st == "UNCHECKED")
        return 1 if (n_sus or (a.strict and n_unk)) else 0

    if a.cmd == "verify":
        exact = resolve_identifier(a.query)
        if exact:
            if a.json:
                print(json.dumps(exact, ensure_ascii=False, indent=2)); return
            print("MATCH (exact identifier)")
            print(fmt_paper(exact))
            for c in SOURCE_CONFLICTS:
                print(f"  ! sources disagree - {c}\n"
                      f"    arXiv is authoritative for an arXiv id; the record above is theirs.")
            return
        if re.search(DOI_RE, a.query) or re.search(r"arxiv", a.query, re.I):
            # It looked like an identifier and did not resolve - say that,
            # rather than falling back to a title search that cannot succeed.
            if _net_failed():
                print(f"INCONCLUSIVE - could not query the sources: {a.query}\n  {_net_hint()}")
            else:
                print(f"NOT FOUND - no record with this identifier: {a.query}\n"
                      f"  Check the DOI or arXiv id; if it is correct, the work may be too "
                      f"new to be indexed.")
            return
        cands = search_openalex(a.query, 3) or []
        if not cands or _match_ratio(a.query, cands[0]["title"]) < 0.6:
            cands = cands + (search_s2(a.query, 3) or [])
        if a.json:
            print(json.dumps(cands, ensure_ascii=False, indent=2)); return
        if not cands:
            # Never call something fake when we simply could not reach the APIs.
            if _net_failed():
                print(f"INCONCLUSIVE - could not query the sources, so nothing can be said "
                      f"about: {a.query}\n  {_net_hint()}"); return
            print(f"NOT FOUND in any of the four sources -> this citation is very likely "
                  f"hallucinated: {a.query}"); return
        best = max(cands, key=lambda p: _match_ratio(a.query, p.get("title", "")))
        r = _match_ratio(a.query, best.get("title", ""))
        # The evidence is asymmetric, and the verdict has to respect that.
        # Finding the paper proves it exists no matter which source was down.
        # NOT finding it proves nothing while a primary source is unreachable -
        # it may well be sitting in the database we could not query. Calling a
        # real citation fabricated is the one error this tool must not make, so
        # a negative verdict is withheld whenever the evidence is incomplete.
        if r < 0.45 and _net_failed():
            print(f"INCONCLUSIVE - nothing close was found, but a primary source was "
                  f"unreachable, so this is not evidence of fabrication: {a.query}")
            print(f"  {_net_hint()}")
            print(f"\n  Closest thing the reachable sources returned ({r:.0%} coverage):")
            print(fmt_paper(best))
            return
        verdict = ("MATCH (high confidence)" if r >= 0.75 else
                   "PARTIAL MATCH - confirm by hand that this is the same paper" if r >= 0.45 else
                   "NO MATCH -> very likely hallucinated")
        print(f"{verdict}   [query term coverage = {r:.0%}]")
        print(fmt_paper(best))
        if r < 0.75:
            others = [p for p in cands if p is not best][:3]
            if others:
                print("\nOther candidates:"); [print(fmt_paper(p)) for p in others]
        return

    if a.cmd == "bibtex":
        b = bibtex(a.query)
        if isinstance(b, tuple) and b and b[0] == "INCONCLUSIVE":
            _, cand, r = b
            print(f"INCONCLUSIVE - a primary source could not be reached, so no entry is emitted "
                  f"for: {a.query}")
            print(f"  {_net_hint()}")
            if cand:
                print(f"  (the partial search's best candidate was {r:.0%} coverage - not enough "
                      f"to stand on while sources are down)")
        elif isinstance(b, tuple) and b and b[0] == "WEAK":
            _, cand, r = b
            if cand:
                print(f"No confident match (best term coverage only {r:.0%}). Refusing to emit a "
                      f"possibly wrong entry. Closest candidate:")
                print(fmt_paper(cand))
                print('-> If that is the paper, re-run with its DOI: scholarcheck bibtex "<DOI>".\n'
                      '   Otherwise give a more exact title.')
            else:
                if _net_failed():
                    print(f"Could not query the sources.\n  {_net_hint()}")
                else:
                    print(f"No candidates resolved - check the spelling: {a.query}")
        elif b:
            print(b)
        else:
            if _net_failed():
                print(f"Could not query the sources.\n  {_net_hint()}")
            else:
                print(f"Could not resolve an entry - check the spelling or DOI: {a.query}")
        return

    if a.cmd == "fetch":
        url, label, stem = resolve_pdf(a.query)
        if not url:
            print(f"❌ {label}"); return
        path = a.out or os.path.join(PDFDIR, (stem or "paper") + ".pdf")
        if download_pdf(url, path):
            print(f"Downloaded [{label}] -> {path}\n"
                  f"   Next: read it and check the claim against the full text.")
        else:
            print(f"Download failed (flaky network, non-PDF response, or paywalled): {url}\n"
                  f"   Retry by hand: curl -sL '{url}' -o paper.pdf")
        return

    if a.cmd == "citedby":
        oid = resolve_openalex_id(a.query)
        if not oid:
            print(f"Could not resolve the target paper (try a title or DOI): {a.query}"); return
        res = citedby_openalex(oid, a.n)
        if a.json:
            print(json.dumps(res or [], ensure_ascii=False, indent=2)); return
        if not res:
            print(f"(OpenAlex {oid}) No citing works recorded yet, or the lookup failed."); return
        print(f"# {len(res)} papers citing {oid} (most cited first) - check whether your\n# extension is already covered:\n")
        for i, p in enumerate(res, 1):
            print(fmt_paper(p, i)); print()
        return

def main():
    """CLI entry point."""
    return _main()


if __name__ == "__main__":
    main()
