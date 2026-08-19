# scholarcheck

**Stop hallucinated citations.** Verify any reference against real metadata — from the command line, with zero dependencies.

Language models invent plausible-looking papers: right-sounding title, plausible authors, a DOI that resolves to nothing. `scholarcheck` answers one question honestly — **does this paper actually exist?** — by querying OpenAlex, Semantic Scholar, Crossref and arXiv directly.

```console
$ scholarcheck verify "Attention Is All You Need"
MATCH (high confidence)   [query term coverage = 100%]
Attention Is All You Need  (2017, NeurIPS; cited=6591)  doi:10.48550/arXiv.1706.03762
    Ashish Vaswani, Noam Shazeer, Niki Parmar et al.

$ scholarcheck verify "Quantum Topological Radiomics for Zebra Diagnosis in Martian Cohorts"
NOT FOUND in any of the four sources -> this citation is very likely hallucinated
```

## Why not just ask an AI assistant?

Because an assistant answers from memory, and memory is exactly what fails here. Three design choices make this different:

**1. It says "I could not check" instead of "it is fake."**
A verifier that reports a network outage as *hallucinated* is worse than no verifier. `scholarcheck` tracks every failed request and distinguishes the two:

```console
$ scholarcheck verify "Attention Is All You Need"     # with the network down
INCONCLUSIVE - could not query the sources, so nothing can be said about: Attention Is All You Need
  Could not reach: api.openalex.org: curl: (7) Connection refused
  (no proxy set; if your network needs one, set SCHOLARCHECK_PROXY)
```

It also knows which sources matter: Semantic Scholar rate-limits aggressively without an API key, so its failure never turns a real answer into "inconclusive" — only the primary sources do.

**2. It refuses to guess.**
Ask for BibTeX from a slightly-wrong title and most tools hand back the nearest hit. Silently citing the *wrong* paper is worse than citing none, so a weak match returns the candidate and stops:

```console
$ scholarcheck bibtex "Deep Residual Learning for Image Recognition in Medicine"
No confident match (best term coverage only 62%). Refusing to emit a possibly wrong entry.
Closest candidate:
  Deep Residual Learning for Image Recognition  (2016, CVPR)  doi:10.1109/CVPR.2016.90
-> If that is the paper, re-run with its DOI: scholarcheck bibtex "<DOI>".
```

**3. Recency is a separate command, on purpose.**
Relevance ranking systematically favours highly-cited older work, which is exactly wrong when you are checking whether someone *just* published your idea. `latest` filters by recency as well as relevance.

## Install

```bash
pip install scholarcheck
```

**No dependencies.** Standard library plus `curl`. Nothing to break, nothing to audit.

## Commands

| | |
|---|---|
| `verify "<title/DOI>"` | Is this citation real? Match confidence, or "likely hallucinated" |
| `bibtex "<DOI/title>"` | A BibTeX entry — refuses to guess on a weak match |
| `search "<keywords>"` | Multi-source search, re-ranked by term overlap |
| `latest "<keywords>"` | Recent work only — relevance **and** recency |
| `priorart "<claim>"` | Nearest N real papers for a claim, plus a checklist for judging whether it is already taken |
| `citedby "<DOI/title>"` | What cited this paper — has someone already extended it? |
| `journal "<name>"` | Live journal metrics, instead of quoting an impact factor from memory |
| `injournal "<name>"` | Recent papers from one journal, to study its actual conventions |
| `fetch "<DOI/arXiv id>"` | Download the open-access PDF so a claim can be checked in full text |

Add `--json` to any command for structured output, `-n` for the number of results, `--since YYYY` to bound the year.

## Use as a library

```python
from scholarcheck import verify_citation, get_bibtex, NET_ERRORS

paper, confidence = verify_citation("Attention Is All You Need")
if paper is None and NET_ERRORS:
    ...          # could not check — not evidence of anything
elif confidence >= 0.75:
    print(get_bibtex(paper["doi"]))
```

## Configuration

All optional:

| variable | effect |
|---|---|
| `SCHOLARCHECK_MAILTO` | your email — joins OpenAlex's polite pool, giving better rate limits |
| `SCHOLARCHECK_S2KEY` | Semantic Scholar API key (free) — avoids the frequent 429s |
| `SCHOLARCHECK_PROXY` | e.g. `socks5h://127.0.0.1:1080`; default is a direct connection |

Proxy behaviour is decided **solely** by `SCHOLARCHECK_PROXY`. Inherited `http_proxy` / `all_proxy` variables are stripped before each request, so the tool behaves the same on every machine.

## Notes from real use

- **Feed focused keywords, not whole sentences.** A long claim drags in off-topic papers; two or three precise terms work far better.
- **`search` favours highly-cited older work.** That is what relevance ranking does. Use `latest` when the question is "has this been done recently?"
- **A title-only judgement is not a prior-art check.** For the closest candidates, `fetch` the PDF and read it.

## License

MIT © Guo Cheng
