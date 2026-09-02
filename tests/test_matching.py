"""scholarcheck exists to answer one question without lying, so the tests cover
the two ways it could lie: calling a real paper fake, and calling a fake paper
real. Everything here is offline — the matching, identifier routing and network
bookkeeping are pure functions, and those are exactly the parts that decide the
verdict.
"""
import os
import re
import sys

import pytest

from scholarcheck import cli


class TestTokens:
    def test_drops_short_words(self):
        t = cli._tokens("the sticky Kakeya sets and the conjecture")
        assert "the" not in t and "and" not in t
        assert "sticky" in t and "kakeya" in t

    def test_is_case_insensitive(self):
        assert cli._tokens("KAKEYA Conjecture") == cli._tokens("kakeya conjecture")

    def test_minlen_is_configurable(self):
        # "gap", not "and": length is only half the filter — stopwords go too,
        # and "and" is one, so it would be dropped at any minlen.
        assert "gap" in cli._tokens("sticky gap sets", minlen=3)
        assert "gap" not in cli._tokens("sticky gap sets", minlen=4)

    def test_stopwords_are_dropped_regardless_of_length(self):
        """Otherwise a title matches on "the" and "with" and every citation
        looks plausible."""
        t = cli._tokens("using the model with that data", minlen=3)
        assert not ({"the", "with", "that", "using"} & set(t))


class TestMatchRatio:
    def test_identical_titles_score_one(self):
        t = "Sticky Kakeya sets and the sticky Kakeya conjecture"
        assert cli._match_ratio(t, t) == pytest.approx(1.0)

    def test_unrelated_titles_score_low(self):
        r = cli._match_ratio("Sticky Kakeya sets and the sticky Kakeya conjecture",
                             "Attention is all you need")
        assert r < 0.2

    def test_subset_query_still_matches(self):
        """A user typing a partial title must not be told their citation is fake."""
        r = cli._match_ratio("sticky Kakeya conjecture",
                             "Sticky Kakeya sets and the sticky Kakeya conjecture")
        assert r > 0.8

    def test_ratio_stays_in_range(self):
        for q, t in [("", "anything"), ("a b c", ""), ("one", "one")]:
            assert 0.0 <= cli._match_ratio(q, t) <= 1.0


class TestIdentifierRouting:
    """An identifier must never be sent through the title matcher. Doing so
    returns whatever paper shares those digits and scores it as a mismatch,
    which reads as 'this citation is fake' when the query was simply mishandled.
    That bug shipped once; these guard the routing that fixed it."""

    @pytest.mark.parametrize("s", [
        "10.48550/arXiv.2502.17655",
        "doi:10.1090/ulect/064",
        "https://doi.org/10.1038/s41586-024-07487-w",
    ])
    def test_dois_are_recognised(self, s):
        assert re.search(cli.DOI_RE, s), f"{s} should look like a DOI"

    @pytest.mark.parametrize("s", [
        "arXiv:2502.17655", "arxiv 2210.09581", "2601.14411", "arXiv:2506.09985v2",
    ])
    def test_arxiv_ids_are_recognised(self, s):
        m = re.search(r"(?:arxiv[:\s/]*)?(" + cli.ARXIV_RE + r")", s, re.I)
        assert m and (re.search(r"arxiv", s, re.I) or re.fullmatch(r"[\d.v]+", s.strip()))

    @pytest.mark.parametrize("s", [
        "Sticky Kakeya sets and the sticky Kakeya conjecture",
        "Attention is all you need",
        "Deep residual learning for image recognition",
    ])
    def test_plain_titles_are_not_mistaken_for_identifiers(self, s):
        assert not re.search(cli.DOI_RE, s)
        m = re.search(r"(?:arxiv[:\s/]*)?(" + cli.ARXIV_RE + r")", s, re.I)
        assert not (m and (re.search(r"arxiv", s, re.I) or re.fullmatch(r"[\d.v]+", s.strip())))


class TestProxyHygiene:
    def test_clean_env_removes_every_proxy_variable(self, monkeypatch):
        """A stale proxy in the environment made lookups fail, and a failed
        lookup used to be reported as 'not found' — i.e. as a fake citation."""
        for k in ("http_proxy", "HTTPS_PROXY", "all_proxy", "NO_PROXY"):
            monkeypatch.setenv(k, "http://127.0.0.1:9")
        env = cli._clean_env()
        assert not [k for k in env if "proxy" in k.lower()]

    def test_clean_env_keeps_everything_else(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("SOME_TOKEN", "keepme")
        env = cli._clean_env()
        assert env.get("SOME_TOKEN") == "keepme" and "PATH" in env


class TestNetworkBookkeeping:
    def test_starts_with_no_recorded_failures(self):
        cli.NET_ERRORS.clear()
        assert cli._net_failed() is False

    def test_a_primary_source_failure_is_recorded(self):
        """The distinction the tool is built on: 'we could not check' must never
        be rendered as 'this paper does not exist'."""
        cli.NET_ERRORS.clear()
        cli.NET_ERRORS.append(("openalex", "timeout"))
        try:
            assert cli._net_failed() is True
        finally:
            cli.NET_ERRORS.clear()

    def test_a_secondary_source_failure_does_not_poison_the_verdict(self):
        """Semantic Scholar rate-limits constantly. If that alone made every
        answer INCONCLUSIVE the tool would be useless."""
        cli.NET_ERRORS.clear()
        cli.NET_ERRORS.append(("semanticscholar", "429"))
        try:
            assert cli._net_failed(primary_only=True) is False
        finally:
            cli.NET_ERRORS.clear()


class TestBibtexRefusesToGuess:
    """`bibtex` promises never to hand back a wrong entry. Two ways it did.

    Found by running the README's own example while OpenAlex was rate-limiting:
    a query for a paper about residual learning in medicine came back with a
    BibTeX entry for an unrelated paper on image retargeting. The partial search
    scored it at exactly 0.50, the gate was `< 0.5`, and nothing anywhere said
    the sources were down.
    """

    def test_a_primary_source_failure_blocks_the_entry(self, monkeypatch):
        monkeypatch.setattr(cli, "best_match",
                            lambda s: ({"title": "Some Unrelated Paper", "doi": "10.0/x"}, 0.9))
        def fail(*a, **k):
            cli.NET_ERRORS.append(("openalex", "HTTP 429"))
            return None, 0.9
        monkeypatch.setattr(cli, "best_match", fail)
        out = cli.bibtex("Deep Residual Learning for Image Recognition in Medicine")
        assert isinstance(out, tuple) and out[0] == "INCONCLUSIVE"
        cli.NET_ERRORS.clear()

    def test_exactly_half_coverage_is_weak_not_confident(self, monkeypatch):
        """A query whose content words are half covered is a coin flip, and the
        boundary used to let it through."""
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "best_match",
                            lambda s: ({"title": "Half Matching Paper", "doi": "10.0/x"}, 0.5))
        out = cli.bibtex("some title here")
        assert isinstance(out, tuple) and out[0] == "WEAK"

    def test_a_strong_match_is_still_allowed_through(self, monkeypatch):
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "best_match",
                            lambda s: ({"title": "Strong", "doi": "10.0/x"}, 0.95))
        monkeypatch.setattr(cli, "_curl", lambda *a, **k: "@article{strong2020, title={Strong}}")
        out = cli.bibtex("strong matching title")
        assert isinstance(out, str) and out.startswith("@article")


class TestNegativeVerdictsNeedCompleteEvidence:
    """The one error this tool must not make is calling a real citation
    fabricated. The evidence for that is asymmetric: finding a paper proves it
    exists whatever else was down, while failing to find it proves nothing at
    all if a primary source could not be reached — it may be sitting in exactly
    the database that was unavailable.
    """

    def test_no_match_is_withheld_when_a_primary_source_failed(self, monkeypatch, capsys):
        cli.NET_ERRORS.clear()
        cli.NET_ERRORS.append("api.openalex.org: HTTP 429")
        monkeypatch.setattr(cli, "search_openalex", lambda q, n=3: [])
        monkeypatch.setattr(cli, "search_s2", lambda q, n=3:
                            [{"title": "Something Entirely Different", "authors": [], "year": 2020}])
        monkeypatch.setattr(sys, "argv", ["scholarcheck", "verify", "A Paper That Does Exist"])
        try:
            cli._main()
            out = capsys.readouterr().out
            assert "INCONCLUSIVE" in out
            assert "hallucinated" not in out, "must not accuse while evidence is incomplete"
            assert "not evidence of fabrication" in out
        finally:
            cli.NET_ERRORS.clear()

    def test_no_match_still_stands_when_every_source_answered(self, monkeypatch, capsys):
        """The guard must not become 'never say anything negative' — that would
        remove the feature."""
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "search_openalex", lambda q, n=3:
                            [{"title": "Something Entirely Different", "authors": [], "year": 2020}])
        monkeypatch.setattr(cli, "search_s2", lambda q, n=3: [])
        monkeypatch.setattr(sys, "argv",
                            ["scholarcheck", "verify",
                             "Quantum Topological Radiomics for Zebra Diagnosis"])
        cli._main()
        out = capsys.readouterr().out
        assert "hallucinated" in out and "INCONCLUSIVE" not in out


class TestBibParsing:
    """The audit command is only useful if it can read a real .bib file, so the
    shapes that actually appear in one are pinned: nested braces in titles,
    quoted values, arXiv eprints, entries with no identifier at all."""

    BIB = r'''
@article{wang2025kakeya,
  title = {Volume estimates for unions of convex sets, and the {Kakeya} set conjecture},
  author = {Wang, Hong and Zahl, Joshua},
  eprint = {2502.17655},
  archivePrefix = {arXiv}
}
@inproceedings{he2016resnet,
  title = "Deep Residual Learning for Image Recognition",
  year = 2016,
  doi = {10.1109/cvpr.2016.90}
}
@misc{nothing2024,
  author = {Nobody},
  year = {2024}
}
'''

    def test_finds_every_entry_with_something_to_check(self):
        refs = cli.parse_refs(self.BIB)
        keys = [r["key"] for r in refs]
        assert "wang2025kakeya" in keys and "he2016resnet" in keys
        assert "nothing2024" not in keys, "an entry with no title or id cannot be checked"

    def test_nested_braces_in_a_title_survive(self):
        r = [x for x in cli.parse_refs(self.BIB) if x["key"] == "wang2025kakeya"][0]
        assert "Kakeya" in r["title"] and "{" not in r["title"]

    def test_quoted_values_are_read(self):
        r = [x for x in cli.parse_refs(self.BIB) if x["key"] == "he2016resnet"][0]
        assert r["title"].startswith("Deep Residual")
        assert r["doi"] == "10.1109/cvpr.2016.90"

    def test_arxiv_eprint_is_picked_up(self):
        r = [x for x in cli.parse_refs(self.BIB) if x["key"] == "wang2025kakeya"][0]
        assert r["arxiv"] == "2502.17655"

    def test_plain_identifier_list_also_works(self):
        refs = cli.parse_refs("10.1109/cvpr.2016.90\narXiv:2502.17655\n# a comment\n\n")
        assert len(refs) == 2
        assert refs[0]["doi"] and refs[1]["arxiv"]


class TestAuditVerdicts:
    """CI is where a wrong verdict does the most damage: a rate-limited run must
    never fail somebody's build by calling their citations invented."""

    def test_nothing_reachable_at_all_yields_unchecked(self, monkeypatch):
        """Aggregators and registry both unreachable: nothing can be concluded.

        `doi_registered` is stubbed rather than left to run, or this test reaches
        doi.org for real and then passes or fails depending on whether the
        machine has a network. That is exactly how it slipped through locally
        and failed in CI.
        """
        cli.NET_ERRORS.clear()
        def boom(ident):
            cli.NET_ERRORS.append("api.openalex.org: HTTP 429")
            return None
        monkeypatch.setattr(cli, "resolve_identifier", boom)
        monkeypatch.setattr(cli, "doi_registered", lambda doi, timeout=15: None)
        st, _ = cli.audit_ref({"key": "k", "title": "", "doi": "10.1/x", "arxiv": ""})
        assert st == "UNCHECKED"
        cli.NET_ERRORS.clear()

    def test_identifier_that_resolves_to_nothing_is_suspect(self, monkeypatch):
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "resolve_identifier", lambda ident: None)
        monkeypatch.setattr(cli, "doi_registered", lambda doi, timeout=15: False)
        st, detail = cli.audit_ref({"key": "k", "title": "", "doi": "10.9999/nope", "arxiv": ""})
        assert st == "SUSPECT" and "doi.org" in detail

    def test_a_real_identifier_is_ok(self, monkeypatch):
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "doi_registered", lambda doi, timeout=15: None)
        monkeypatch.setattr(cli, "resolve_identifier",
                            lambda ident: {"title": "A Real Paper"})
        st, detail = cli.audit_ref({"key": "k", "title": "", "doi": "10.1/x", "arxiv": ""})
        assert st == "OK" and "A Real Paper" in detail

    def test_title_only_entry_far_from_everything_is_suspect(self, monkeypatch):
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "search_openalex", lambda q, n=3:
                            [{"title": "Totally Different Subject Matter Here"}])
        monkeypatch.setattr(cli, "search_s2", lambda q, n=3: [])
        st, _ = cli.audit_ref({"key": "k", "arxiv": "", "doi": "",
                               "title": "Quantum Topological Radiomics for Zebra Diagnosis"})
        assert st == "SUSPECT"


class TestDoiRegistry:
    """doi.org is the registry, so it answers "does this DOI exist" on its own.
    That path is what keeps `audit` working while OpenAlex is throttling, which
    on a shared CI runner is the ordinary case rather than the exotic one.
    """

    @staticmethod
    def _openalex_down(ident):
        cli.NET_ERRORS.append("api.openalex.org: HTTP 429")
        return None

    def test_unregistered_doi_is_suspect_even_with_aggregators_down(self, monkeypatch):
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "resolve_identifier", self._openalex_down)
        monkeypatch.setattr(cli, "doi_registered", lambda doi, timeout=15: False)
        st, detail = cli.audit_ref({"key": "k", "title": "", "doi": "10.9999/nope", "arxiv": ""})
        assert st == "SUSPECT" and "doi.org" in detail
        cli.NET_ERRORS.clear()

    def test_registered_doi_is_ok_even_with_aggregators_down(self, monkeypatch):
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "resolve_identifier", self._openalex_down)
        monkeypatch.setattr(cli, "doi_registered", lambda doi, timeout=15: True)
        st, _ = cli.audit_ref({"key": "k", "title": "", "doi": "10.1/real", "arxiv": ""})
        assert st == "OK"
        cli.NET_ERRORS.clear()

    def test_registry_unreachable_falls_back_to_unchecked(self, monkeypatch):
        """If even doi.org cannot be reached there is nothing to conclude."""
        cli.NET_ERRORS.clear()
        monkeypatch.setattr(cli, "resolve_identifier", self._openalex_down)
        monkeypatch.setattr(cli, "doi_registered", lambda doi, timeout=15: None)
        st, _ = cli.audit_ref({"key": "k", "title": "", "doi": "10.1/x", "arxiv": ""})
        assert st == "UNCHECKED"
        cli.NET_ERRORS.clear()


class TestBibEntryShapes:
    """Hand-written .bib files come in shapes the old regex silently dropped.

    Each of these was parsed as zero entries before 2026-09-02, so `audit`
    reported a clean file it had never actually read - the worst kind of
    failure for a checker.
    """

    CASES = [
        ("closing brace on its own line", "@article{a,\n title={T},\n doi={10.1/x}\n}\n", 1),
        ("last field and brace on one line", "@article{b,\n title={T},\n doi={10.1/x}}\n", 1),
        ("whole entry on one line", "@article{c, title={T}, doi={10.1/x}}\n", 1),
        ("two one-line entries", "@article{d, title={T1}}\n@article{e, title={T2}}\n", 2),
        ("braces nested in the title", "@article{f, title={A {BERT} model}, doi={10.1/y}}\n", 1),
        ("all three shapes mixed", "@a{x, title={T1}\n}\n@b{y, title={T2}}\n"
                                   "@c{z, title={T3}, doi={10.2/z}}\n", 3),
    ]

    def test_every_shape_is_parsed(self):
        for name, text, expected in self.CASES:
            got = len(cli.parse_refs(text))
            assert got == expected, f"{name}: parsed {got}, expected {expected}"

    def test_fields_survive_the_shapes(self):
        (ref,) = cli.parse_refs("@article{c, title={T}, doi={10.1/x}}\n")
        assert ref["title"] == "T" and ref["doi"] == "10.1/x"
