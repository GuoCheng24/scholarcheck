"""scholarcheck exists to answer one question without lying, so the tests cover
the two ways it could lie: calling a real paper fake, and calling a fake paper
real. Everything here is offline — the matching, identifier routing and network
bookkeeping are pure functions, and those are exactly the parts that decide the
verdict.
"""
import os
import re

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
