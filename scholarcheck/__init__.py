"""scholarcheck - verifiable literature grounding from the command line.

Never cite a paper that does not exist. Every answer is backed by live
metadata from OpenAlex, Semantic Scholar, Crossref and arXiv.

Command line::

    scholarcheck verify "Attention Is All You Need"
    scholarcheck bibtex "10.1038/s41586-025-10014-0"
    scholarcheck priorart "conformal risk control" -n 6

As a library::

    from scholarcheck import verify_citation, get_bibtex, search
    paper, confidence = verify_citation("Attention Is All You Need")
"""

from .cli import (
    multi_search as search,
    latest,
    bibtex as get_bibtex,
    best_match,
    citedby_openalex as cited_by,
    journal_lookup as journal,
    injournal,
    resolve_pdf,
    download_pdf,
    NET_ERRORS,
)

__version__ = "0.1.2"


def verify_citation(query, n=5):
    """Check whether a citation refers to a real paper.

    Returns ``(paper, confidence)``, where confidence is the fraction of the
    query's content words covered by the matched title: >=0.75 is a confident
    match, <0.45 means the citation is very likely hallucinated. ``paper`` is
    None when nothing matched.

    An empty result together with a non-empty :data:`NET_ERRORS` means the
    sources could not be reached - that is *not* evidence the paper is fake.
    """
    return best_match(query, n)


__all__ = ["search", "latest", "get_bibtex", "best_match", "cited_by",
           "journal", "injournal", "resolve_pdf", "download_pdf",
           "verify_citation", "NET_ERRORS", "__version__"]
