from biomed_rag.generation.citations import parse_pmids, validate_citations
from biomed_rag.models.retrieval import RetrievalHit


def _hit(pmid: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id="c1",
        parent_id="p1",
        text="child",
        parent_text="parent",
        metadata={"pmid": pmid},
    )


def test_parse_pmids():
    assert parse_pmids("See [PMID:27718847] and PMID:25891304") == ["27718847", "25891304"]


def test_validate_keeps_retrieved_pmid():
    cleaned, valid, _ = validate_citations(
        "PFS HR 0.50 [PMID:27718847]",
        [_hit("27718847")],
    )
    assert valid == ["27718847"]
    assert "[PMID:27718847]" in cleaned


def test_validate_strips_invented_pmid():
    cleaned, valid, _ = validate_citations(
        "Claim [PMID:99999999] and real [PMID:27718847]",
        [_hit("27718847")],
    )
    assert valid == ["27718847"]
    assert "99999999" not in cleaned
