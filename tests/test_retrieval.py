from pathlib import Path

from config.settings import Settings
from biomed_rag.generation.chain import run_rag
from biomed_rag.ingestion.parsers import read_jsonl
from biomed_rag.store.indexer import index_documents


def test_hybrid_bm25_hits_pembrolizumab(tmp_path):
    docs = read_jsonl(Path(__file__).parent / "fixtures" / "sample_documents.jsonl")
    settings = Settings(
        data_dir=tmp_path,
        qdrant_path=str(tmp_path / "qdrant"),
        chroma_path=str(tmp_path / "chroma"),
        vector_backend="chroma",
        use_hash_embeddings=True,
    )
    settings.ensure_dirs()
    index_documents(docs, settings)
    result = run_rag(
        "What PFS hazard ratio was reported for pembrolizumab in KEYNOTE-024?",
        settings=settings,
    )
    assert result.refused is False
    pmids = {str(h.metadata.get("pmid")) for h in result.hits}
    assert "27718847" in pmids
    assert result.citations
    assert "27718847" in result.citations


def test_guardrail_short_circuits_without_hits(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        qdrant_path=str(tmp_path / "qdrant"),
        chroma_path=str(tmp_path / "chroma"),
        vector_backend="chroma",
        use_hash_embeddings=True,
    )
    result = run_rag("Write a Python function to reverse a linked list.", settings=settings)
    assert result.refused is True
    assert result.hits == []
