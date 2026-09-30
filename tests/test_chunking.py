from config.settings import Settings
from biomed_rag.ingestion.parsers import read_jsonl
from biomed_rag.retrieval.chunking import split_documents
from pathlib import Path


def test_parent_child_linkage():
    docs = read_jsonl(Path(__file__).parent / "fixtures" / "sample_documents.jsonl")
    settings = Settings(child_chunk_tokens=40, parent_chunk_tokens=2000)
    children, parents = split_documents(docs, settings)
    assert parents
    assert children
    parent_ids = {p.parent_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids
        assert child.metadata.get("pmid")
        assert child.metadata.get("chunk_type") == "child"
