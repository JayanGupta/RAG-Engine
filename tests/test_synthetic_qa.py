from pathlib import Path

from config.settings import Settings
from biomed_rag.eval.synthetic_qa import generate_synthetic_qa
from biomed_rag.ingestion.parsers import read_jsonl


def test_generates_at_least_50(tmp_path):
    docs = read_jsonl(Path(__file__).parent / "fixtures" / "sample_documents.jsonl")
    settings = Settings(data_dir=tmp_path, use_hash_embeddings=True)
    settings.ensure_dirs()
    items = generate_synthetic_qa(settings, n=50, documents=docs)
    assert len(items) >= 50
    qtypes = {i.qtype for i in items}
    assert "comparison" in qtypes
    assert "adverse_event" in qtypes
    assert "adversarial_oos" in qtypes
    assert "adversarial_dx" in qtypes
