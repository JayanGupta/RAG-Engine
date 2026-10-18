"""Tests for Document creation, serialization, and file loading."""

from pathlib import Path
from rag_engine.documents import (
    Document,
    load_text_file,
    load_documents_jsonl,
    save_documents_jsonl,
)


def test_document_creation_and_dict():
    doc = Document(text="Hello world of RAG", title="Test Doc")
    assert doc.title == "Test Doc"
    assert doc.text == "Hello world of RAG"
    assert len(doc.doc_id) > 0

    data = doc.to_dict()
    restored = Document.from_dict(data)
    assert restored.title == doc.title
    assert restored.text == doc.text
    assert restored.doc_id == doc.doc_id


def test_save_and_load_jsonl(tmp_path):
    docs = [
        Document(text="Document 1 content", title="Doc 1"),
        Document(text="Document 2 content", title="Doc 2"),
    ]
    filepath = tmp_path / "test_docs.jsonl"
    save_documents_jsonl(docs, filepath)
    assert filepath.exists()

    loaded = load_documents_jsonl(filepath)
    assert len(loaded) == 2
    assert loaded[0].title == "Doc 1"
    assert loaded[1].title == "Doc 2"


def test_load_text_file(tmp_path):
    txt_path = tmp_path / "machine_learning.txt"
    txt_path.write_text("Machine learning is a field of artificial intelligence.", encoding="utf-8")
    doc = load_text_file(txt_path)
    assert "Machine Learning" in doc.title
    assert "artificial intelligence" in doc.text
