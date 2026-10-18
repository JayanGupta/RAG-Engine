"""Tests for text chunking and document splitting."""

from rag_engine.documents import Document
from rag_engine.chunking import split_text_into_chunks, chunk_document, chunk_documents


def test_split_text_into_chunks():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = split_text_into_chunks(text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 1
    # Check that chunks are not empty
    for chunk in chunks:
        assert len(chunk.split()) <= 30


def test_chunk_document():
    doc = Document(
        text="James Webb Space Telescope observes the cosmos in infrared wavelengths.",
        title="JWST",
        doc_id="jwst_01",
    )
    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=2)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_id == "jwst_01"
        assert chunk.title == "JWST"
        assert chunk.parent_text == doc.text


def test_chunk_documents():
    docs = [
        Document(text="First doc text", title="Doc 1"),
        Document(text="Second doc text", title="Doc 2"),
    ]
    chunks = chunk_documents(docs, chunk_size=10, chunk_overlap=2)
    assert len(chunks) == 2
