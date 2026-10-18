"""Tests for BM25SearchIndex and HybridRetriever."""

from rag_engine.documents import Document
from rag_engine.chunking import chunk_document
from rag_engine.vector_store import VectorStore
from rag_engine.bm25_search import BM25SearchIndex
from rag_engine.hybrid_search import HybridRetriever, reciprocal_rank_fusion


def test_bm25_search():
    doc1 = Document(text="NIRCam and MIRI are instruments on the James Webb Space Telescope.", title="JWST")
    doc2 = Document(text="Photosynthesis produces glucose and oxygen from carbon dioxide.", title="Biology")

    chunks = chunk_document(doc1) + chunk_document(doc2)
    bm25 = BM25SearchIndex()
    bm25.build(chunks)

    hits = bm25.search("NIRCam instruments", top_k=1)
    assert len(hits) == 1
    assert hits[0]["chunk"].title == "JWST"


def test_reciprocal_rank_fusion():
    doc1 = Document(text="Superconducting qubits require millikelvin cooling.", title="Doc1")
    doc2 = Document(text="Vanadium redox flow batteries for grid storage.", title="Doc2")
    chunks = chunk_document(doc1) + chunk_document(doc2)

    dense_results = [
        {"chunk": chunks[0], "score": 0.9},
        {"chunk": chunks[1], "score": 0.4},
    ]
    bm25_results = [
        {"chunk": chunks[0], "score": 2.5},
        {"chunk": chunks[1], "score": 1.1},
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    assert len(fused) == 2
    assert fused[0]["chunk"].title == "Doc1"


def test_hybrid_retriever():
    doc1 = Document(text="Quantum algorithms like Shor's algorithm.", title="Quantum")
    doc2 = Document(text="Modern renewable energy grid storage.", title="Energy")
    chunks = chunk_document(doc1) + chunk_document(doc2)

    vs = VectorStore()
    vs.add_chunks(chunks)

    bm25 = BM25SearchIndex()
    bm25.build(chunks)

    retriever = HybridRetriever(vs, bm25)
    output = retriever.retrieve("Shor algorithm", top_k=1)
    assert len(output["results"]) == 1
    assert output["results"][0]["chunk"].title == "Quantum"
