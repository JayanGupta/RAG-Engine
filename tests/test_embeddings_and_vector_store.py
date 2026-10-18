"""Tests for HashEmbeddings and VectorStore."""

from rag_engine.documents import Document
from rag_engine.chunking import chunk_document
from rag_engine.embeddings import HashEmbeddings, cosine_similarity
from rag_engine.vector_store import VectorStore


def test_hash_embeddings():
    embedder = HashEmbeddings(dim=128)
    vec = embedder.embed_text("quantum computing qubits")
    assert len(vec) == 128
    # Test normalization
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_cosine_similarity():
    embedder = HashEmbeddings(dim=128)
    vec_a = embedder.embed_text("quantum computing")
    vec_b = embedder.embed_text("quantum mechanics computing")
    vec_c = embedder.embed_text("recipe for chocolate cake")

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)
    assert sim_ab > sim_ac


def test_vector_store_search():
    doc1 = Document(text="Quantum computers use qubits in superposition.", title="Quantum")
    doc2 = Document(text="Solar panels convert sunlight directly into electricity.", title="Solar")

    chunks = chunk_document(doc1) + chunk_document(doc2)
    store = VectorStore()
    store.add_chunks(chunks)

    results = store.search("qubits superposition", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk"].title == "Quantum"
