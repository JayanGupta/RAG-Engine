"""Open the three stores used at query time: dense vectors, BM25, parents."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config.settings import Settings
from biomed_rag.store.bm25_index import BM25Index
from biomed_rag.store.embeddings import get_embeddings
from biomed_rag.store.parent_store import ParentDocStore


@dataclass
class VectorStoreBundle:
    vectorstore: VectorStore
    embeddings: Embeddings
    backend: str
    bm25: BM25Index
    parents: ParentDocStore


def build_vector_store(settings: Settings) -> VectorStoreBundle:
    embeddings = get_embeddings(settings)
    requested_backend = (settings.vector_backend or "qdrant").lower()
    dense_store: VectorStore | None = None
    active_backend = requested_backend
    if requested_backend == "qdrant":
        try:
            from biomed_rag.store.qdrant_store import build_qdrant

            dense_store = build_qdrant(settings, embeddings)
        except Exception:
            active_backend = "chroma"
    if dense_store is None:
        from biomed_rag.store.chroma_store import build_chroma

        dense_store = build_chroma(settings, embeddings)
        active_backend = "chroma"
    return VectorStoreBundle(
        vectorstore=dense_store,
        embeddings=embeddings,
        backend=active_backend,
        bm25=BM25Index.load(settings.bm25_path),
        parents=ParentDocStore(settings.parent_store_path),
    )
