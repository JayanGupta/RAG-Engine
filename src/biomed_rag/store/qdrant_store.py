"""Embedded Qdrant collection for child-chunk dense search."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config.settings import Settings


def build_qdrant(settings: Settings, embeddings: Embeddings) -> VectorStore:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    path = str(settings.qdrant_abs_path)
    settings.qdrant_abs_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=path)
    dim = len(embeddings.embed_query("dimension probe"))
    names = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in names:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )


def upsert_documents(store: VectorStore, documents: list[Document]) -> None:
    if not documents:
        return
    store.add_documents(documents)
