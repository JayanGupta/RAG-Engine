"""Chroma fallback when Qdrant fails to start."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config.settings import Settings

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def build_chroma(settings: Settings, embeddings: Embeddings) -> VectorStore:
    from langchain_chroma import Chroma

    settings.chroma_abs_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_abs_path),
    )
