"""Write child chunks into Qdrant/Chroma + BM25 and parents into parents.jsonl."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document

from config.settings import Settings, get_settings
from biomed_rag.models.documents import SourceDocument
from biomed_rag.retrieval.chunking import split_documents
from biomed_rag.store.factory import build_vector_store


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
            clean[key] = [str(v) if not isinstance(v, bool) else v for v in value] if value else []
            if clean[key] and not all(isinstance(v, type(clean[key][0])) for v in clean[key]):
                clean[key] = json.dumps(value, ensure_ascii=True)
        else:
            clean[key] = json.dumps(value, ensure_ascii=True)
    return clean


def index_documents(
    documents: list[SourceDocument],
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    settings.ensure_dirs()
    child_chunks, parent_chunks = split_documents(documents, settings)
    stores = build_vector_store(settings)
    langchain_children = [
        Document(
            page_content=child.text,
            metadata=sanitize_metadata(
                {
                    **child.metadata,
                    "chunk_id": child.chunk_id,
                    "parent_id": child.parent_id,
                    "chunk_type": "child",
                }
            ),
        )
        for child in child_chunks
    ]
    if langchain_children:
        chunk_ids = [doc.metadata["chunk_id"] for doc in langchain_children]
        try:
            stores.vectorstore.add_documents(langchain_children, ids=chunk_ids)
        except Exception:
            stores.vectorstore.add_documents(langchain_children)
    stores.bm25.build(child_chunks)
    stores.bm25.save(settings.bm25_path)
    stores.parents.upsert(parent_chunks)
    return len(child_chunks)
