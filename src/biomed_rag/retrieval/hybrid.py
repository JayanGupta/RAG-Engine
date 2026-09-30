"""Hybrid search: dense ANN + BM25, then Reciprocal Rank Fusion.

retrieve() returns parent-attached hits. Rerank happens in rerank.py.
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document

from config.settings import Settings
from biomed_rag.models.documents import ExtractedStat
from biomed_rag.models.retrieval import RetrievalHit
from biomed_rag.store.factory import VectorStoreBundle


def _stats_from_meta(meta: dict) -> list[ExtractedStat]:
    import json

    raw = meta.get("extracted_stats") or []
    if not raw:
        raw = meta.get("extracted_stats_json") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    out: list[ExtractedStat] = []
    for item in raw:
        try:
            out.append(ExtractedStat.model_validate(item))
        except Exception:
            continue
    return out


def _rrf_merge(
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    dense_weight: float,
    sparse_weight: float,
    k: int = 60,
) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievalHit] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] += dense_weight * (1.0 / (k + rank))
        by_id[hit.chunk_id] = hit
    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] += sparse_weight * (1.0 / (k + rank))
        if hit.chunk_id not in by_id:
            by_id[hit.chunk_id] = hit
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged: list[RetrievalHit] = []
    for chunk_id, score in ranked:
        hit = by_id[chunk_id].model_copy()
        hit.score = score
        hit.source = "ensemble"
        merged.append(hit)
    return merged


class HybridRetriever:
    def __init__(self, bundle: VectorStoreBundle, settings: Settings) -> None:
        self.bundle = bundle
        self.settings = settings

    def dense_search(self, query: str) -> list[RetrievalHit]:
        try:
            pairs = self.bundle.vectorstore.similarity_search_with_score(
                query, k=self.settings.dense_k
            )
        except Exception:
            docs = self.bundle.vectorstore.similarity_search(query, k=self.settings.dense_k)
            pairs = [(d, 0.0) for d in docs]
        hits: list[RetrievalHit] = []
        for doc, score in pairs:
            if not isinstance(doc, Document):
                continue
            meta = dict(doc.metadata or {})
            hits.append(
                RetrievalHit(
                    chunk_id=str(meta.get("chunk_id") or meta.get("id") or hash(doc.page_content)),
                    parent_id=str(meta.get("parent_id") or ""),
                    text=doc.page_content,
                    parent_text="",
                    score=float(score or 0.0),
                    source="dense",
                    metadata=meta,
                    extracted_stats=_stats_from_meta(meta),
                )
            )
        return hits

    def attach_parent_sections(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        attached: list[RetrievalHit] = []
        seen_parent_ids: set[str] = set()
        for hit in hits:
            parent = self.bundle.parents.get(hit.parent_id)
            if parent:
                if parent.parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent.parent_id)
                hit = hit.model_copy()
                hit.parent_text = parent.text
                hit.metadata = {**parent.metadata, **hit.metadata}
                if parent.extracted_stats:
                    hit.extracted_stats = parent.extracted_stats
            attached.append(hit)
        return attached

    def retrieve(self, query: str) -> list[RetrievalHit]:
        dense_hits = self.dense_search(query)
        bm25_hits = self.bundle.bm25.search(query, k=self.settings.bm25_k)
        fused_hits = _rrf_merge(
            dense_hits,
            bm25_hits,
            self.settings.ensemble_dense_weight,
            self.settings.ensemble_bm25_weight,
        )
        return self.attach_parent_sections(fused_hits)
