"""Reorder hybrid hits before generation.

Prefer Cohere if USER_COHERE_API_KEY is set, else local BGE cross-encoder,
else lexical overlap. Always return at most settings.rerank_k hits.
"""

from __future__ import annotations

from functools import lru_cache

from config.settings import Settings
from biomed_rag.models.retrieval import RetrievalHit


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _local_rerank(query: str, hits: list[RetrievalHit], model_name: str) -> list[RetrievalHit]:
    try:
        model = _cross_encoder(model_name)
        pairs = [(query, h.parent_text or h.text) for h in hits]
        scores = model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        out: list[RetrievalHit] = []
        for hit, score in ranked:
            item = hit.model_copy()
            item.score = float(score)
            item.source = "rerank"
            out.append(item)
        return out
    except Exception:
        return _lexical_rerank(query, hits)


def _lexical_rerank(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    q_tokens = set(query.lower().split())
    scored: list[RetrievalHit] = []
    for hit in hits:
        text = (hit.parent_text or hit.text).lower()
        overlap = sum(1 for t in q_tokens if t in text)
        item = hit.model_copy()
        item.score = float(overlap) + float(hit.score)
        item.source = "rerank"
        scored.append(item)
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored


def _cohere_rerank(query: str, hits: list[RetrievalHit], api_key: str, top_n: int) -> list[RetrievalHit] | None:
    try:
        import cohere
    except ImportError:
        return None
    try:
        client = cohere.Client(api_key)
        documents = [h.parent_text or h.text for h in hits]
        response = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=top_n,
        )
        out: list[RetrievalHit] = []
        for item in response.results:
            hit = hits[item.index].model_copy()
            hit.score = float(item.relevance_score)
            hit.source = "rerank"
            out.append(hit)
        return out
    except Exception:
        return None


def rerank_hits(query: str, hits: list[RetrievalHit], settings: Settings) -> list[RetrievalHit]:
    if not hits:
        return []
    top_n = settings.rerank_k
    if settings.user_cohere_api_key:
        cohere_hits = _cohere_rerank(query, hits, settings.user_cohere_api_key, top_n)
        if cohere_hits is not None:
            return cohere_hits[:top_n]
    reranked = _local_rerank(query, hits, settings.rerank_model)
    return reranked[:top_n]
