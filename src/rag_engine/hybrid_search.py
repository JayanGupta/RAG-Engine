"""Hybrid search combining Dense Vector Search and BM25 Keyword Search.

Uses Reciprocal Rank Fusion (RRF) to merge the ranked result lists from both
retrieval methods into a single, high-quality candidate list.

Why Hybrid Search?
- Dense vectors understand semantic similarity (e.g. 'space telescope' -> 'astronomy').
- BM25 understands exact identifiers and keywords (e.g. 'NIRCam', 'L2', '15 millikelvin').
- Together, they significantly outperform either method on its own.
"""

from collections import defaultdict


def reciprocal_rank_fusion(dense_results, bm25_results, dense_weight=0.6, bm25_weight=0.4, k=60):
    """Merge two ranked lists using Reciprocal Rank Fusion (RRF).
    
    Formula for each document:
      Score = sum( weight / (k + rank) )
      
    Args:
        dense_results: List of dicts from vector search [{"chunk": ..., "score": ...}].
        bm25_results: List of dicts from BM25 search.
        dense_weight: Relative importance of vector search (default: 0.6).
        bm25_weight: Relative importance of keyword search (default: 0.4).
        k: Smoothing constant to dampen extreme high ranks (standard is 60).
        
    Returns:
        List of fused dicts sorted by combined RRF score.
    """
    rrf_scores = defaultdict(float)
    chunks_by_id = {}
    dense_ranks = {}
    bm25_ranks = {}

    # Accumulate dense ranks
    for rank, item in enumerate(dense_results, start=1):
        chunk = item["chunk"]
        cid = chunk.chunk_id
        chunks_by_id[cid] = chunk
        dense_ranks[cid] = rank
        rrf_scores[cid] += dense_weight * (1.0 / (k + rank))

    # Accumulate BM25 ranks
    for rank, item in enumerate(bm25_results, start=1):
        chunk = item["chunk"]
        cid = chunk.chunk_id
        chunks_by_id[cid] = chunk
        bm25_ranks[cid] = rank
        rrf_scores[cid] += bm25_weight * (1.0 / (k + rank))

    # Sort descending by fused score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for cid, combined_score in sorted_items:
        fused_results.append({
            "chunk": chunks_by_id[cid],
            "score": round(combined_score, 5),
            "dense_rank": dense_ranks.get(cid),
            "bm25_rank": bm25_ranks.get(cid),
            "source": "hybrid",
        })

    return fused_results


class HybridRetriever:
    """Orchestrates dense vector search and BM25 search."""

    def __init__(self, vector_store, bm25_index, dense_weight=0.6, bm25_weight=0.4):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query, top_k=4, dense_k=10, bm25_k=10):
        """Retrieve the best matching chunks using hybrid search.
        
        Returns:
            Dict containing:
              - 'results': top_k fused results
              - 'dense_results': raw vector results
              - 'bm25_results': raw BM25 results
        """
        dense_hits = self.vector_store.search(query, top_k=dense_k)
        bm25_hits = self.bm25_index.search(query, top_k=bm25_k)

        fused = reciprocal_rank_fusion(
            dense_results=dense_hits,
            bm25_results=bm25_hits,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )

        return {
            "results": fused[:top_k],
            "dense_results": dense_hits,
            "bm25_results": bm25_hits,
        }
