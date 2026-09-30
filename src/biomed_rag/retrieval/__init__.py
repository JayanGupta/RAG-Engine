"""Search: child chunks for ranking, parent sections for generation.

chunking.split_documents -> HybridRetriever.retrieve -> rerank.rerank_hits
"""

from biomed_rag.retrieval.hybrid import HybridRetriever

__all__ = ["HybridRetriever"]
