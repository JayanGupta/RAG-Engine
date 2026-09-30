"""Vector, keyword, and parent-section stores.

factory.build_vector_store() opens Qdrant (or Chroma), BM25, and parents.jsonl.
indexer.index_documents() chunks SourceDocuments and writes all three stores.
"""

from biomed_rag.store.factory import VectorStoreBundle, build_vector_store
from biomed_rag.store.indexer import index_documents

__all__ = ["VectorStoreBundle", "build_vector_store", "index_documents"]
