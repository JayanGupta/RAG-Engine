"""RAG Engine: A clear, educational Retrieval-Augmented Generation system.

Designed to illustrate how a production-grade RAG pipeline works:
- Ingestion & Text Extraction (PDF / Text)
- Semantic Text Chunking
- Dense Vector Embeddings (with offline Hash fallback)
- Sparse BM25 Keyword Search
- Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
- Augmented Generation with Source Citations
- Faithfulness and Relevancy Evaluation
"""

from rag_engine.documents import Document, load_text_file, load_pdf_file
from rag_engine.chunking import TextChunk, chunk_document, chunk_documents
from rag_engine.embeddings import HashEmbeddings, get_embedding_model
from rag_engine.vector_store import VectorStore
from rag_engine.bm25_search import BM25SearchIndex
from rag_engine.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from rag_engine.generator import generate_answer
from rag_engine.eval import evaluate_faithfulness, evaluate_relevancy, evaluate_response
from rag_engine.pipeline import RAGPipeline

__all__ = [
    "Document",
    "load_text_file",
    "load_pdf_file",
    "TextChunk",
    "chunk_document",
    "chunk_documents",
    "HashEmbeddings",
    "get_embedding_model",
    "VectorStore",
    "BM25SearchIndex",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "generate_answer",
    "evaluate_faithfulness",
    "evaluate_relevancy",
    "evaluate_response",
    "RAGPipeline",
]
