"""Complete end-to-end RAG Pipeline orchestrator.

Coordinates all 5 core stages of Retrieval-Augmented Generation:
1. Ingestion: Load raw text, PDFs, or demo articles into Document objects.
2. Chunking: Split documents into smaller, overlapping TextChunk passages.
3. Indexing: Index chunks into both Dense Vector Store and BM25 Keyword Search.
4. Hybrid Retrieval: Retrieve and rank chunks using Reciprocal Rank Fusion (RRF).
5. Generation: Augment prompt with retrieved chunks and generate answer with LLM.
"""

import time
from pathlib import Path

from config.settings import get_settings
from rag_engine.documents import Document, load_text_file, load_pdf_file, save_documents_jsonl, load_documents_jsonl
from rag_engine.chunking import chunk_documents
from rag_engine.vector_store import VectorStore
from rag_engine.bm25_search import BM25SearchIndex
from rag_engine.hybrid_search import HybridRetriever
from rag_engine.generator import generate_answer
from rag_engine.eval import evaluate_response


class RAGPipeline:
    """The central RAG engine coordinating ingestion, search, and generation."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.documents = []
        self.chunks = []
        self.vector_store = VectorStore()
        self.bm25_index = BM25SearchIndex()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            dense_weight=self.settings.dense_weight,
            bm25_weight=self.settings.bm25_weight,
        )

    def add_document(self, text, title=None, doc_id=None, source="upload"):
        """Add a single raw text document and index it."""
        doc = Document(text=text, title=title, doc_id=doc_id, source=source)
        self.documents.append(doc)
        new_chunks = chunk_documents(
            [doc],
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.chunks.extend(new_chunks)
        self.vector_store.add_chunks(new_chunks)
        self.bm25_index.build(self.chunks)
        save_documents_jsonl(self.documents, self.settings.documents_file)
        return doc

    def add_pdf(self, filepath):
        """Extract text from a PDF file and index it."""
        doc = load_pdf_file(filepath)
        return self.add_document(text=doc.text, title=doc.title, doc_id=doc.doc_id, source=doc.source)

    def load_demo_corpus(self):
        """Load bundled educational demo articles."""
        demo_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo_corpus"
        if not demo_dir.exists():
            return []

        loaded = []
        for txt_file in sorted(demo_dir.glob("*.txt")):
            doc = load_text_file(txt_file)
            self.documents.append(doc)
            loaded.append(doc)

        if loaded:
            self.chunks = chunk_documents(
                self.documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self.vector_store.clear()
            self.vector_store.add_chunks(self.chunks)
            self.bm25_index.build(self.chunks)
            save_documents_jsonl(self.documents, self.settings.documents_file)

        return loaded

    def query(self, question, top_k=None):
        """Execute the end-to-end RAG workflow for a user question.
        
        Steps:
        1. Query dense vector store and BM25 keyword index.
        2. Merge results using Reciprocal Rank Fusion.
        3. Format retrieved context into LLM prompt.
        4. Synthesize final answer with source citations.
        5. Evaluate faithfulness and relevancy.
        
        Returns:
            Dict containing:
              - 'question': input question
              - 'answer': generated answer
              - 'sources': cited document titles
              - 'retrieved_chunks': top ranked chunks
              - 'dense_results': vector search results
              - 'bm25_results': keyword search results
              - 'evaluation': faithfulness & relevancy metrics
              - 'latency_ms': elapsed execution time
        """
        start_time = time.perf_counter()
        top_k = top_k or self.settings.final_top_k

        # Step 1: Hybrid Retrieval
        retrieval_output = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            dense_k=self.settings.dense_top_k,
            bm25_k=self.settings.bm25_top_k,
        )

        retrieved_chunks = retrieval_output["results"]

        # Step 2: Generation
        gen_output = generate_answer(
            question=question,
            retrieved_items=retrieved_chunks,
            settings=self.settings,
        )

        # Step 3: Evaluation
        evaluation = evaluate_response(
            question=question,
            answer=gen_output["answer"],
            retrieved_items=retrieved_chunks,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "question": question,
            "answer": gen_output["answer"],
            "sources": gen_output["sources"],
            "used_llm": gen_output["used_llm"],
            "retrieved_chunks": retrieved_chunks,
            "dense_results": retrieval_output["dense_results"],
            "bm25_results": retrieval_output["bm25_results"],
            "evaluation": evaluation,
            "latency_ms": elapsed_ms,
        }

    def get_stats(self):
        """Return summary statistics of the indexed knowledge base."""
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "vector_store_size": len(self.vector_store),
            "bm25_index_size": len(self.bm25_index),
        }
