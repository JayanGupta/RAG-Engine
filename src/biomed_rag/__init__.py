"""BioMed-RAG Engine: oncology / immunotherapy literature synthesis.

Two paths, same modules:

Index (offline):
  ingestion.pipeline -> retrieval.chunking -> store.indexer
  writes Qdrant/Chroma children, BM25, and parents.jsonl

Query (Ask page):
  1. guardrails.input_filter   block diagnosis / out-of-scope
  2. retrieval.hybrid          dense + BM25, RRF merge
  3. retrieval.rerank          top 5 parents
  4. generation.chain          LLM or extractive fallback
  5. generation.citations      drop PMIDs not in retrieved hits
  6. guardrails.output_policy  strip treatment language, append disclaimer

Start at app.py, then generation/chain.py run_rag().
"""

__version__ = "1.0.0"
