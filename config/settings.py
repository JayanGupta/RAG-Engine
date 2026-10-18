"""Configuration settings for the RAG engine.

Loads environment variables from .env or uses sensible defaults.
All settings can be customized via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings:
    """Central configuration for paths, models, and retrieval hyperparameters."""

    def __init__(self):
        # LLM configuration
        self.user_llm_api_key = os.getenv("USER_LLM_API_KEY", "").strip()
        self.user_llm_base_url = os.getenv("USER_LLM_BASE_URL", "")
        self.user_llm_model = os.getenv("USER_LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1000"))

        # Embeddings & Vector Store
        # HashEmbeddings = 100% offline, zero-dependency token hashing for demo/testing
        self.use_hash_embeddings = os.getenv("USE_HASH_EMBEDDINGS", "true").lower() == "true"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.collection_name = "rag_engine_docs"

        # Chunking parameters
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "400"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", "1500"))

        # Retrieval hyperparameters
        self.dense_top_k = int(os.getenv("DENSE_TOP_K", "10"))
        self.bm25_top_k = int(os.getenv("BM25_TOP_K", "10"))
        self.final_top_k = int(os.getenv("FINAL_TOP_K", "4"))
        self.dense_weight = float(os.getenv("DENSE_WEIGHT", "0.6"))
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", "0.4"))

        # Storage paths
        self.data_dir = ROOT / "data"
        self.documents_file = self.data_dir / "documents.jsonl"
        self.chroma_dir = self.data_dir / "chroma"
        self.bm25_file = self.data_dir / "bm25_index.pkl"
        self.parent_store_file = self.data_dir / "parents.jsonl"

        # Ensure base directories exist
        self.ensure_dirs()

    def ensure_dirs(self):
        """Create necessary data storage directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def has_llm_key(self):
        """Check if a valid LLM API key has been provided."""
        return bool(self.user_llm_api_key) and self.user_llm_api_key != "your-api-key-here"


_cached_settings = None


def get_settings():
    """Returns a singleton Settings instance."""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings()
    return _cached_settings
