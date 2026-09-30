"""Typed env knobs. Names are USER_LLM_* / NCBI_* only — never platform keys.

get_settings() is cached. Paths under data/ are created on first load.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    user_llm_api_key: str = Field(default="")
    user_llm_base_url: str = Field(default="https://api.example.com/v1")
    user_llm_model: str = Field(default="your-model-name")

    ncbi_email: str = Field(default="")
    ncbi_api_key: str = Field(default="")

    user_cohere_api_key: str = Field(default="")
    user_langsmith_api_key: str = Field(default="")
    user_langsmith_project: str = Field(default="biomed-rag-engine")

    vector_backend: str = Field(default="qdrant")
    qdrant_path: str = Field(default="data/qdrant")
    chroma_path: str = Field(default="data/chroma")
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")
    rerank_model: str = Field(default="BAAI/bge-reranker-base")
    use_medcpt: bool = Field(default=False)
    use_hash_embeddings: bool = Field(default=True)

    collection_name: str = Field(default="biomed_rag_oncology")
    pubmed_fetch_limit: int = Field(default=80)
    ctgov_fetch_limit: int = Field(default=40)
    dense_k: int = Field(default=20)
    bm25_k: int = Field(default=20)
    rerank_k: int = Field(default=5)
    ensemble_dense_weight: float = Field(default=0.6)
    ensemble_bm25_weight: float = Field(default=0.4)
    child_chunk_tokens: int = Field(default=400)
    parent_chunk_tokens: int = Field(default=2000)
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=1200)

    therapeutic_area: str = Field(default="oncology_immunotherapy")
    pubmed_query: str = Field(
        default=(
            '(immunotherapy OR "immune checkpoint" OR "PD-1" OR "PD-L1" OR '
            "pembrolizumab OR nivolumab OR atezolizumab) AND "
            "(cancer OR oncology OR NSCLC OR melanoma OR carcinoma) AND "
            "(clinical trial[pt] OR randomized)"
        )
    )
    pubmed_mindate: str = Field(default="2016/01/01")

    data_dir: Path = Field(default=ROOT / "data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def documents_path(self) -> Path:
        return self.processed_dir / "documents.jsonl"

    @property
    def tables_dir(self) -> Path:
        return self.processed_dir / "tables"

    @property
    def bm25_path(self) -> Path:
        return self.processed_dir / "bm25.pkl"

    @property
    def parent_store_path(self) -> Path:
        return self.processed_dir / "parents.jsonl"

    @property
    def qa_path(self) -> Path:
        return self.synthetic_dir / "qa.jsonl"

    @property
    def ragas_report_path(self) -> Path:
        return self.eval_dir / "ragas_report.json"

    @property
    def qdrant_abs_path(self) -> Path:
        path = Path(self.qdrant_path)
        return path if path.is_absolute() else ROOT / path

    @property
    def chroma_abs_path(self) -> Path:
        path = Path(self.chroma_path)
        return path if path.is_absolute() else ROOT / path

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir / "pubmed",
            self.raw_dir / "pmc",
            self.raw_dir / "clinicaltrials",
            self.processed_dir,
            self.tables_dir,
            self.synthetic_dir,
            self.eval_dir,
            self.qdrant_abs_path,
            self.chroma_abs_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def llm_configured(self) -> bool:
        key = self.user_llm_api_key.strip()
        return bool(key) and key != "your-api-key-here"

    def ncbi_rate_limit(self) -> float:
        return 0.11 if self.ncbi_api_key else 0.34


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
