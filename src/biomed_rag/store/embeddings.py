"""Dense encoders for child chunks.

Default is HashEmbeddings (offline, no model download). Set USE_HASH_EMBEDDINGS=false
to load BGE or MedCPT from Hugging Face.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config.settings import Settings


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        if not text:
            return values
        tokens = text.lower().split()
        for i, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            values[bucket] += 1.0 + (i % 7) * 0.01
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@lru_cache(maxsize=4)
def _hf_embeddings(model_name: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(settings: Settings, allow_hash_fallback: bool = True) -> Embeddings:
    if settings.use_hash_embeddings:
        return HashEmbeddings()
    if settings.use_medcpt:
        model = "ncbi/MedCPT-Query-Encoder"
    else:
        model = settings.embedding_model
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model, local_files_only=True)
        return _hf_embeddings(model)
    except Exception:
        if allow_hash_fallback:
            return HashEmbeddings()
        raise
