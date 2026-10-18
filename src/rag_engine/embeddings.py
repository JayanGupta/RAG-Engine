"""Embedding functions for converting text to semantic vector representations.

Embeddings map human text into high-dimensional vectors such that texts with
similar meanings have high cosine similarity (close to 1.0).
"""

import hashlib
import math


class HashEmbeddings:
    """A deterministic, zero-dependency embedding generator.
    
    Uses SHA-256 token hashing into 384 buckets.
    Great for instant offline demos, testing, and machines without GPUs or PyTorch.
    """

    def __init__(self, dim=384):
        self.dim = dim

    def embed_text(self, text):
        """Convert a single string into a normalized float vector."""
        vector = [0.0] * self.dim
        if not text:
            return vector

        tokens = text.lower().split()
        for idx, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            # Weigh by position and presence
            vector[bucket] += 1.0 + (idx % 5) * 0.05

        # L2 normalize
        squared_sum = sum(v * v for v in vector)
        norm = math.sqrt(squared_sum) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        """Batch embed a list of texts."""
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query):
        """Embed a search query."""
        return self.embed_text(query)


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two unit-normalized vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    return float(dot_product)


def get_embedding_model(settings=None):
    """Factory to get the configured embedding model."""
    if settings is None or getattr(settings, "use_hash_embeddings", True):
        return HashEmbeddings()

    # If requested and available, use SentenceTransformers
    try:
        from sentence_transformers import SentenceTransformer

        class STEmbeddings:
            def __init__(self, model_name):
                self.model = SentenceTransformer(model_name)

            def embed_text(self, text):
                return self.model.encode(text, normalize_embeddings=True).tolist()

            def embed_documents(self, texts):
                return self.model.encode(texts, normalize_embeddings=True).tolist()

            def embed_query(self, query):
                return self.embed_text(query)

        return STEmbeddings(settings.embedding_model)
    except Exception:
        # Fall back to HashEmbeddings if sentence-transformers is missing
        return HashEmbeddings()
