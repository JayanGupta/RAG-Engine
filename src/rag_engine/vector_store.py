"""Vector Store implementation for indexing and semantic similarity search.

A vector database stores chunks alongside their embedding vectors and finds
the nearest neighbors (most relevant chunks) using cosine similarity.
"""

from rag_engine.embeddings import cosine_similarity, get_embedding_model


class VectorStore:
    """Simple in-memory vector store with cosine similarity ranking."""

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or get_embedding_model()
        self.chunks = []
        self.vectors = []

    def add_chunks(self, chunks):
        """Add TextChunk objects and calculate their embeddings."""
        if not chunks:
            return
        new_texts = [c.text for c in chunks]
        new_vectors = self.embedding_model.embed_documents(new_texts)
        self.chunks.extend(chunks)
        self.vectors.extend(new_vectors)

    def search(self, query, top_k=5):
        """Find the top_k most semantically similar chunks for a query.
        
        Returns:
            List of dicts: [{"chunk": TextChunk, "score": float, "source": "dense"}]
        """
        if not self.chunks:
            return []

        query_vector = self.embedding_model.embed_query(query)
        scored_results = []

        for idx, (chunk, doc_vector) in enumerate(zip(self.chunks, self.vectors)):
            similarity = cosine_similarity(query_vector, doc_vector)
            scored_results.append({
                "chunk": chunk,
                "score": round(similarity, 4),
                "source": "dense",
            })

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def clear(self):
        """Empty the store."""
        self.chunks = []
        self.vectors = []

    def __len__(self):
        return len(self.chunks)
