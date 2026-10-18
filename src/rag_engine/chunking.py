"""Text chunking strategies for the RAG engine.

Splitting text into smaller chunks is essential in RAG because:
1. Embedding models have context length limits (typically 256-512 tokens).
2. Small chunks provide high-precision retrieval matching.
3. Overlap between chunks ensures ideas aren't cut in half at boundaries.
"""

import re


class TextChunk:
    """Represents a chunk of text derived from a source document."""

    def __init__(self, chunk_id, doc_id, text, title=None, chunk_index=0, parent_text=None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text.strip()
        self.title = title or ""
        self.chunk_index = chunk_index
        # Parent text is the broader surrounding context (optional parent-child retrieval)
        self.parent_text = parent_text or self.text

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "parent_text": self.parent_text,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            chunk_id=data.get("chunk_id"),
            doc_id=data.get("doc_id"),
            text=data.get("text", ""),
            title=data.get("title", ""),
            chunk_index=data.get("chunk_index", 0),
            parent_text=data.get("parent_text"),
        )

    def __repr__(self):
        return f"<TextChunk id={self.chunk_id} doc={self.doc_id} len={len(self.text)}>"


def split_text_into_chunks(text, chunk_size=400, chunk_overlap=50):
    """Split a string of text into overlapping word chunks.
    
    Args:
        text: The string content to split.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.
        
    Returns:
        List of string chunks.
    """
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks


def chunk_document(document, chunk_size=400, chunk_overlap=50):
    """Chunk a Document into TextChunk objects.
    
    Args:
        document: A Document instance.
        chunk_size: Words per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of TextChunk instances.
    """
    text_chunks = split_text_into_chunks(
        document.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = []
    for idx, piece in enumerate(text_chunks):
        cid = f"{document.doc_id}_chunk_{idx}"
        chunk = TextChunk(
            chunk_id=cid,
            doc_id=document.doc_id,
            text=piece,
            title=document.title,
            chunk_index=idx,
            parent_text=document.text,  # Keep full document text as parent context
        )
        chunks.append(chunk)
    return chunks


def chunk_documents(documents, chunk_size=400, chunk_overlap=50):
    """Chunk multiple Document objects."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
