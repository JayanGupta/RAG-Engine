"""Document representations and file loaders for the RAG engine.

Zero external complexity: clean, readable objects representing documents.
"""

import json
import hashlib
from pathlib import Path


class Document:
    """Represents a source document in the RAG system."""

    def __init__(self, text, doc_id=None, title=None, source="upload", metadata=None):
        self.text = text.strip() if text else ""
        self.doc_id = doc_id or hashlib.md5(self.text[:100].encode("utf-8")).hexdigest()[:12]
        self.title = title or f"Document-{self.doc_id[:6]}"
        self.source = source
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            text=data.get("text", ""),
            doc_id=data.get("doc_id"),
            title=data.get("title"),
            source=data.get("source", "upload"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self):
        return f"<Document id={self.doc_id} title='{self.title}'>"


def load_text_file(filepath):
    """Load a plain text or markdown file into a Document."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8").strip()
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return Document(text=text, doc_id=path.stem, title=title, source=path.name)


def load_pdf_file(filepath):
    """Extract text from a PDF file into a Document."""
    path = Path(filepath)
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
    except Exception:
        text = f"Failed to extract text from {path.name}"
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return Document(text=text, doc_id=path.stem, title=title, source=path.name)


def save_documents_jsonl(documents, filepath):
    """Save a list of Document objects to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict()) + "\n")


def load_documents_jsonl(filepath):
    """Load Document objects from a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        return []
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document.from_dict(json.loads(line)))
    return docs
