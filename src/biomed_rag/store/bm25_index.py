"""Keyword index over child chunks (drug names, PMIDs, NCT ids, acronyms)."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from biomed_rag.models.retrieval import ChildChunk, RetrievalHit

TOKEN_RE = re.compile(r"[A-Za-z0-9+\-./]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class BM25Index:
    chunks: list[ChildChunk] = field(default_factory=list)
    corpus_tokens: list[list[str]] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def build(self, chunks: list[ChildChunk]) -> None:
        self.chunks = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            keyword = " ".join(
                str(chunk.metadata.get(k, "") or "")
                for k in ("title", "pmid", "nct_id", "section", "keyword_field")
            )
            self.corpus_tokens.append(tokenize(chunk.text + " " + keyword))
        self._bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, k: int = 20) -> list[RetrievalHit]:
        if not self._bm25 or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        hits: list[RetrievalHit] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    parent_id=chunk.parent_id,
                    text=chunk.text,
                    parent_text="",
                    score=float(score),
                    source="bm25",
                    metadata=chunk.metadata,
                )
            )
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self.chunks],
            "corpus_tokens": self.corpus_tokens,
        }
        path.write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        index = cls()
        if not path.exists():
            return index
        payload = pickle.loads(path.read_bytes())
        chunks = [ChildChunk.model_validate(c) for c in payload.get("chunks", [])]
        index.chunks = chunks
        index.corpus_tokens = payload.get("corpus_tokens") or [tokenize(c.text) for c in chunks]
        index._bm25 = BM25Okapi(index.corpus_tokens) if index.corpus_tokens else None
        return index
