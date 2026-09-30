"""Split SourceDocuments into child search chunks and parent generation chunks.

split_documents() is the only public function. Children are ~400 tokens; parents
are the full section or ~2000 tokens.
"""

from __future__ import annotations

import hashlib
import json
import re

from config.settings import Settings
from biomed_rag.models.documents import ExtractedStat, SourceDocument
from biomed_rag.models.retrieval import ChildChunk, ParentChunk

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _chunk_id(parent_id: str, idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{parent_id}:c{idx}:{digest}"


def semantic_split(text: str, target_tokens: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if _approx_tokens(text) <= target_tokens:
        return [text]
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        words = text.split()
        return [
            " ".join(words[i : i + target_tokens])
            for i in range(0, len(words), target_tokens)
        ]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sent_tokens = _approx_tokens(sentence)
        if current and current_tokens + sent_tokens > target_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sent_tokens
        else:
            current.append(sentence)
            current_tokens += sent_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _keyword_field(doc: SourceDocument) -> str:
    stats = " ".join(s.value for s in doc.extracted_stats)
    return " ".join(
        [
            doc.title,
            doc.pmid or "",
            doc.nct_id or "",
            " ".join(doc.trial_ids),
            " ".join(doc.mesh_terms),
            stats,
        ]
    )


def _stats_payload(stats: list[ExtractedStat]) -> str:
    return json.dumps([s.model_dump() for s in stats], ensure_ascii=True)


def split_documents(
    documents: list[SourceDocument],
    settings: Settings,
) -> tuple[list[ChildChunk], list[ParentChunk]]:
    children: list[ChildChunk] = []
    parents: list[ParentChunk] = []
    for doc in documents:
        parent_id = f"{doc.doc_id}:{doc.section}"
        parent_text = doc.text
        if _approx_tokens(parent_text) > settings.parent_chunk_tokens:
            parent_parts = semantic_split(parent_text, settings.parent_chunk_tokens)
        else:
            parent_parts = [parent_text]
        for p_idx, part in enumerate(parent_parts):
            pid = parent_id if len(parent_parts) == 1 else f"{parent_id}:p{p_idx}"
            metadata = {
                "doc_id": doc.doc_id,
                "pmid": doc.pmid,
                "doi": doc.doi,
                "nct_id": doc.nct_id,
                "title": doc.title,
                "year": doc.year,
                "section": doc.section,
                "parent_id": pid,
                "chunk_type": "parent",
                "source": doc.source,
                "trial_ids": doc.trial_ids,
                "extracted_stats": [s.model_dump() for s in doc.extracted_stats],
                "keyword_field": _keyword_field(doc),
            }
            parents.append(
                ParentChunk(
                    parent_id=pid,
                    text=part,
                    metadata=metadata,
                    extracted_stats=doc.extracted_stats,
                )
            )
            child_texts = semantic_split(part, settings.child_chunk_tokens)
            for c_idx, child_text in enumerate(child_texts):
                cid = _chunk_id(pid, c_idx, child_text)
                child_meta = dict(metadata)
                child_meta.update(
                    {
                        "chunk_id": cid,
                        "chunk_type": "child",
                        "extracted_stats_json": _stats_payload(doc.extracted_stats),
                    }
                )
                children.append(
                    ChildChunk(chunk_id=cid, parent_id=pid, text=child_text, metadata=child_meta)
                )
    return children, parents
