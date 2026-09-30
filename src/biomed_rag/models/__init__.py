"""Shared pydantic shapes: papers, chunks, hits, QA items, RAGResult."""

from biomed_rag.models.documents import ExtractedStat, SourceDocument, TableBlob
from biomed_rag.models.qa import ClinicalQA
from biomed_rag.models.retrieval import ChildChunk, ParentChunk, RetrievalHit, RAGResult

__all__ = [
    "ExtractedStat",
    "SourceDocument",
    "TableBlob",
    "ClinicalQA",
    "ChildChunk",
    "ParentChunk",
    "RetrievalHit",
    "RAGResult",
]
