"""Child/parent chunks, a ranked RetrievalHit, and the RAGResult the UI stores."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from biomed_rag.models.documents import ExtractedStat


class ChildChunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentChunk(BaseModel):
    parent_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_stats: list[ExtractedStat] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    parent_text: str
    score: float = 0.0
    source: Literal["dense", "bm25", "ensemble", "rerank"] = "ensemble"
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_stats: list[ExtractedStat] = Field(default_factory=list)


class RAGResult(BaseModel):
    question: str
    answer: str
    refused: bool = False
    refusal_reason: str | None = None
    citations: list[str] = Field(default_factory=list)
    dois: list[str] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = 0.0
    disclaimer: str = ""
