"""Paper and table records written to documents.jsonl / tables/."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractedStat(BaseModel):
    metric: Literal[
        "hr",
        "or",
        "p_value",
        "n",
        "dose",
        "ae_grade",
        "median_os",
        "median_pfs",
        "other",
    ] = "other"
    value: str
    arm: str | None = None
    comparator: str | None = None
    unit: str | None = None
    table_id: str | None = None


class TableBlob(BaseModel):
    table_id: str
    doc_id: str
    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    html: str = ""


class SourceDocument(BaseModel):
    doc_id: str
    source: Literal["pubmed", "pmc", "clinicaltrials"]
    pmid: str | None = None
    doi: str | None = None
    nct_id: str | None = None
    title: str
    year: int | None = None
    journal: str | None = None
    section: str = "abstract"
    text: str
    extracted_stats: list[ExtractedStat] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    therapeutic_area: str = "oncology_immunotherapy"
    extra: dict[str, Any] = Field(default_factory=dict)
