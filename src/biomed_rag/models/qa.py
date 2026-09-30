"""One synthetic or adversarial evaluation item."""

from typing import Literal

from pydantic import BaseModel, Field


class ClinicalQA(BaseModel):
    id: str
    question: str
    answer: str
    evidence_pmids: list[str] = Field(default_factory=list)
    qtype: Literal[
        "comparison",
        "adverse_event",
        "statistic",
        "mechanism",
        "acronym",
        "adversarial_oos",
        "adversarial_dx",
    ]
    adversarial: bool = False
    expected_behavior: Literal["answer", "refuse_oos", "refuse_dx"] = "answer"
