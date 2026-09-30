"""Medical groundedness: citations subset of retrieved PMIDs, plus numeric overlap."""

from __future__ import annotations

import re

from biomed_rag.generation.citations import parse_pmids
from biomed_rag.models.qa import ClinicalQA
from biomed_rag.models.retrieval import RAGResult

NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def medical_groundedness(result: RAGResult, item: ClinicalQA) -> float:
    if item.expected_behavior == "refuse_oos":
        return 1.0 if result.refused and result.refusal_reason == "out_of_scope" else 0.0
    if item.expected_behavior == "refuse_dx":
        return 1.0 if result.refused and result.refusal_reason == "diagnostic" else 0.0
    if result.refused:
        return 0.0
    retrieved = {str(h.metadata.get("pmid")) for h in result.hits if h.metadata.get("pmid")}
    cited = set(result.citations or parse_pmids(result.answer))
    if not cited:
        return 0.0
    citation_score = 1.0 if cited.issubset(retrieved) else 0.5 if cited & retrieved else 0.0
    expected = set(item.evidence_pmids)
    evidence_score = 1.0 if not expected or (cited & expected) or (retrieved & expected) else 0.4
    numeric_score = 1.0
    stats_values = []
    for hit in result.hits:
        stats_values.extend(s.value for s in hit.extracted_stats)
    answer_nums = set(NUM_RE.findall(result.answer))
    if stats_values and answer_nums:
        if any(v in result.answer for v in stats_values):
            numeric_score = 1.0
        elif any(v in answer_nums for v in stats_values):
            numeric_score = 1.0
        else:
            numeric_score = 0.6
    return round((0.5 * citation_score) + (0.3 * evidence_score) + (0.2 * numeric_score), 4)
