"""PMID / DOI parsers and citation grounding.

validate_citations() keeps PMIDs that appear on retrieved hits and strips the rest.
"""

from __future__ import annotations

import re

from biomed_rag.models.retrieval import RetrievalHit

PMID_RE = re.compile(r"\[PMID:(\d{5,9})\]", re.I)
BARE_PMID_RE = re.compile(r"\bPMID[:\s]+(\d{5,9})\b", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


def parse_pmids(text: str) -> list[str]:
    found = PMID_RE.findall(text or "") + BARE_PMID_RE.findall(text or "")
    out: list[str] = []
    seen: set[str] = set()
    for pmid in found:
        if pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


def parse_dois(text: str) -> list[str]:
    found = DOI_RE.findall(text or "")
    out: list[str] = []
    seen: set[str] = set()
    for doi in found:
        if doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def retrieved_pmids(hits: list[RetrievalHit]) -> set[str]:
    values: set[str] = set()
    for hit in hits:
        pmid = hit.metadata.get("pmid")
        if pmid:
            values.add(str(pmid))
    return values


def validate_citations(answer: str, hits: list[RetrievalHit]) -> tuple[str, list[str], list[str]]:
    allowed = retrieved_pmids(hits)
    cited = parse_pmids(answer)
    valid = [p for p in cited if p in allowed]
    invalid = [p for p in cited if p not in allowed]
    cleaned = answer
    for pmid in invalid:
        cleaned = re.sub(rf"\[PMID:{pmid}\]", "", cleaned, flags=re.I)
        cleaned = re.sub(rf"PMID[:\s]+{pmid}", "", cleaned, flags=re.I)
    dois = parse_dois(cleaned)
    return cleaned.strip(), valid, dois
