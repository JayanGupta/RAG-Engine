"""Block diagnostic and non-medical questions before retrieval.

filter_query() returns GuardrailDecision. allowed=False means no search, no LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.prompts import REFUSAL_DX, REFUSAL_OOS

DX_PATTERNS = [
    r"\bdo i have\b",
    r"\bam i (dying|sick|cancer)\b",
    r"\bdiagnos(e|is|ing)\s+me\b",
    r"\bwhat should i take\b",
    r"\bshould i take\b",
    r"\bprescrib(e|e me|tion)\b",
    r"\bdose me\b",
    r"\binterpret my (labs?|blood|scan|mri|ct)\b",
    r"\bmy (symptoms?|labs?|tumor) \b",
    r"\btreat me\b",
    r"\bwhat medication should i\b",
    r"\bfor my cancer\b",
    r"\bact as (a |an )?(doctor|physician|oncologist)\b",
    r"\bignore (the )?(disclaimer|rules|guardrail)\b",
]

OOS_PATTERNS = [
    r"\bwrite (a |me )?(python|javascript|code)\b",
    r"\bleetcode\b",
    r"\bfootball\b|\bsoccer\b|\bnba\b|\bworld cup\b",
    r"\bcapital of\b",
    r"\bweather\b",
    r"\bcryptocurrency\b|\bbitcoin\b",
    r"\bhow to (hack|jailbreak)\b",
    r"\bmake a cake\b|\brecipe\b",
    r"\bhomework\b",
    r"\bwho won the\b",
]

MEDICAL_HINTS = [
    "trial",
    "pmid",
    "hazard",
    "pfs",
    "os",
    "pd-1",
    "pd-l1",
    "pembrolizumab",
    "nivolumab",
    "atezolizumab",
    "immunotherapy",
    "checkpoint",
    "nsclc",
    "melanoma",
    "adverse",
    "ae",
    "irae",
    "biomarker",
    "egfr",
    "kras",
    "ctla-4",
    "nct",
    "phase",
    "oncolog",
    "carcinoma",
    "chemotherapy",
    "dose",
    "mg/kg",
    "median",
    "randomized",
    "placebo",
    "arm",
]


@dataclass
class GuardrailDecision:
    allowed: bool
    reason: str | None = None
    message: str | None = None


def filter_query(query: str) -> GuardrailDecision:
    text = (query or "").strip()
    if not text:
        return GuardrailDecision(False, "empty", REFUSAL_OOS)
    lower = text.lower()
    for pattern in DX_PATTERNS:
        if re.search(pattern, lower):
            return GuardrailDecision(False, "diagnostic", REFUSAL_DX)
    for pattern in OOS_PATTERNS:
        if re.search(pattern, lower):
            return GuardrailDecision(False, "out_of_scope", REFUSAL_OOS)
    if any(hint in lower for hint in MEDICAL_HINTS):
        return GuardrailDecision(True)
    biomedical_tokens = re.findall(
        r"\b(cancer|tumor|tumour|drug|gene|mutation|antibody|inhibitor|therapy|survival)\b",
        lower,
    )
    if biomedical_tokens:
        return GuardrailDecision(True)
    return GuardrailDecision(False, "out_of_scope", REFUSAL_OOS)
