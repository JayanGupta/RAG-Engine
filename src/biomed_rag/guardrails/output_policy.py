"""Post-generation policy: redact treatment directives, append disclaimer once."""

from __future__ import annotations

import re

from config.prompts import CLINICAL_DISCLAIMER, INSUFFICIENT_EVIDENCE_TEMPLATE

TREATMENT_RE = re.compile(
    r"\b(you should take|i recommend you take|start taking|increase your dose|"
    r"stop your medication|this means you have)\b",
    re.I,
)


def apply_output_policy(
    answer: str,
    valid_pmids: list[str],
    refused: bool = False,
) -> str:
    text = (answer or "").strip()
    if not refused:
        text = TREATMENT_RE.sub("[redacted non-literature directive]", text)
        if valid_pmids is not None and not valid_pmids:
            return INSUFFICIENT_EVIDENCE_TEMPLATE
    text = re.sub(
        r"\n*This output is a literature synthesis[\s\S]*$",
        "",
        text,
        flags=re.I,
    ).strip()
    return f"{text}\n\n{CLINICAL_DISCLAIMER}"
