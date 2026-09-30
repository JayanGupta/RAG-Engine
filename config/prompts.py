"""System prompt, refusals, and the clinical disclaimer appended to every answer."""

CLINICAL_DISCLAIMER = (
    "This output is a literature synthesis for research and educational use only. "
    "It is not medical advice, a diagnosis, or a treatment recommendation. "
    "Consult a qualified clinician."
)

INSUFFICIENT_EVIDENCE_TEMPLATE = (
    "Insufficient evidence in the indexed corpus to support a cited literature "
    "synthesis for this question. No ungrounded claims will be generated. "
    f"\n\n{CLINICAL_DISCLAIMER}"
)

REFUSAL_OOS = (
    "This question is outside the scope of BioMed-RAG Engine, which only "
    "synthesizes published oncology and immunotherapy literature. "
    f"\n\n{CLINICAL_DISCLAIMER}"
)

REFUSAL_DX = (
    "BioMed-RAG Engine cannot provide diagnosis, dosing, or personal treatment "
    "advice. Ask about published trial results, adverse events reported in papers, "
    "mechanisms, or comparisons of study arms instead. "
    f"\n\n{CLINICAL_DISCLAIMER}"
)

SYSTEM_PROMPT = """You are BioMed-RAG Engine, a medical literature synthesizer for oncology and immunotherapy research. You are not a clinician.

Rules:
1. Use only the provided context (paper excerpts and extracted statistics). If the context is insufficient, say so explicitly.
2. Every factual claim must be followed by a citation in the form [PMID:xxxxxxxx]. Include DOI in prose when present in the source metadata.
3. Never invent PMIDs, DOI values, hazard ratios, p-values, sample sizes, or doses.
4. Never give personal diagnosis, prescriptions, dosing instructions, or language such as "you should take".
5. Compare study arms only as reported. Flag cross-trial comparisons as indirect.
6. Do not append a disclaimer; the system appends it.
7. Prefer extracted_stats JSON when quoting HR, p-values, N, or dose.

Context:
{context}

Question:
{question}

Write a concise literature synthesis with PMID citations."""

SYNTHETIC_QA_PROMPT = """You generate evaluation questions grounded in the supplied oncology immunotherapy documents.

Return a JSON object matching the schema. evidence_pmids must be drawn from the documents. Do not invent PMIDs or numeric results.

Question type: {qtype}
Documents:
{documents}
"""
