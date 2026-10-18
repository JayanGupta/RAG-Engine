"""System prompt and fallback messages for the RAG pipeline."""

SYSTEM_PROMPT = """You are a helpful knowledge assistant. Answer questions using ONLY the provided context.

Rules:
1. Use only the provided context to answer. If the context is insufficient, say so.
2. Cite your sources using [Source: title] format after each claim.
3. Never invent facts, sources, or data that aren't in the context.
4. Be concise and direct.

Context:
{context}

Question:
{question}

Answer using the context above, citing sources."""

INSUFFICIENT_EVIDENCE = (
    "I don't have enough information in my knowledge base to answer this question. "
    "Try uploading more documents or rephrasing your query."
)

REFUSAL_MESSAGE = (
    "This query was blocked by the safety filter. "
    "Please rephrase your question."
)

SYNTHETIC_QA_PROMPT = """Generate an evaluation question grounded in the supplied documents.

Return a JSON object with: id, question, answer, evidence_doc_ids, qtype.
evidence_doc_ids must come from the documents. Do not invent facts.

Question type: {qtype}
Documents:
{documents}
"""
