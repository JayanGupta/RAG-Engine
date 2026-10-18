"""Lightweight RAG evaluation metrics: Faithfulness and Answer Relevancy.

Evaluation answers the core question: 'Is my RAG system working well?'
- Faithfulness: Does the answer stick to the facts present in the retrieved context?
- Answer Relevancy: Does the answer actually address what the user asked?
"""

import re

WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _extract_keywords(text):
    """Extract lowercase keywords of 3+ letters."""
    if not text:
        return set()
    return set(WORD_RE.findall(text.lower()))


def evaluate_faithfulness(answer, retrieved_items):
    """Compute lexical faithfulness of an answer against retrieved context.
    
    Returns a score between 0.0 and 1.0 representing what percentage of
    key terms in the answer are supported by the retrieved context.
    """
    if not answer or not retrieved_items:
        return 0.0

    context_text = " ".join(item["chunk"].text for item in retrieved_items)
    answer_words = _extract_keywords(answer)
    context_words = _extract_keywords(context_text)

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    score = len(overlap) / len(answer_words)
    return round(min(1.0, score), 3)


def evaluate_relevancy(question, answer):
    """Compute lexical relevancy between question and answer."""
    if not question or not answer:
        return 0.0

    q_words = _extract_keywords(question)
    a_words = _extract_keywords(answer)

    if not q_words or not a_words:
        return 0.0

    overlap = q_words.intersection(a_words)
    score = len(overlap) / len(q_words)
    return round(min(1.0, score), 3)


def evaluate_response(question, answer, retrieved_items):
    """Evaluate a complete RAG query result."""
    faithfulness = evaluate_faithfulness(answer, retrieved_items)
    relevancy = evaluate_relevancy(question, answer)
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "overall_score": round((faithfulness + relevancy) / 2.0, 3),
    }
