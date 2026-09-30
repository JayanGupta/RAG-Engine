"""Query pipeline: guardrail -> retrieve -> generate -> cite -> disclaimer.

Public entry: run_rag(question). Streamlit and eval both call this.
Without USER_LLM_API_KEY, generation uses extractive_fallback() instead of ChatOpenAI.
"""

from __future__ import annotations

import json
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config.prompts import (
    CLINICAL_DISCLAIMER,
    INSUFFICIENT_EVIDENCE_TEMPLATE,
    SYSTEM_PROMPT,
)
from config.settings import Settings, get_settings
from biomed_rag.generation.citations import validate_citations
from biomed_rag.guardrails.input_filter import GuardrailDecision, filter_query
from biomed_rag.guardrails.output_policy import apply_output_policy
from biomed_rag.models.retrieval import RAGResult, RetrievalHit
from biomed_rag.observability.tracing import configure_tracing
from biomed_rag.retrieval.hybrid import HybridRetriever
from biomed_rag.retrieval.rerank import rerank_hits
from biomed_rag.store.factory import VectorStoreBundle, build_vector_store


def format_context(hits: list[RetrievalHit]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        stats = [s.model_dump() for s in hit.extracted_stats[:12]]
        blocks.append(
            "\n".join(
                [
                    f"[Source {i}]",
                    f"Title: {meta.get('title')}",
                    f"PMID: {meta.get('pmid')}",
                    f"DOI: {meta.get('doi')}",
                    f"NCT: {meta.get('nct_id')}",
                    f"Year: {meta.get('year')}",
                    f"Section: {meta.get('section')}",
                    f"Extracted stats JSON: {json.dumps(stats)}",
                    f"Text: {hit.parent_text or hit.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_chat_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.user_llm_api_key or "missing",
        base_url=settings.user_llm_base_url,
        model=settings.user_llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def build_chain_inputs(question: str, hits: list[RetrievalHit]) -> dict:
    return {"question": question, "context": format_context(hits), "hits": hits}


def extractive_fallback(question: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return INSUFFICIENT_EVIDENCE_TEMPLATE
    lines = [
        f"Literature synthesis based on indexed sources for: {question}",
        "",
    ]
    for hit in hits[:3]:
        pmid = hit.metadata.get("pmid") or "n/a"
        title = hit.metadata.get("title") or "Untitled"
        snippet = (hit.parent_text or hit.text)[:420]
        stats = ", ".join(
            f"{s.metric}={s.value}{(' ' + s.unit) if s.unit else ''}"
            for s in hit.extracted_stats[:4]
        )
        extra = f" Extracted: {stats}." if stats else ""
        lines.append(f"- {title} [PMID:{pmid}]. {snippet}{extra}")
    return "\n".join(lines)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def refuse_blocked_query(question: str, decision: GuardrailDecision, started: float) -> RAGResult:
    return RAGResult(
        question=question,
        answer=decision.message or "",
        refused=True,
        refusal_reason=decision.reason,
        latency_ms=_elapsed_ms(started),
        disclaimer=CLINICAL_DISCLAIMER,
    )


def insufficient_evidence_result(question: str, started: float, hits: list[RetrievalHit] | None = None) -> RAGResult:
    return RAGResult(
        question=question,
        answer=INSUFFICIENT_EVIDENCE_TEMPLATE,
        refused=False,
        hits=hits or [],
        latency_ms=_elapsed_ms(started),
        disclaimer=CLINICAL_DISCLAIMER,
    )


def retrieve_and_rerank(
    question: str,
    settings: Settings,
    bundle: VectorStoreBundle | None,
) -> tuple[VectorStoreBundle, list[RetrievalHit]]:
    bundle = bundle or build_vector_store(settings)
    dense_and_sparse_hits = HybridRetriever(bundle, settings).retrieve(question)
    ranked_parents = rerank_hits(question, dense_and_sparse_hits, settings)
    return bundle, ranked_parents


def synthesize_answer(question: str, hits: list[RetrievalHit], settings: Settings) -> str:
    if not settings.llm_configured():
        return extractive_fallback(question, hits)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    chain = (
        RunnablePassthrough.assign(context=lambda x: format_context(x["hits"]))
        | prompt
        | build_chat_llm(settings)
        | StrOutputParser()
    )
    try:
        return chain.invoke({"question": question, "hits": hits})
    except Exception:
        return extractive_fallback(question, hits)


def ground_answer(answer: str, hits: list[RetrievalHit]) -> tuple[str, list[str], list[str]]:
    cleaned, valid_pmids, dois = validate_citations(answer, hits)
    if not valid_pmids:
        return INSUFFICIENT_EVIDENCE_TEMPLATE, [], dois
    return apply_output_policy(cleaned, valid_pmids, refused=False), valid_pmids, dois


def run_rag(
    question: str,
    settings: Settings | None = None,
    bundle: VectorStoreBundle | None = None,
) -> RAGResult:
    settings = settings or get_settings()
    configure_tracing(settings)
    started = time.perf_counter()

    decision = filter_query(question)
    if not decision.allowed:
        return refuse_blocked_query(question, decision, started)

    _bundle, hits = retrieve_and_rerank(question, settings, bundle)
    if not hits:
        return insufficient_evidence_result(question, started)

    draft = synthesize_answer(question, hits, settings)
    final, valid_pmids, dois = ground_answer(draft, hits)
    return RAGResult(
        question=question,
        answer=final,
        refused=False,
        citations=valid_pmids,
        dois=dois,
        hits=hits,
        latency_ms=_elapsed_ms(started),
        disclaimer=CLINICAL_DISCLAIMER,
    )


def guardrail_runnable() -> RunnableLambda:
    def _inner(payload: dict) -> dict:
        question = payload.get("question", "")
        decision = filter_query(question)
        payload["guardrail"] = decision
        return payload

    return RunnableLambda(_inner)
