"""Streamlit UI. Pages call into src/biomed_rag; they do not implement RAG.

Ask -> generation.chain.run_rag
Ingest -> ingestion.pipeline.run_ingestion + store.indexer
Evaluation -> eval.ragas_bench.run_benchmark
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config.prompts import CLINICAL_DISCLAIMER
from config.settings import get_settings
from biomed_rag.eval.ragas_bench import run_benchmark
from biomed_rag.eval.synthetic_qa import generate_synthetic_qa, load_qa
from biomed_rag.generation.chain import run_rag
from biomed_rag.ingestion.parsers import read_jsonl
from biomed_rag.ingestion.pipeline import run_ingestion, seed_demo_tables
from biomed_rag.store.factory import build_vector_store
from biomed_rag.store.indexer import index_documents

st.set_page_config(page_title="BioMed-RAG Engine", layout="wide")


@st.cache_resource
def load_settings():
    return get_settings()


def load_or_build_index(settings):
    documents = read_jsonl(settings.documents_path)
    if not documents:
        documents = run_ingestion(settings, live=False, index=False)
        seed_demo_tables(settings)
    store_bundle = build_vector_store(settings)
    if len(store_bundle.bm25.chunks) == 0:
        index_documents(documents, settings)
        store_bundle = build_vector_store(settings)
    return documents, store_bundle


def pubmed_url(pmid: str | None) -> str | None:
    if not pmid:
        return None
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def render_sidebar(settings):
    st.sidebar.title("BioMed-RAG Engine")
    st.sidebar.caption("Oncology / immunotherapy literature synthesis")
    page = st.sidebar.radio(
        "Navigate",
        ["Ask", "Sources", "Extracted tables", "Chunk inspector", "Evaluation", "Ingest"],
    )
    st.sidebar.markdown("---")
    st.sidebar.write(f"Vector backend: `{settings.vector_backend}`")
    st.sidebar.write(f"LLM configured: {'yes' if settings.llm_configured() else 'no (extractive fallback)'}")
    st.sidebar.info(CLINICAL_DISCLAIMER)
    return page


def page_ask(settings, store_bundle):
    st.header("Ask the literature")
    st.write(
        "Questions must be about published oncology or immunotherapy trials. "
        "Diagnostic and personal treatment queries are blocked."
    )
    query = st.text_area(
        "Clinical literature question",
        placeholder="What PFS hazard ratio was reported for pembrolizumab in KEYNOTE-024?",
        height=100,
    )
    if st.button("Run RAG", type="primary") and query.strip():
        result = run_rag(query.strip(), settings=settings, bundle=store_bundle)
        st.session_state["last_result"] = result.model_dump()
    payload = st.session_state.get("last_result")
    if not payload:
        st.info("Run a query to see a grounded synthesis.")
        return
    if payload.get("refused"):
        st.warning(payload["answer"])
        st.caption(f"Guardrail: {payload.get('refusal_reason')}")
        return
    st.markdown(payload["answer"])
    st.caption(f"Latency {payload.get('latency_ms', 0):.0f} ms")
    cites = payload.get("citations") or []
    if cites:
        st.subheader("PMID citations")
        for pmid in cites:
            url = pubmed_url(pmid)
            st.markdown(f"- [{pmid}]({url})")


def page_sources():
    payload = st.session_state.get("last_result")
    if not payload:
        st.info("Run a query on Ask first.")
        return
    st.header("Sources")
    for i, hit in enumerate(payload.get("hits") or [], start=1):
        meta = hit.get("metadata") or {}
        pmid = meta.get("pmid")
        url = pubmed_url(pmid)
        title = meta.get("title") or "Untitled"
        st.subheader(f"{i}. {title}")
        cols = st.columns(4)
        cols[0].write(f"PMID: {pmid or 'n/a'}")
        cols[1].write(f"DOI: {meta.get('doi') or 'n/a'}")
        cols[2].write(f"Year: {meta.get('year') or 'n/a'}")
        cols[3].write(f"Section: {meta.get('section') or 'n/a'}")
        if url:
            st.markdown(f"[Open on PubMed]({url})")
        st.write((hit.get("parent_text") or hit.get("text") or "")[:1200])


def page_tables():
    payload = st.session_state.get("last_result")
    if not payload:
        st.info("Run a query on Ask first.")
        return
    st.header("Extracted result tables")
    rows = []
    for hit in payload.get("hits") or []:
        meta = hit.get("metadata") or {}
        for stat in hit.get("extracted_stats") or []:
            rows.append(
                {
                    "pmid": meta.get("pmid"),
                    "title": meta.get("title"),
                    "metric": stat.get("metric"),
                    "value": stat.get("value"),
                    "unit": stat.get("unit"),
                    "arm": stat.get("arm"),
                    "comparator": stat.get("comparator"),
                    "table_id": stat.get("table_id"),
                }
            )
    if not rows:
        st.write("No extracted statistics on the retrieved parents.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_chunks():
    payload = st.session_state.get("last_result")
    if not payload:
        st.info("Run a query on Ask first.")
        return
    st.header("Chunk inspector")
    for i, hit in enumerate(payload.get("hits") or [], start=1):
        meta = hit.get("metadata") or {}
        with st.expander(f"Hit {i} parent={hit.get('parent_id')} score={hit.get('score', 0):.4f}"):
            st.json(
                {
                    "chunk_id": hit.get("chunk_id"),
                    "parent_id": hit.get("parent_id"),
                    "source": hit.get("source"),
                    "pmid": meta.get("pmid"),
                    "section": meta.get("section"),
                }
            )
            st.markdown("**Child / retrieved text**")
            st.write(hit.get("text") or "")
            st.markdown("**Parent text**")
            st.write(hit.get("parent_text") or "")


def page_eval(settings):
    st.header("Evaluation")
    if st.button("Run heuristic RAGAS-style benchmark"):
        with st.spinner("Running benchmark..."):
            report = run_benchmark(settings)
            st.session_state["eval_report"] = report
    report = st.session_state.get("eval_report")
    path = settings.ragas_report_path
    if report is None and path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    if not report:
        st.info("No report yet.")
        return
    metrics = report.get("metrics") or {}
    cols = st.columns(len(metrics) or 1)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, value)
    st.dataframe(pd.DataFrame(report.get("rows") or []), use_container_width=True)


def page_ingest(settings):
    st.header("Ingest")
    st.write("Default path loads the offline oncology demo corpus. Live NCBI requires NCBI_EMAIL.")
    col1, col2, col3 = st.columns(3)
    if col1.button("Load demo corpus and index"):
        docs = run_ingestion(settings, live=False, index=True)
        seed_demo_tables(settings)
        st.success(f"Indexed {len(docs)} documents")
        st.cache_resource.clear()
    if col2.button("Generate 50+ synthetic QA"):
        items = generate_synthetic_qa(settings, n=50)
        st.success(f"Wrote {len(items)} QA pairs to {settings.qa_path}")
    if col3.button("Live PubMed ingest", disabled=not settings.ncbi_email):
        docs = run_ingestion(settings, live=True, index=True)
        st.success(f"Live ingest wrote {len(docs)} documents")
        st.cache_resource.clear()
    docs = read_jsonl(settings.documents_path)
    st.write(f"Documents on disk: {len(docs)}")
    qa = load_qa(settings.qa_path)
    st.write(f"Synthetic QA: {len(qa)}")
    if docs:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "doc_id": d.doc_id,
                        "pmid": d.pmid,
                        "section": d.section,
                        "title": d.title,
                        "year": d.year,
                    }
                    for d in docs
                ]
            ),
            use_container_width=True,
        )


def main():
    settings = load_settings()
    page = render_sidebar(settings)
    documents, store_bundle = load_or_build_index(settings)
    st.session_state.setdefault("doc_count", len(documents))
    if page == "Ask":
        page_ask(settings, store_bundle)
    elif page == "Sources":
        page_sources()
    elif page == "Extracted tables":
        page_tables()
    elif page == "Chunk inspector":
        page_chunks()
    elif page == "Evaluation":
        page_eval(settings)
    else:
        page_ingest(settings)


if __name__ == "__main__":
    main()
