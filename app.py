"""RAG-Engine: Interactive Streamlit Showcase.

Demonstrates how a Retrieval-Augmented Generation (RAG) system works step-by-step:
1. Ingestion & Document Chunking
2. Dense Vector Embeddings & Similarity Search
3. Sparse BM25 Keyword Search
4. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
5. Context-Augmented Prompting & Answer Generation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st
from config.settings import get_settings
from rag_engine.pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Engine Showcase",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the RAGPipeline instance."""
    pipeline = RAGPipeline()
    pipeline.load_demo_corpus()
    return pipeline


def main():
    settings = get_settings()
    pipeline = get_pipeline()

    # Sidebar Navigation & System Status
    st.sidebar.title("🧠 RAG Engine")
    st.sidebar.caption("Domain-Agnostic Retrieval-Augmented Generation")

    stats = pipeline.get_stats()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Knowledge Base Stats:**")
    st.sidebar.markdown(f"- 📄 Documents: `{stats['document_count']}`")
    st.sidebar.markdown(f"- 🧩 Chunks: `{stats['chunk_count']}`")
    st.sidebar.markdown(f"- ⚡ Vector Backend: `{'HashEmbeddings (Offline)' if settings.use_hash_embeddings else settings.embedding_model}`")
    st.sidebar.markdown(f"- 🤖 LLM Mode: `{'OpenAI (' + settings.user_llm_model + ')' if settings.has_llm_key() else 'Extractive Fallback (Offline)'}`")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["💬 Ask Question", "🔬 Pipeline Inspector", "📚 Knowledge Base"],
    )

    if page == "💬 Ask Question":
        render_ask_page(pipeline, settings)
    elif page == "🔬 Pipeline Inspector":
        render_inspector_page(pipeline)
    elif page == "📚 Knowledge Base":
        render_knowledge_base_page(pipeline)


def render_ask_page(pipeline, settings):
    st.header("💬 Ask Questions Grounded in Knowledge")
    st.write(
        "Enter a question to retrieve relevant passages from the knowledge base and generate an answer."
    )

    # Sample query buttons for quick demonstration
    st.markdown("**Try a sample question:**")
    cols = st.columns(3)
    sample_q = None
    if cols[0].button("Quantum: Cooling temperature?"):
        sample_q = "What temperature must superconducting qubits be cooled down to?"
    if cols[1].button("JWST: Primary mirror materials?"):
        sample_q = "What is the primary mirror of the James Webb Space Telescope made of?"
    if cols[2].button("Energy: Long-duration storage?"):
        sample_q = "What technologies are used for long-duration energy storage exceeding 8 hours?"

    query = st.text_input(
        "Your Question:",
        value=sample_q if sample_q else "",
        placeholder="e.g. How does quantum superposition enable qubits to solve complex problems?",
    )

    if st.button("Ask RAG Engine", type="primary") or query:
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving relevant context and generating answer..."):
            result = pipeline.query(query)

        st.subheader("💡 Answer")
        st.markdown(result["answer"])

        # Metadata badges
        st.markdown("---")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("⏱ Latency", f"{result['latency_ms']} ms")
        m_col2.metric("🎯 Faithfulness", f"{result['evaluation']['faithfulness'] * 100:.1f}%")
        m_col3.metric("❓ Relevancy", f"{result['evaluation']['answer_relevancy'] * 100:.1f}%")
        m_col4.metric("📚 Sources Cited", len(result["sources"]))

        if result["sources"]:
            st.markdown(f"**Referenced Documents:** {', '.join([f'`{s}`' for s in result['sources']])}")

        # Show retrieved passages in an expander
        with st.expander("📖 View Retrieved Passages (Top Ranked Chunks)"):
            for idx, item in enumerate(result["retrieved_chunks"], start=1):
                chunk = item["chunk"]
                st.markdown(f"**Passage {idx}** | Source: *{chunk.title}* | Hybrid Score: `{item['score']}`")
                st.info(chunk.text)


def render_inspector_page(pipeline):
    st.header("🔬 RAG Pipeline Inspector")
    st.write(
        "See behind the scenes: watch how the query flows through **Vector Search**, **BM25 Keyword Search**, "
        "and **Reciprocal Rank Fusion (RRF)** before reaching the LLM."
    )

    query = st.text_input(
        "Test Query to Inspect:",
        value="superconducting circuits cooling dilution refrigerator",
    )

    if query.strip():
        result = pipeline.query(query)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1️⃣ Dense Vector Search (Top Hits)")
            st.caption("Captures semantic similarity using vector embeddings.")
            dense_hits = result["dense_results"]
            if dense_hits:
                for idx, h in enumerate(dense_hits[:5], start=1):
                    chunk = h["chunk"]
                    st.markdown(f"**Rank {idx}** (Cosine Sim: `{h['score']}`) - *{chunk.title}*")
                    st.text(chunk.text[:200] + "...")
            else:
                st.info("No vector hits.")

        with col2:
            st.subheader("2️⃣ Sparse BM25 Search (Top Hits)")
            st.caption("Captures exact lexical matches and keyword occurrences.")
            bm25_hits = result["bm25_results"]
            if bm25_hits:
                for idx, h in enumerate(bm25_hits[:5], start=1):
                    chunk = h["chunk"]
                    st.markdown(f"**Rank {idx}** (BM25 Score: `{h['score']}`) - *{chunk.title}*")
                    st.text(chunk.text[:200] + "...")
            else:
                st.info("No BM25 hits.")

        st.subheader("3️⃣ Hybrid Reciprocal Rank Fusion (RRF) Merge")
        st.caption("Combines dense and sparse rankings using `RRF = sum(weight / (60 + rank))`")

        fused_items = result["retrieved_chunks"]
        for idx, item in enumerate(fused_items, start=1):
            chunk = item["chunk"]
            d_rank = item.get("dense_rank", "N/A")
            b_rank = item.get("bm25_rank", "N/A")
            st.markdown(
                f"**Final Rank {idx}** | Fused Score: `{item['score']}` | "
                f"Vector Rank: `{d_rank}` | BM25 Rank: `{b_rank}` | Source: *{chunk.title}*"
            )
            st.success(chunk.text)


def render_knowledge_base_page(pipeline):
    st.header("📚 Knowledge Base Manager")
    st.write("Add documents to your RAG system via plain text, paste, or PDF upload.")

    tab1, tab2, tab3 = st.tabs(["📄 Existing Documents", "✍️ Add Text Document", "📎 Upload PDF"])

    with tab1:
        st.subheader("Indexed Documents")
        if not pipeline.documents:
            st.info("No documents currently indexed. Click 'Reload Demo Corpus' below to populate.")
        else:
            for idx, doc in enumerate(pipeline.documents, start=1):
                with st.expander(f"{idx}. {doc.title} (ID: {doc.doc_id})"):
                    st.caption(f"Source: {doc.source} | Words: {len(doc.text.split())}")
                    st.text(doc.text[:500] + ("..." if len(doc.text) > 500 else ""))

        if st.button("🔄 Reload Demo Corpus"):
            docs = pipeline.load_demo_corpus()
            st.success(f"Loaded and indexed {len(docs)} demo documents!")
            st.rerun()

    with tab2:
        st.subheader("Paste Document Content")
        new_title = st.text_input("Document Title", placeholder="e.g. Introduction to Renewable Energy")
        new_text = st.text_area("Document Text", placeholder="Paste the text of the article or notes here...", height=200)

        if st.button("Index Document", type="primary"):
            if not new_text.strip():
                st.warning("Please provide text content.")
            else:
                title = new_title.strip() or "Untitled Document"
                doc = pipeline.add_document(text=new_text, title=title)
                st.success(f"Successfully indexed document: '{doc.title}' into RAG knowledge base!")
                st.rerun()

    with tab3:
        st.subheader("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Process & Index PDF", type="primary"):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)

                doc = pipeline.add_pdf(tmp_path)
                st.success(f"Extracted and indexed PDF: '{doc.title}' ({len(doc.text.split())} words)!")
                st.rerun()


if __name__ == "__main__":
    main()
