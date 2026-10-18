"""End-to-end tests for the complete RAGPipeline."""

from rag_engine.pipeline import RAGPipeline


def test_pipeline_demo_corpus():
    pipeline = RAGPipeline()
    docs = pipeline.load_demo_corpus()
    assert len(docs) > 0
    stats = pipeline.get_stats()
    assert stats["document_count"] > 0
    assert stats["chunk_count"] > 0


def test_pipeline_query():
    pipeline = RAGPipeline()
    pipeline.load_demo_corpus()
    result = pipeline.query("What instruments does the James Webb Space Telescope use?")

    assert "answer" in result
    assert len(result["answer"]) > 0
    assert len(result["retrieved_chunks"]) > 0
    assert "evaluation" in result
    assert result["evaluation"]["answer_relevancy"] >= 0.0
    assert result["latency_ms"] > 0
