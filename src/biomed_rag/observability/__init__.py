"""Optional LangSmith tracing. No-op unless USER_LANGSMITH_API_KEY is set."""

from biomed_rag.observability.tracing import configure_tracing

__all__ = ["configure_tracing"]
