"""Enable LangChain tracing v2 only when USER_LANGSMITH_API_KEY is set."""

from __future__ import annotations

import os

from config.settings import Settings


def configure_tracing(settings: Settings) -> bool:
    key = (settings.user_langsmith_api_key or "").strip()
    if not key:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = settings.user_langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )
    return True
