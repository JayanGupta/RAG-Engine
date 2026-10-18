"""Answer generator for RAG: Combines retrieved context with the user question.

Augments the prompt sent to the LLM with relevant retrieved chunks.
Provides an extractive fallback mode if no API key is set so the entire pipeline
runs 100% offline out-of-the-box.
"""

from config.prompts import INSUFFICIENT_EVIDENCE, SYSTEM_PROMPT


def format_context(retrieved_items):
    """Format retrieved chunks into a clean, numbered context block for the prompt."""
    if not retrieved_items:
        return "No relevant context found."

    blocks = []
    for idx, item in enumerate(retrieved_items, start=1):
        chunk = item["chunk"]
        title = chunk.title or f"Document {chunk.doc_id}"
        blocks.append(f"[Source {idx}: {title}]\n{chunk.text}")
    return "\n\n".join(blocks)


def extractive_fallback(question, retrieved_items):
    """Offline extractive summary when no LLM API key is configured."""
    if not retrieved_items:
        return INSUFFICIENT_EVIDENCE

    lines = [
        f"Answer synthesized from indexed documents for: '{question}'",
        "",
        "Key relevant excerpts from your knowledge base:",
    ]
    for idx, item in enumerate(retrieved_items[:3], start=1):
        chunk = item["chunk"]
        snippet = chunk.text.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        lines.append(f"{idx}. [Source: {chunk.title}] \"{snippet}\"")

    lines.append("")
    lines.append(
        "*Note: Set USER_LLM_API_KEY in your .env file to enable full generative AI answers with OpenAI/Claude.*"
    )
    return "\n".join(lines)


def generate_answer(question, retrieved_items, settings):
    """Generate an answer using OpenAI LLM or graceful offline fallback."""
    if not retrieved_items:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "sources": [],
            "used_llm": False,
        }

    # Extract unique source titles
    sources = []
    seen_titles = set()
    for item in retrieved_items:
        title = item["chunk"].title
        if title and title not in seen_titles:
            seen_titles.add(title)
            sources.append(title)

    # Check if LLM API key is present
    if not settings.has_llm_key():
        return {
            "answer": extractive_fallback(question, retrieved_items),
            "sources": sources,
            "used_llm": False,
        }

    # Generate with LLM (OpenAI via LangChain)
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        context = format_context(retrieved_items)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        llm_kwargs = {
            "api_key": settings.user_llm_api_key,
            "model": settings.user_llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if settings.user_llm_base_url:
            llm_kwargs["base_url"] = settings.user_llm_base_url

        llm = ChatOpenAI(**llm_kwargs)
        chain = prompt | llm | StrOutputParser()

        raw_answer = chain.invoke({
            "context": context,
            "question": question,
        })

        return {
            "answer": raw_answer.strip(),
            "sources": sources,
            "used_llm": True,
        }
    except Exception as exc:
        # If LLM API call fails (e.g. invalid key or network error), fallback cleanly
        return {
            "answer": f"{extractive_fallback(question, retrieved_items)}\n\n*(LLM generation failed: {str(exc)})*",
            "sources": sources,
            "used_llm": False,
        }
