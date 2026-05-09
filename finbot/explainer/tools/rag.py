"""RAG tool for the explainer agent — queries the Vertex AI hybrid corpus."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# finagent/.env — single source of truth for all env vars
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

_retriever: Any | None = None
_init_attempted: bool = False
_init_error: str | None = None


def _init_retriever() -> None:
    """Lazy-init the HybridRetriever once; cache error if it fails."""
    global _retriever, _init_attempted, _init_error
    if _init_attempted:
        return
    _init_attempted = True
    try:
        from rag_tool.config import Settings
        from rag_tool.retrieval.vector_search import HybridRetriever

        settings = Settings(_env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None)

        missing = [
            v for v in ("vector_index_endpoint_id", "deployed_index_id", "gcs_bucket_uri")
            if not getattr(settings, v, "")
        ]
        if missing:
            _init_error = f"RAG corpus not configured — missing: {', '.join(missing)}"
            logger.warning(_init_error)
            return

        _retriever = HybridRetriever(settings)
        logger.info("RAG HybridRetriever initialised.")
    except ImportError as exc:
        _init_error = f"rag_tool package not available: {exc}"
        logger.warning(_init_error)
    except Exception as exc:
        _init_error = str(exc)
        logger.warning("RAG retriever init failed: %s", exc)


def _sync_search_corpus(query: str, top_k: int) -> dict:
    _init_retriever()
    if _init_error:
        return {"available": False, "reason": _init_error}
    try:
        chunks = _retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return {"available": True, "chunks_found": 0, "context": ""}
        context = "\n\n".join(
            f"[{i}] (source: {c.metadata.get('source', c.id)})\n{c.text}"
            for i, c in enumerate(chunks, 1)
        )
        sources = list({c.metadata.get("source", c.id) for c in chunks})
        return {
            "available": True,
            "chunks_found": len(chunks),
            "context": context,
            "sources": sources,
        }
    except Exception as exc:
        logger.error("search_corpus failed: %s", exc)
        return {"available": True, "error": str(exc), "context": ""}


async def search_corpus(query: str, tool_context: ToolContext, top_k: int = 5) -> dict:
    """Search the financial knowledge corpus for context about a metric, concept, or term.

    Use this tool first when explaining any financial concept — the corpus may contain
    domain-specific definitions, examples, and interpretations. If the corpus returns
    no results or is unavailable, fall back to your own knowledge.

    Args:
        query: The concept or metric to look up (e.g. "P/E ratio", "Sharpe ratio").
        top_k: Number of context passages to retrieve (default 5).
    """
    result = await asyncio.to_thread(_sync_search_corpus, query, top_k)

    # Persist to session state so the root agent and InMemorySessionService can read it
    tool_context.state["rag_last_context"] = result

    history: list[dict] = tool_context.state.get("rag_lookups") or []
    history.append({
        "query": query,
        "chunks_found": result.get("chunks_found", 0),
        "sources": result.get("sources", []),
        "available": result.get("available", False),
    })
    tool_context.state["rag_lookups"] = history

    return result


rag_explain_tool = FunctionTool(search_corpus)
