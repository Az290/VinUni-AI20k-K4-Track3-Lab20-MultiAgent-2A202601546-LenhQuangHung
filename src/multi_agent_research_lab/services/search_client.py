"""Search client abstraction for ResearcherAgent."""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).resolve().parents[3] / "ai_agent_offline_research_corpus_v2" / "topics"
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "is",
    "are",
    "with",
    "about",
    "write",
    "research",
    "summary",
    "summarize",
    "compare",
    "word",
    "500-word",
    "explain",
    "describe",
    "produce",
    "report",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict[str, Any]]:
    """Load and flatten every offline topic into a list of searchable documents."""

    documents: list[dict[str, Any]] = []
    if not _CORPUS_DIR.exists():
        return documents

    for path in sorted(_CORPUS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable corpus file: %s", path)
            continue

        topic = data.get("topic", {})
        topic_name = topic.get("name", path.stem)
        tags = set(topic.get("tags", []))
        kb = data.get("knowledge_base", {})

        for doc in kb.get("source_documents", []):
            documents.append(
                {
                    "title": doc.get("title", "Untitled source"),
                    "url": doc.get("provenance_url"),
                    "text": doc.get("full_text", ""),
                    "source_id": doc.get("document_id"),
                    "topic": topic_name,
                    "tags": tags,
                    "is_synthetic": doc.get("is_synthetic", False),
                }
            )
        for article in kb.get("knowledge_articles", []):
            documents.append(
                {
                    "title": article.get("title", "Untitled article"),
                    "url": None,
                    "text": article.get("content", ""),
                    "source_id": article.get("article_id"),
                    "topic": topic_name,
                    "tags": tags,
                    "is_synthetic": False,
                }
            )
    return documents


class SearchClient:
    """Provider-agnostic search client.

    Uses the Tavily web search API when `TAVILY_API_KEY` is configured. Otherwise falls back
    to a deterministic, offline keyword search over `ai_agent_offline_research_corpus_v2/`, so
    the lab runs end-to-end without any external search dependency.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if self._settings.tavily_api_key:
            return self._search_tavily(query, max_results)
        return self._search_offline(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=self._settings.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        results = response.json().get("results", [])
        return [
            SourceDocument(
                title=item.get("title", "Untitled"),
                url=item.get("url"),
                snippet=(item.get("content") or "")[:600],
                metadata={"provider": "tavily", "score": item.get("score")},
            )
            for item in results[:max_results]
        ]

    def _search_offline(self, query: str, max_results: int) -> list[SourceDocument]:
        corpus = _load_corpus()
        if not corpus:
            logger.warning("Offline corpus not found at %s", _CORPUS_DIR)
            return []

        query_tokens = _tokenize(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in corpus:
            doc_tokens = _tokenize(doc["title"]) | doc["tags"] | _tokenize(doc["text"][:1000])
            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue
            scored.append((overlap, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = [doc for _, doc in scored[:max_results]]

        # If nothing matched by keyword, fall back to the first topic's core sources so the
        # workflow always has something to reason about instead of failing outright.
        if not top:
            top = corpus[:max_results]

        return [
            SourceDocument(
                title=doc["title"],
                url=doc["url"],
                snippet=doc["text"][:600].strip(),
                metadata={
                    "provider": "offline_corpus",
                    "source_id": doc["source_id"],
                    "topic": doc["topic"],
                    "is_synthetic": doc["is_synthetic"],
                },
            )
            for doc in top
        ]
