"""Researcher agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a meticulous research assistant. You are given a user question and a list of "
    "candidate sources (title, url, snippet). Write concise research notes (250-400 words) "
    "that synthesize what these sources say, in plain prose. Every factual claim must be "
    "followed by a bracketed citation using the source index, e.g. [1], [2]. Do not invent "
    "sources or facts that are not supported by the provided snippets."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        query = state.request.query
        try:
            sources = self._search_client.search(query, max_results=state.request.max_sources)
        except AgentExecutionError as exc:
            state.errors.append(f"researcher.search: {exc}")
            logger.error("Researcher search failed: %s", exc)
            sources = []

        state.sources = sources
        state.add_trace_event("researcher.search", {"query": query, "result_count": len(sources)})

        if not sources:
            state.research_notes = "No sources were found for this query."
            return state

        numbered_sources = "\n".join(
            f"[{i + 1}] {src.title} ({src.url or 'no url'}): {src.snippet}"
            for i, src in enumerate(sources)
        )
        user_prompt = f"Question: {query}\n\nCandidate sources:\n{numbered_sources}"

        llm = self._llm_client or LLMClient()
        try:
            response = llm.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.2)
        except AgentExecutionError as exc:
            state.errors.append(f"researcher.llm: {exc}")
            logger.error("Researcher LLM call failed: %s", exc)
            # Fallback: still hand off usable, if less polished, notes downstream.
            state.research_notes = numbered_sources
            return state

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "source_count": len(sources),
                },
            )
        )
        return state
