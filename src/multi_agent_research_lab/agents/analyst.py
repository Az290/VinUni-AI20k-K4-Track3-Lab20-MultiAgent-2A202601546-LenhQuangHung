"""Analyst agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a critical research analyst. You receive research notes (with bracketed source "
    "citations like [1], [2]) and a list of the sources those citations refer to. Produce "
    "structured analysis notes (200-350 words) that: (1) extract the key claims, (2) compare "
    "or reconcile viewpoints where sources disagree, (3) flag any claim that is weakly "
    "supported, synthetic/fictional, or has only one source backing it. Preserve the original "
    "[n] citation markers when you reference a claim."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            state.errors.append("analyst: no research_notes available to analyze")
            state.analysis_notes = "No research notes were available to analyze."
            return state

        numbered_sources = "\n".join(
            f"[{i + 1}] {src.title} ({src.url or 'no url'})"
            + (" [synthetic/benchmark source]" if src.metadata.get("is_synthetic") else "")
            for i, src in enumerate(state.sources)
        )
        user_prompt = (
            f"Research notes:\n{state.research_notes}\n\nSources:\n{numbered_sources or 'none'}"
        )

        llm = self._llm_client or LLMClient()
        try:
            response = llm.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.1)
        except AgentExecutionError as exc:
            state.errors.append(f"analyst.llm: {exc}")
            logger.error("Analyst LLM call failed: %s", exc)
            # Fallback: pass research notes through unchanged rather than blocking the writer.
            state.analysis_notes = (
                "Analysis unavailable (LLM error); passing raw research notes through:\n"
                + state.research_notes
            )
            return state

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
