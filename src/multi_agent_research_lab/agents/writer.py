"""Writer agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a technical writer. You receive research notes, analysis notes, and a numbered "
    "list of sources. Write a clear, well-structured final answer for the given audience. "
    "Requirements: (1) answer the question directly, (2) keep the requested length unless the "
    "material genuinely doesn't support it, (3) every non-obvious factual claim must carry a "
    "bracketed citation like [1] referencing the numbered source list, (4) end with a "
    "'Sources' section that lists each numbered source's title and URL (or 'no url')."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        numbered_sources = "\n".join(
            f"[{i + 1}] {src.title} ({src.url or 'no url'})" for i, src in enumerate(state.sources)
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or 'none'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or 'none'}\n\n"
            f"Numbered sources:\n{numbered_sources or 'none'}"
        )

        llm = self._llm_client or LLMClient()
        try:
            response = llm.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.4)
        except AgentExecutionError as exc:
            state.errors.append(f"writer.llm: {exc}")
            logger.error("Writer LLM call failed: %s", exc)
            # Fallback: stitch together whatever we have so the workflow still terminates
            # with a usable (if unpolished) answer instead of raising.
            state.final_answer = (
                "Final answer unavailable due to an LLM error. Best-effort synthesis:\n\n"
                f"{state.analysis_notes or state.research_notes or 'No information gathered.'}"
            )
            return state

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
