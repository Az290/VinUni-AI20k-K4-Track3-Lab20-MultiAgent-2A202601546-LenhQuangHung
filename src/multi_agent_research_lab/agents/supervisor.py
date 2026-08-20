"""Supervisor / router skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy is a plain state-inspection decision tree (no LLM call needed): it looks
    at which fields are still empty in the shared state and picks the next worker that can
    fill the earliest gap. This keeps routing cheap, deterministic, and easy to trace.
    """

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def decide_route(self, state: ResearchState) -> str:
        """Return the next route: researcher, analyst, writer, or done.

        Order of checks mirrors the handoff chain: sources -> analysis -> final answer.
        Guardrail: once `max_iterations` is reached, force `done` even if the state is
        incomplete, so the workflow can never loop forever.
        """

        if state.iteration >= self._settings.max_iterations:
            logger.warning(
                "Max iterations (%s) reached; stopping with best-effort state.",
                self._settings.max_iterations,
            )
            return DONE

        if state.final_answer:
            return DONE

        if not state.sources or not state.research_notes:
            return "researcher"

        if not state.analysis_notes:
            return "analyst"

        return "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        route = self.decide_route(state)
        state.record_route(route)
        state.add_trace_event(
            "route",
            {
                "next": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
