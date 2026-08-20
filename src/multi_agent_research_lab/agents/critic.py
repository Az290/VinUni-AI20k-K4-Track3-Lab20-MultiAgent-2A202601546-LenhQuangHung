"""Optional critic agent skeleton for bonus work."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CriticAgent(BaseAgent):
    """Validates the final answer and appends findings.

    Deliberately non-LLM: a cheap, deterministic validation pass that checks citation
    coverage and flags an empty/too-short final answer. Cheap guardrails like this catch
    obvious failures without spending another LLM call.
    """

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        findings: list[str] = []

        if not state.final_answer or not state.final_answer.strip():
            findings.append("final_answer is empty")
        elif len(state.final_answer.split()) < 50:
            findings.append("final_answer looks too short (<50 words)")

        cited_indices = {int(m) for m in _CITATION_PATTERN.findall(state.final_answer or "")}
        max_index = len(state.sources)
        out_of_range = {i for i in cited_indices if i < 1 or i > max_index}
        if out_of_range:
            findings.append(f"citations reference unknown source indices: {sorted(out_of_range)}")
        if state.sources and not cited_indices:
            findings.append("final_answer has sources available but cites none")

        coverage = len(cited_indices - out_of_range) / max_index if max_index else 0.0

        state.add_trace_event(
            "critic.validate",
            {
                "findings": findings,
                "citation_coverage": coverage,
                "cited_indices": sorted(cited_indices),
            },
        )
        if findings:
            state.errors.extend(f"critic: {f}" for f in findings)

        return state
