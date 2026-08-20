"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of numbered sources actually cited in the final answer.

    A cheap, deterministic proxy for grounding quality: it does not verify the citation is
    used *correctly*, only that the answer references most of what was retrieved instead of
    ignoring it.
    """

    if not state.sources:
        return None
    cited = {int(m) for m in _CITATION_PATTERN.findall(state.final_answer or "")}
    valid = {i for i in cited if 1 <= i <= len(state.sources)}
    return len(valid) / len(state.sources)


def _quality_score(state: ResearchState) -> float | None:
    """Cheap automatic proxy for quality (0-10), meant to complement — not replace — human
    peer review scoring from `docs/peer_review_rubric.md`.

    Heuristic: rewards a present, reasonably long answer and citation coverage; penalizes
    recorded errors.
    """

    if not state.final_answer:
        return 0.0

    word_count = len(state.final_answer.split())
    length_score = min(word_count / 400, 1.0) * 5  # up to 5 points for length/completeness
    coverage = _citation_coverage(state) or 0.0
    citation_score = coverage * 4  # up to 4 points for citation coverage
    error_penalty = min(len(state.errors), 3)  # up to -3 points
    score = max(0.0, min(10.0, length_score + citation_score + 1 - error_penalty))
    return round(score, 2)


def _estimated_cost(state: ResearchState) -> float | None:
    costs: list[float] = [
        float(cost)
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    if not costs:
        return None
    return round(sum(costs), 6)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one query through `runner`, measure latency, and score quality/cost/citations."""

    from multi_agent_research_lab.core.schemas import ResearchQuery

    started = perf_counter()
    failed = False
    notes = ""
    try:
        state = runner(query)
    except AgentExecutionError as exc:
        latency = perf_counter() - started
        logger.error("Benchmark run %r failed for query %r: %s", run_name, query, exc)
        empty_state = ResearchState(request=ResearchQuery(query=query))
        empty_state.errors.append(str(exc))
        return empty_state, BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=round(latency, 3),
            failure_rate=1.0,
            notes=f"runner raised: {exc}",
        )
    latency = perf_counter() - started

    if state.errors:
        failed = True
        notes = "; ".join(state.errors[:3])

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=_estimated_cost(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
        notes=notes,
    )
    return state, metrics


def run_benchmark_suite() -> list[BenchmarkMetrics]:
    """Run every configured benchmark query through both baseline and multi-agent, averaged
    into one `BenchmarkMetrics` per run type.
    """

    from multi_agent_research_lab.cli import run_baseline
    from multi_agent_research_lab.core.schemas import ResearchQuery
    from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

    settings = get_settings()
    queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    def baseline_runner(query: str) -> ResearchState:
        return run_baseline(ResearchQuery(query=query))

    def multi_agent_runner(query: str) -> ResearchState:
        workflow = MultiAgentWorkflow()
        return workflow.run(ResearchState(request=ResearchQuery(query=query)))

    all_metrics: list[BenchmarkMetrics] = []
    for run_name, runner in (("baseline", baseline_runner), ("multi-agent", multi_agent_runner)):
        run_metrics = []
        for query in queries:
            _, metrics = run_benchmark(run_name, query, runner)
            run_metrics.append(metrics)

        n = len(run_metrics)
        costs = [m.estimated_cost_usd for m in run_metrics if m.estimated_cost_usd is not None]
        qualities = [m.quality_score for m in run_metrics if m.quality_score is not None]
        coverages = [m.citation_coverage for m in run_metrics if m.citation_coverage is not None]
        aggregated = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=round(sum(m.latency_seconds for m in run_metrics) / n, 3),
            estimated_cost_usd=round(sum(costs) / len(costs), 6) if costs else None,
            quality_score=round(sum(qualities) / len(qualities), 2) if qualities else None,
            citation_coverage=round(sum(coverages) / len(coverages), 3) if coverages else None,
            failure_rate=round(sum(m.failure_rate or 0.0 for m in run_metrics) / n, 3),
            notes=f"avg over {n} queries (max_iterations={settings.max_iterations})",
        )
        all_metrics.append(aggregated)

    return all_metrics
