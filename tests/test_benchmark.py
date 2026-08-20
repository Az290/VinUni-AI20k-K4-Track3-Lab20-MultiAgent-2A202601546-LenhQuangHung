from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def _state_with_answer(cited: bool) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title="A", snippet="a"),
        SourceDocument(title="B", snippet="b"),
    ]
    if cited:
        state.final_answer = "Multi-agent systems split work across roles [1][2]. " * 40
    else:
        state.final_answer = "Multi-agent systems split work across roles. " * 40
    return state


def test_run_benchmark_measures_latency_and_quality() -> None:
    state, metrics = run_benchmark(
        "unit-test", "Explain multi-agent systems", lambda _q: _state_with_answer(cited=True)
    )

    assert metrics.run_name == "unit-test"
    assert metrics.latency_seconds >= 0
    assert metrics.citation_coverage == 1.0
    assert metrics.failure_rate == 0.0
    assert metrics.quality_score is not None and metrics.quality_score > 0
    assert state.final_answer


def test_run_benchmark_penalizes_missing_citations() -> None:
    _, cited_metrics = run_benchmark("cited", "q", lambda _q: _state_with_answer(cited=True))
    _, uncited_metrics = run_benchmark("uncited", "q", lambda _q: _state_with_answer(cited=False))

    assert uncited_metrics.citation_coverage == 0.0
    assert (uncited_metrics.quality_score or 0) < (cited_metrics.quality_score or 0)
