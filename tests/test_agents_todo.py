"""Unit tests for SupervisorAgent's routing policy.

Originally a skeleton-guard test asserting `StudentTodoError`. Replaced with real routing
tests now that `SupervisorAgent.run` is implemented (see agents/supervisor.py).
"""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _settings(max_iterations: int = 6) -> Settings:
    return Settings(_env_file=None, max_iterations=max_iterations)


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_when_no_sources() -> None:
    state = _state()
    supervisor = SupervisorAgent(settings=_settings())

    result = supervisor.run(state)

    assert result.route_history == ["researcher"]
    assert result.iteration == 1


def test_routes_to_analyst_once_sources_and_notes_exist() -> None:
    state = _state()
    state.sources = [SourceDocument(title="Doc", snippet="snippet")]
    state.research_notes = "notes"
    supervisor = SupervisorAgent(settings=_settings())

    result = supervisor.run(state)

    assert result.route_history == ["analyst"]


def test_routes_to_writer_once_analysis_exists() -> None:
    state = _state()
    state.sources = [SourceDocument(title="Doc", snippet="snippet")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    supervisor = SupervisorAgent(settings=_settings())

    result = supervisor.run(state)

    assert result.route_history == ["writer"]


def test_routes_to_done_once_final_answer_exists() -> None:
    state = _state()
    state.final_answer = "the answer"
    supervisor = SupervisorAgent(settings=_settings())

    result = supervisor.run(state)

    assert result.route_history == ["done"]


def test_max_iterations_forces_done() -> None:
    state = _state()
    state.iteration = 6
    supervisor = SupervisorAgent(settings=_settings(max_iterations=6))

    result = supervisor.run(state)

    assert result.route_history == ["done"]
