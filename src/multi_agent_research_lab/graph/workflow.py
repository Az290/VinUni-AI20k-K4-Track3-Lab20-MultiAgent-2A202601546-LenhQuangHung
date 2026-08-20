"""LangGraph workflow skeleton."""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.

    Graph shape:

        supervisor --route--> researcher --> supervisor
                          \\--> analyst    --> supervisor
                          \\--> writer     --> critic --> END
                          \\--> done (END)

    The supervisor is a conditional-routing hub: every worker hands control back to it, and
    it decides the next hop by inspecting `ResearchState` (see `SupervisorAgent.decide_route`).
    The stop condition is enforced both by the supervisor (once `final_answer` is set or
    `max_iterations` is reached it returns "done") and structurally (writer always flows to
    critic then END, so the graph cannot loop past a completed answer).
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher or ResearcherAgent()
        self._analyst = analyst or AnalystAgent()
        self._writer = writer or WriterAgent()
        self._critic = critic or CriticAgent()
        self._graph = self.build()

    def _node(self, agent: Any, span_name: str) -> Any:
        def _run(state: ResearchState) -> ResearchState:
            with trace_span(span_name) as span:
                result: ResearchState = agent.run(state)
                span["attributes"]["iteration"] = result.iteration
            result.add_trace_event(span_name, {"duration_seconds": span["duration_seconds"]})
            return result

        return _run

    def build(self) -> Any:
        """Create a LangGraph graph with supervisor routing and a hard stop condition."""

        graph: StateGraph[ResearchState] = StateGraph(ResearchState)

        graph.add_node("supervisor", self._node(self._supervisor, "supervisor"))
        graph.add_node("researcher", self._node(self._researcher, "researcher"))
        graph.add_node("analyst", self._node(self._analyst, "analyst"))
        graph.add_node("writer", self._node(self._writer, "writer"))
        graph.add_node("critic", self._node(self._critic, "critic"))

        graph.set_entry_point("supervisor")

        def _route(state: ResearchState) -> str:
            return state.route_history[-1] if state.route_history else "done"

        graph.add_conditional_edges(
            "supervisor",
            _route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "critic")
        graph.add_edge("critic", END)

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Wraps the whole run in one outer `trace_span` so every node span (supervisor,
        researcher, analyst, writer, critic) nests under a single trace in LangSmith/Langfuse
        instead of appearing as unrelated top-level spans.
        """

        with trace_span("multi_agent_run", {"query": state.request.query}):
            result = self._graph.invoke(state)
        # LangGraph returns whatever type the nodes return; our nodes always return
        # ResearchState instances, but re-validate defensively in case a dict slips through.
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
