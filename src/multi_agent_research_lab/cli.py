"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


_BASELINE_SYSTEM_PROMPT = (
    "You are a single, generalist research assistant. Given a user question, you must find "
    "information, judge its credibility, and write a final answer yourself in one pass — "
    "there is no separate researcher, analyst, or writer to help you. Use the provided "
    "candidate sources; every non-obvious factual claim must carry a bracketed citation like "
    "[1] referencing the numbered source list. End with a 'Sources' section listing each "
    "numbered source's title and URL (or 'no url')."
)


def run_baseline(request: ResearchQuery) -> ResearchState:
    """Run the single-agent baseline: one LLM call does search-awareness, analysis, and
    writing together, using the same SearchClient/LLMClient the multi-agent workflow uses so
    the comparison is apples-to-apples on tooling.
    """

    state = ResearchState(request=request)
    search_client = SearchClient()
    llm = LLMClient()

    sources = search_client.search(state.request.query, max_results=state.request.max_sources)
    state.sources = sources
    numbered_sources = "\n".join(
        f"[{i + 1}] {src.title} ({src.url or 'no url'}): {src.snippet}"
        for i, src in enumerate(sources)
    )
    user_prompt = (
        f"Question: {state.request.query}\n\nCandidate sources:\n{numbered_sources or 'none'}"
    )

    try:
        response = llm.complete(_BASELINE_SYSTEM_PROMPT, user_prompt, temperature=0.3)
    except AgentExecutionError as exc:
        state.errors.append(f"baseline.llm: {exc}")
        state.final_answer = f"Baseline failed: {exc}"
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
                "mode": "single-agent-baseline",
            },
        )
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline end to end."""

    _init()
    request = _parse_query(query)
    try:
        state = run_baseline(request)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher -> Analyst -> Writer."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    output: Annotated[
        str, typer.Option("--output", "-o", help="Report output path")
    ] = "reports/benchmark_report.md",
) -> None:
    """Run the configured benchmark queries through baseline and multi-agent, write a report."""

    from pathlib import Path

    from multi_agent_research_lab.evaluation.benchmark import run_benchmark_suite
    from multi_agent_research_lab.evaluation.report import render_markdown_report

    _init()
    results = run_benchmark_suite()
    report = render_markdown_report(results)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(report, encoding="utf-8")
    console.print(Panel.fit(f"Benchmark report written to {output}", title="Benchmark"))


if __name__ == "__main__":
    app()
