"""Command-line entrypoint for the lab starter."""

import logging
import time
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()
logger = logging.getLogger(__name__)


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


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using LLM.

    This is a simple single-agent implementation for benchmarking comparison.
    """

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    console.print(Panel.fit("[bold cyan]Running Single-Agent Baseline...[/bold cyan]"))

    start_time = time.time()

    try:
        llm = LLMClient()

        system_prompt = """You are a research assistant. Answer the user's query
comprehensively with proper structure. Include citations where applicable.
Format your response with clear sections."""

        user_prompt = f"""Query: {request.query}

Please provide a comprehensive response addressing this query.
Include relevant facts, analysis, and proper citations to sources."""

        response = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        state.final_answer = response.content

        elapsed = time.time() - start_time

        # Display results
        console.print(
            Panel.fit(state.final_answer, title="Single-Agent Baseline Response")
        )

        # Display metrics
        table = Table(title="Baseline Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Latency", f"{elapsed:.2f}s")
        if response.input_tokens:
            table.add_row("Input Tokens", str(response.input_tokens))
        if response.output_tokens:
            table.add_row("Output Tokens", str(response.output_tokens))
        if response.cost_usd:
            table.add_row("Est. Cost", f"${response.cost_usd:.6f}")

        console.print(table)

        # Store trace
        state.add_trace_event(
            name="baseline_complete",
            payload={
                "latency": elapsed,
                "output_length": len(response.content),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )

    except Exception as exc:
        logger.error(f"Baseline failed: {exc}")
        console.print(
            Panel.fit(f"Baseline failed: {exc}", title="Error", style="red")
        )
        raise typer.Exit(code=1) from exc


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))

    console.print(Panel.fit("[bold cyan]Running Multi-Agent Workflow...[/bold cyan]"))

    workflow = MultiAgentWorkflow()

    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    # Display results
    if result.final_answer:
        console.print(
            Panel.fit(result.final_answer, title="Multi-Agent Final Answer")
        )
    else:
        console.print(
            Panel.fit(
                "No final answer generated. Check errors below.",
                title="Warning",
                style="yellow",
            )
        )

    # Display trace
    table = Table(title="Multi-Agent Trace")
    table.add_column("Step", style="cyan")
    table.add_column("Agent", style="green")
    table.add_column("Details", style="white")

    for i, route in enumerate(result.route_history, 1):
        table.add_row(str(i), route, "")

    console.print(table)

    # Display agent results summary
    if result.agent_results:
        agent_table = Table(title="Agent Results Summary")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Content Length", style="green")
        agent_table.add_column("Tokens (In/Out)", style="yellow")
        agent_table.add_column("Cost", style="magenta")

        for agent_result in result.agent_results:
            metadata = agent_result.metadata
            tokens_info = f"{metadata.get('input_tokens', 'N/A')}/{metadata.get('output_tokens', 'N/A')}"
            cost = f"${metadata.get('cost_usd', 0):.6f}" if metadata.get('cost_usd') else "N/A"
            agent_table.add_row(
                agent_result.agent.value,
                f"{len(agent_result.content)} chars",
                tokens_info,
                cost,
            )

        console.print(agent_table)

    # Display errors if any
    if result.errors:
        console.print(Panel.fit(
            "\n".join(result.errors),
            title=f"Errors ({len(result.errors)})",
            style="red",
        ))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run both baseline and multi-agent, then compare results."""

    _init()

    console.print(Panel.fit("[bold cyan]Running Benchmark Comparison...[/bold cyan]"))

    # Run baseline
    def baseline_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient()
        response = llm.complete(
            system_prompt="You are a research assistant. Answer the query comprehensively.",
            user_prompt=f"Query: {q}\n\nPlease provide a comprehensive response.",
        )
        state.final_answer = response.content
        state.add_trace_event("baseline", {"output_length": len(response.content)})
        return state

    # Run multi-agent
    def multi_agent_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        return workflow.run(state)

    # Run benchmarks
    console.print("[yellow]Running baseline...[/yellow]")
    baseline_state, baseline_metrics = run_benchmark("baseline", query, baseline_runner)

    console.print("[yellow]Running multi-agent...[/yellow]")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, multi_agent_runner)

    # Display comparison
    table = Table(title="Benchmark Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="green")
    table.add_column("Multi-Agent", style="magenta")

    table.add_row("Latency (s)", f"{baseline_metrics.latency_seconds:.2f}", f"{multi_metrics.latency_seconds:.2f}")
    table.add_row("Output Length", f"{len(baseline_state.final_answer or '')} chars", f"{len(multi_state.final_answer or '')} chars")
    table.add_row("Agents Used", "1 (single)", str(len(multi_state.agent_results)))

    console.print(table)

    # Store comparison result
    comparison = {
        "baseline": baseline_metrics.model_dump(),
        "multi_agent": multi_metrics.model_dump(),
        "query": query,
    }

    console.print(Panel.fit(
        f"Benchmark complete! Multi-agent used {len(multi_state.agent_results)} agents "
        f"in {multi_state.iteration} iterations.",
        title="Summary",
    ))


if __name__ == "__main__":
    app()
