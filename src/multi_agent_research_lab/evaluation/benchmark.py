"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and compute benchmark metrics.

    This function:
    1. Times the execution
    2. Collects token usage from agent results
    3. Calculates citation coverage
    4. Estimates failure rate
    5. Computes a simple quality score
    """
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Calculate metrics
    metrics = compute_benchmark_metrics(run_name, state, latency)

    logger.info(
        f"Benchmark '{run_name}' completed: "
        f"latency={latency:.2f}s, quality={metrics.quality_score:.2f}"
    )

    return state, metrics


def compute_benchmark_metrics(
    run_name: str, state: ResearchState, latency: float
) -> BenchmarkMetrics:
    """Compute comprehensive benchmark metrics from a completed run."""
    # Estimate total cost from agent results
    total_cost = sum(
        r.metadata.get("cost_usd", 0) or 0
        for r in state.agent_results
    )

    # Calculate total tokens
    total_input_tokens = sum(
        r.metadata.get("input_tokens", 0) or 0
        for r in state.agent_results
    )
    total_output_tokens = sum(
        r.metadata.get("output_tokens", 0) or 0
        for r in state.agent_results
    )

    # Calculate citation coverage
    citation_coverage = calculate_citation_coverage(state)

    # Calculate quality score
    quality_score = calculate_quality_score(state)

    # Calculate failure rate
    failure_rate = calculate_failure_rate(state)

    # Build notes
    notes = build_metrics_notes(state, total_input_tokens, total_output_tokens)

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=failure_rate,
        notes=notes,
    )


def calculate_citation_coverage(state: ResearchState) -> float:
    """Calculate what fraction of sources are cited in the final answer.

    Returns a value between 0.0 and 1.0.
    """
    final_answer = state.final_answer or ""
    sources = state.sources or []

    if not sources:
        return 0.0

    # Find citation patterns: [Source N], [N], [cite: N]
    citation_pattern = r'\[Source\s*(\d+)\]|\[(\d+)\]|\[cite:?\s*(\d+)\]'
    found_citations = re.findall(citation_pattern, final_answer, re.IGNORECASE)

    # Extract unique source numbers
    cited_sources = set()
    for match in found_citations:
        for group in match:
            if group:
                cited_sources.add(int(group))

    # Also look for URLs in the answer
    url_pattern = r'https?://[^\s\)\]]+'
    urls_in_answer = re.findall(url_pattern, final_answer)

    # Check if any source URLs appear in the answer
    for source in sources:
        if source.url and source.url in final_answer:
            # Try to match by index or title
            pass

    # Coverage = cited sources / total sources
    coverage = len(cited_sources) / len(sources)

    # Boost coverage if URLs are present
    if urls_in_answer and coverage == 0:
        # URLs might be cited differently
        coverage = min(0.3, len(urls_in_answer) / len(sources))

    return min(coverage, 1.0)


def calculate_quality_score(state: ResearchState) -> float:
    """Calculate a quality score (0-10) based on various factors.

    Factors:
    - Has final answer (2 points)
    - Answer length adequate (2 points)
    - Has research notes (2 points)
    - Has analysis notes (2 points)
    - Citation coverage (2 points)
    """
    score = 0.0
    final_answer = state.final_answer or ""

    # Has final answer
    if final_answer:
        score += 2.0

        # Length check (adequate is 200+ chars)
        if len(final_answer) >= 200:
            score += 1.0
        elif len(final_answer) >= 500:
            score += 2.0

    # Has research notes
    if state.research_notes:
        score += 2.0

    # Has analysis notes
    if state.analysis_notes:
        score += 2.0

    # Citation coverage
    citation_coverage = calculate_citation_coverage(state)
    score += citation_coverage * 2.0

    return min(score, 10.0)


def calculate_failure_rate(state: ResearchState) -> float:
    """Calculate the failure rate based on errors and missing outputs.

    Returns a value between 0.0 and 1.0 (higher = more failures).
    """
    if not state.agent_results:
        return 1.0 if not state.final_answer else 0.0

    # Count errors
    error_count = len(state.errors)

    # Check for missing expected outputs
    missing_outputs = []
    if not state.sources:
        missing_outputs.append("sources")
    if not state.research_notes:
        missing_outputs.append("research_notes")
    if not state.final_answer:
        missing_outputs.append("final_answer")

    # Calculate failure rate
    # Weight: errors more heavily than missing outputs
    failure_score = (error_count * 0.3 + len(missing_outputs) * 0.2) / 3.0

    return min(failure_score, 1.0)


def build_metrics_notes(
    state: ResearchState,
    total_input_tokens: int,
    total_output_tokens: int,
) -> str:
    """Build a human-readable notes string for the metrics."""
    parts = []

    if state.agent_results:
        agent_names = [r.agent.value for r in state.agent_results]
        parts.append(f"Agents: {', '.join(agent_names)}")

    parts.append(f"Tokens: {total_input_tokens} in / {total_output_tokens} out")

    if state.iteration > 0:
        parts.append(f"Iterations: {state.iteration}")

    if state.errors:
        parts.append(f"Errors: {len(state.errors)}")

    return "; ".join(parts)
