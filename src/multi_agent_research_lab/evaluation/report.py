"""Benchmark report rendering."""

from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(
    baseline_metrics: BenchmarkMetrics,
    multi_metrics: BenchmarkMetrics,
    baseline_state: ResearchState | None = None,
    multi_state: ResearchState | None = None,
    query: str = "",
) -> str:
    """Render comprehensive benchmark report to markdown.

    Args:
        baseline_metrics: Metrics from single-agent baseline run.
        multi_metrics: Metrics from multi-agent run.
        baseline_state: Full state from baseline run (optional).
        multi_state: Full state from multi-agent run (optional).
        query: The research query used (optional).

    Returns:
        Markdown-formatted report string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Multi-Agent Research System - Benchmark Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Executive Summary",
        "",
        _generate_executive_summary(baseline_metrics, multi_metrics),
        "",
        "## Detailed Metrics Comparison",
        "",
        "| Metric | Baseline (Single-Agent) | Multi-Agent | Delta |",
        "|---|---|---|---|",
    ]

    # Latency comparison
    latency_delta = multi_metrics.latency_seconds - baseline_metrics.latency_seconds
    latency_delta_pct = (
        f"{(latency_delta / baseline_metrics.latency_seconds * 100):.1f}%"
        if baseline_metrics.latency_seconds > 0
        else "N/A"
    )
    lines.append(
        f"| Latency (seconds) | {baseline_metrics.latency_seconds:.2f}s | "
        f"{multi_metrics.latency_seconds:.2f}s | {latency_delta:+.2f}s ({latency_delta_pct}) |"
    )

    # Cost comparison
    if baseline_metrics.estimated_cost_usd and multi_metrics.estimated_cost_usd:
        cost_delta = multi_metrics.estimated_cost_usd - baseline_metrics.estimated_cost_usd
        cost_delta_pct = (
            f"{(cost_delta / baseline_metrics.estimated_cost_usd * 100):.1f}%"
            if baseline_metrics.estimated_cost_usd > 0
            else "N/A"
        )
        lines.append(
            f"| Est. Cost (USD) | ${baseline_metrics.estimated_cost_usd:.6f} | "
            f"${multi_metrics.estimated_cost_usd:.6f} | ${cost_delta:+.6f} ({cost_delta_pct}) |"
        )
    else:
        lines.append(
            f"| Est. Cost (USD) | {baseline_metrics.estimated_cost_usd or 'N/A'} | "
            f"{multi_metrics.estimated_cost_usd or 'N/A'} | - |"
        )

    # Quality comparison
    if baseline_metrics.quality_score is not None and multi_metrics.quality_score is not None:
        quality_delta = multi_metrics.quality_score - baseline_metrics.quality_score
        lines.append(
            f"| Quality Score (0-10) | {baseline_metrics.quality_score:.1f} | "
            f"{multi_metrics.quality_score:.1f} | {quality_delta:+.1f} |"
        )

    # Citation coverage
    if baseline_metrics.citation_coverage is not None and multi_metrics.citation_coverage is not None:
        cov_delta = multi_metrics.citation_coverage - baseline_metrics.citation_coverage
        lines.append(
            f"| Citation Coverage | {baseline_metrics.citation_coverage:.0%} | "
            f"{multi_metrics.citation_coverage:.0%} | {cov_delta:+.0%} |"
        )

    # Failure rate
    if baseline_metrics.failure_rate is not None and multi_metrics.failure_rate is not None:
        failure_delta = multi_metrics.failure_rate - baseline_metrics.failure_rate
        lines.append(
            f"| Failure Rate | {baseline_metrics.failure_rate:.0%} | "
            f"{multi_metrics.failure_rate:.0%} | {failure_delta:+.0%} |"
        )

    lines.append("")
    lines.append("## Agent Breakdown (Multi-Agent)")
    lines.append("")

    if multi_state and multi_state.agent_results:
        lines.append("| Step | Agent | Output Length | Tokens (In/Out) | Cost |")
        lines.append("|---|---|---|---|---|")

        for i, result in enumerate(multi_state.agent_results, 1):
            metadata = result.metadata
            tokens = f"{metadata.get('input_tokens', '?')}/{metadata.get('output_tokens', '?')}"
            cost = f"${metadata.get('cost_usd', 0):.6f}" if metadata.get('cost_usd') else "N/A"
            lines.append(
                f"| {i} | {result.agent.value} | {len(result.content)} chars | {tokens} | {cost} |"
            )

        lines.append("")
        lines.append(f"**Total iterations:** {multi_state.iteration}")
        lines.append(f"**Route history:** {' → '.join(multi_state.route_history)}")
    else:
        lines.append("No agent results available.")

    lines.append("")
    lines.append("## Error Analysis")
    lines.append("")

    if multi_state and multi_state.errors:
        lines.append("### Errors Encountered")
        for error in multi_state.errors:
            lines.append(f"- {error}")
        lines.append("")
    elif baseline_state and baseline_state.errors:
        lines.append("### Baseline Errors")
        for error in baseline_state.errors:
            lines.append(f"- {error}")
        lines.append("")
    else:
        lines.append("No errors encountered during benchmarking.")
        lines.append("")

    lines.append("")
    lines.append("## Failure Mode Analysis")
    lines.append("")
    lines.append(_generate_failure_mode_analysis(baseline_metrics, multi_metrics))

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(_generate_recommendations(baseline_metrics, multi_metrics))

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Multi-Agent Research Lab Benchmark System*")

    return "\n".join(lines)


def _generate_executive_summary(
    baseline: BenchmarkMetrics, multi: BenchmarkMetrics
) -> str:
    """Generate executive summary paragraph."""
    summary_parts = []

    # Latency comparison
    if multi.latency_seconds > baseline.latency_seconds * 1.5:
        summary_parts.append(
            f"The multi-agent system is significantly slower ({multi.latency_seconds:.1f}s vs "
            f"{baseline.latency_seconds:.1f}s for baseline), which is expected due to "
            f"coordination overhead between agents."
        )
    elif multi.latency_seconds < baseline.latency_seconds * 0.8:
        summary_parts.append(
            f"The multi-agent system is faster than baseline ({multi.latency_seconds:.1f}s vs "
            f"{baseline.latency_seconds:.1f}s), possibly due to parallel agent execution."
        )

    # Quality comparison
    if multi.quality_score and baseline.quality_score:
        if multi.quality_score > baseline.quality_score:
            summary_parts.append(
                f"Multi-agent achieved higher quality ({multi.quality_score:.1f}/10 vs "
                f"{baseline.quality_score:.1f}/10), demonstrating the value of specialized agents."
            )
        else:
            summary_parts.append(
                f"Baseline achieved comparable or higher quality ({baseline.quality_score:.1f}/10 vs "
                f"{multi.quality_score:.1f}/10). Consider optimizing the multi-agent pipeline."
            )

    # Citation coverage
    if multi.citation_coverage is not None and baseline.citation_coverage is not None:
        if multi.citation_coverage > baseline.citation_coverage:
            summary_parts.append(
                f"Multi-agent has better citation coverage ({multi.citation_coverage:.0%} vs "
                f"{baseline.citation_coverage:.0%})."
            )

    # Cost
    if multi.estimated_cost_usd and baseline.estimated_cost_usd:
        if multi.estimated_cost_usd > baseline.estimated_cost_usd * 2:
            summary_parts.append(
                f"Multi-agent is more expensive (${multi.estimated_cost_usd:.4f} vs "
                f"${baseline.estimated_cost_usd:.4f}), reflecting multiple LLM calls."
            )

    if not summary_parts:
        return (
            "The multi-agent system shows comparable performance to the baseline. "
            "Further analysis is needed to determine optimal use cases for each approach."
        )

    return " ".join(summary_parts)


def _generate_failure_mode_analysis(
    baseline: BenchmarkMetrics, multi: BenchmarkMetrics
) -> str:
    """Generate failure mode analysis section."""
    sections = []

    sections.append("### Common Failure Modes in Multi-Agent Systems")
    sections.append("")

    failure_modes = [
        ("**Cascading Errors**", "Errors in earlier agents (e.g., Researcher) propagate "
         "to downstream agents (Analyst, Writer), amplifying failures."),
        ("**Context Loss**", "Information can be lost between agent handoffs if state "
         "management is not robust."),
        ("**Infinite Loops**", "Supervisor may route to the same agent repeatedly if "
         "the agent fails to make progress."),
        ("**Hallucination Accumulation**", "Each LLM call adds hallucination risk; "
         "multi-agent pipelines compound this risk."),
        ("**Coordination Overhead**", "Time spent routing and state management adds "
         "latency without adding directly to output quality."),
    ]

    for mode, description in failure_modes:
        sections.append(f"- {mode}: {description}")

    sections.append("")
    sections.append("### Mitigation Strategies")
    sections.append("")

    mitigations = [
        "Implement **max_iterations** guardrails to prevent infinite loops.",
        "Add **validation** at each agent boundary to catch errors early.",
        "Use **structured output** (Pydantic) to ensure consistent state.",
        "Implement **retry with exponential backoff** for transient failures.",
        "Add a **CriticAgent** to verify outputs before final delivery.",
        "Log all state transitions for **debugging and trace analysis**.",
    ]

    for mitigation in mitigations:
        sections.append(f"- {mitigation}")

    return "\n".join(sections)


def _generate_recommendations(
    baseline: BenchmarkMetrics, multi: BenchmarkMetrics
) -> str:
    """Generate recommendations based on benchmark results."""
    recommendations = []

    # Latency recommendation
    if multi.latency_seconds > baseline.latency_seconds * 1.5:
        recommendations.append(
            "**For low-latency requirements**: Consider using the baseline single-agent "
            "approach, or implement parallel agent execution where possible."
        )

    # Cost recommendation
    if multi.estimated_cost_usd and multi.estimated_cost_usd > 0.01:
        recommendations.append(
            "**For cost-sensitive applications**: Evaluate if the quality improvement "
            "justifies the additional cost. Multi-agent is better for complex, multi-faceted queries."
        )

    # Quality recommendation
    if multi.quality_score and multi.quality_score >= 8.0:
        recommendations.append(
            "**For high-quality requirements**: Multi-agent excels when thorough research, "
            "analysis, and proper citations are needed. Use for production-quality outputs."
        )
    elif multi.quality_score and multi.quality_score < 6.0:
        recommendations.append(
            "**Quality improvement needed**: Review the agent pipeline, add more iterations, "
            "or implement a CriticAgent for quality assurance."
        )

    # General recommendation
    recommendations.append(
        "**Hybrid approach**: Use baseline for simple queries and multi-agent for complex, "
        "multi-faceted research tasks."
    )

    return "\n\n".join(recommendations)
