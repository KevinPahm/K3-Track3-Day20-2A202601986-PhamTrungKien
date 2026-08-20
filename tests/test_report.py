from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    baseline = BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)
    multi = BenchmarkMetrics(run_name="multi-agent", latency_seconds=3.45)
    report = render_markdown_report(baseline, multi)
    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "multi-agent" in report
