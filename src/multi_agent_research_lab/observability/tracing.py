"""Tracing hooks with LangSmith integration.

This module provides tracing capabilities that can use LangSmith, Langfuse,
or a simple local trace collector.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


class LocalTraceCollector:
    """Simple in-memory trace collector for local development."""

    def __init__(self) -> None:
        self._traces: list[dict[str, Any]] = []

    def log_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._traces.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })

    def get_traces(self) -> list[dict[str, Any]]:
        return self._traces.copy()

    def clear(self) -> None:
        self._traces.clear()


# Global trace collector
_trace_collector = LocalTraceCollector()


def get_trace_collector() -> LocalTraceCollector:
    """Get the global trace collector instance."""
    return _trace_collector


class LangSmithTracer:
    """LangSmith tracing integration.

    To enable LangSmith tracing, set LANGSMITH_API_KEY in your .env file.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._client = None
        self._settings = get_settings()

        if self._settings.langsmith_api_key:
            try:
                from langsmith import traceable
                self._traceable = traceable
                self._enabled = True
                logger.info("LangSmith tracing enabled")
            except ImportError:
                logger.warning("langsmith not installed, using local tracing")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace(self, name: str):
        """Decorator to make a function traceable by LangSmith.

        Usage:
            @tracer.trace("my_function")
            def my_function():
                ...
        """
        if self._enabled:
            return self._traceable(name=name)
        # Return a no-op decorator if not enabled
        def decorator(func):
            return func
        return decorator


# Global LangSmith tracer instance
_langsmith_tracer = LangSmithTracer()


def get_langsmith_tracer() -> LangSmithTracer:
    """Get the global LangSmith tracer instance."""
    return _langsmith_tracer


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Create a trace span with timing and logging.

    This context manager:
    1. Logs the span start to local collector
    2. Times the execution
    3. Logs the span completion with duration
    4. Optionally sends to LangSmith if configured

    Args:
        name: Name of the span (e.g., "researcher.search")
        attributes: Additional attributes to log

    Yields:
        The span dictionary, which will have 'duration_seconds' set on exit.

    Example:
        with trace_span("llm.complete", {"model": "gpt-4o"}) as span:
            result = llm.complete(...)
            span["result_length"] = len(result)
    """
    from time import perf_counter

    started = perf_counter()

    # Log to local collector
    _trace_collector.log_event(
        name=f"{name}.start",
        attributes={"attributes": attributes} if attributes else None,
    )

    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "start_time": datetime.now().isoformat(),
        "duration_seconds": None,
    }

    try:
        yield span

        # Calculate duration
        span["duration_seconds"] = perf_counter() - started
        span["end_time"] = datetime.now().isoformat()

        # Log completion
        _trace_collector.log_event(
            name=f"{name}.end",
            attributes={
                "duration_seconds": span["duration_seconds"],
                "attributes": attributes,
            },
        )

        logger.debug(
            f"Trace span '{name}' completed in {span['duration_seconds']:.3f}s"
        )

        # If LangSmith is enabled, also log there
        if _langsmith_tracer.enabled:
            _log_to_langsmith(name, attributes, span["duration_seconds"])

    except Exception as exc:
        span["duration_seconds"] = perf_counter() - started
        span["error"] = str(exc)

        _trace_collector.log_event(
            name=f"{name}.error",
            attributes={
                "duration_seconds": span["duration_seconds"],
                "error": str(exc),
            },
        )

        logger.error(f"Trace span '{name}' failed: {exc}")
        raise


def _log_to_langsmith(
    name: str, attributes: dict[str, Any] | None, duration: float
) -> None:
    """Log a span to LangSmith (if enabled)."""
    # LangSmith integration happens through the @traceable decorator
    # This function provides additional logging context
    try:
        from langsmith import trace as ls_trace

        ls_trace.log(
            name=name,
            inputs={"attributes": attributes} if attributes else {},
            outputs={"duration_seconds": duration},
        )
    except Exception as exc:
        logger.debug(f"Could not log to LangSmith: {exc}")


def configure_langsmith_tracing() -> None:
    """Configure LangSmith environment variables for tracing.

    This should be called during application initialization.
    """
    settings = get_settings()

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info(
            f"LangSmith tracing configured for project: {settings.langsmith_project}"
        )
    else:
        logger.info("LangSmith API key not set, using local tracing only")


def get_trace_url() -> str | None:
    """Get the URL to view traces in LangSmith dashboard.

    Returns None if LangSmith is not configured.
    """
    settings = get_settings()

    if settings.langsmith_api_key and settings.langsmith_project:
        return f"https://smith.langchain.com/o/{settings.langsmith_project}/public"

    return None
