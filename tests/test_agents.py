"""Unit tests for agent implementations."""

import pytest

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


class TestSupervisorAgent:
    """Tests for SupervisorAgent routing logic."""

    def test_routes_to_researcher_when_no_sources(self) -> None:
        """Supervisor should route to researcher when no sources available."""
        state = ResearchState(request=ResearchQuery(query="Test query for routing"))
        supervisor = SupervisorAgent()
        result = supervisor.run(state)

        assert "researcher" in result.route_history
        assert result.iteration == 1

    def test_routes_to_analyst_when_has_sources_but_no_analysis(self) -> None:
        """Supervisor should route to analyst when sources exist but no analysis."""
        state = ResearchState(
            request=ResearchQuery(query="Test query for routing"),
            sources=[],
            research_notes="Some notes",
        )
        supervisor = SupervisorAgent()
        result = supervisor.run(state)

        assert "analyst" in result.route_history
        assert result.iteration == 1

    def test_routes_to_writer_when_has_analysis_but_no_final_answer(self) -> None:
        """Supervisor should route to writer when analysis exists but no final answer."""
        state = ResearchState(
            request=ResearchQuery(query="Test query for routing"),
            sources=[],
            research_notes="Some notes",
            analysis_notes="Analysis here",
        )
        supervisor = SupervisorAgent()
        result = supervisor.run(state)

        assert "writer" in result.route_history

    def test_routes_to_done_when_complete(self) -> None:
        """Supervisor should route to done when all components are present."""
        state = ResearchState(
            request=ResearchQuery(query="Test query for routing"),
            sources=[],
            research_notes="Some notes",
            analysis_notes="Analysis here",
            final_answer="Final answer here",
        )
        supervisor = SupervisorAgent()
        result = supervisor.run(state)

        assert "done" in result.route_history

    def test_respects_max_iterations(self) -> None:
        """Supervisor should stop routing when max iterations are reached."""
        # Test that the supervisor's _max_iterations setting affects routing
        supervisor = SupervisorAgent()
        supervisor._max_iterations = 2

        # Create a state with max iterations already reached
        state = ResearchState(request=ResearchQuery(query="Test query for routing"))
        state.iteration = 2  # Set to max

        # Should route to done when iteration >= max_iterations
        result = supervisor.run(state)
        assert result.route_history[-1] == "done"


class TestResearchState:
    """Tests for ResearchState."""

    def test_record_route_increments_iteration(self) -> None:
        """Recording a route should increment the iteration counter."""
        state = ResearchState(request=ResearchQuery(query="Test query here"))
        assert state.iteration == 0

        state.record_route("researcher")
        assert state.iteration == 1
        assert state.route_history == ["researcher"]

        state.record_route("analyst")
        assert state.iteration == 2
        assert state.route_history == ["researcher", "analyst"]

    def test_add_trace_event(self) -> None:
        """Adding a trace event should append to trace list."""
        state = ResearchState(request=ResearchQuery(query="Test query here"))

        state.add_trace_event("test_event", {"key": "value"})
        assert len(state.trace) == 1
        assert state.trace[0]["name"] == "test_event"
        assert state.trace[0]["payload"] == {"key": "value"}


class TestAgentName:
    """Tests for agent name enum."""

    def test_agent_names_exist(self) -> None:
        """All expected agent names should be defined."""
        from multi_agent_research_lab.core.schemas import AgentName

        assert AgentName.SUPERVISOR == "supervisor"
        assert AgentName.RESEARCHER == "researcher"
        assert AgentName.ANALYST == "analyst"
        assert AgentName.WRITER == "writer"
        assert AgentName.CRITIC == "critic"
