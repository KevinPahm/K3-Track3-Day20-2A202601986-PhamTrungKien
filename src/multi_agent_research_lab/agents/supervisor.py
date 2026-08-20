"""Supervisor / router skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing Logic:
    - If no sources/research_notes → researcher
    - If has research_notes but no analysis_notes → analyst
    - If has analysis_notes but no final_answer → writer
    - If has final_answer → done (stop)
    - If max_iterations reached → done (stop)
    - If error occurred → handle based on error type
    """

    name = "supervisor"

    def __init__(self) -> None:
        settings = get_settings()
        self._max_iterations = settings.max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        This supervisor implements a sequential pipeline:
        1. researcher → 2. analyst → 3. writer → done

        Returns the state with updated route_history.
        """
        current_route = self._decide_next_route(state)
        state.record_route(current_route)

        logger.info(
            f"Supervisor decision: route='{current_route}', "
            f"iteration={state.iteration}/{self._max_iterations}"
        )

        state.add_trace_event(
            name="supervisor_route",
            payload={"route": current_route, "iteration": state.iteration},
        )

        return state

    def _decide_next_route(self, state: ResearchState) -> str:
        """Decide the next route based on current state.

        Returns one of: 'researcher', 'analyst', 'writer', 'done'
        """
        # Check if we've exceeded max iterations
        if state.iteration >= self._max_iterations:
            logger.info("Max iterations reached, stopping")
            return "done"

        # Check for critical errors that should stop the workflow
        if self._should_stop_on_error(state):
            logger.warning("Stopping due to critical errors")
            return "done"

        # Sequential routing based on missing data
        # Priority: check final answer first, then analysis, then research
        if state.final_answer and state.analysis_notes and state.research_notes:
            # All components are present, workflow is complete
            logger.info("All components complete, workflow finished")
            return "done"

        if not state.research_notes:
            # Need to do research first
            return "researcher"

        if not state.analysis_notes:
            # Have research, need analysis
            return "analyst"

        if not state.final_answer:
            # Have research and analysis, need final answer
            return "writer"

        # Fallback (should not reach here normally)
        return "done"

    def _should_stop_on_error(self, state: ResearchState) -> bool:
        """Determine if we should stop the workflow based on errors.

        We stop only on critical errors (e.g., repeated failures),
        not on warnings or recoverable issues.
        """
        if not state.errors:
            return False

        # Count consecutive agent errors
        recent_errors = state.errors[-3:] if len(state.errors) >= 3 else state.errors

        # Stop if we have 3+ consecutive errors suggesting systemic failure
        if len(recent_errors) >= 3:
            logger.error(f"Multiple recent errors detected: {recent_errors}")
            return True

        return False


def get_supervisor_routing(state: ResearchState) -> str:
    """Standalone function for supervisor routing decision.

    This can be used by the workflow graph to determine the next node.
    """
    supervisor = SupervisorAgent()
    return supervisor._decide_next_route(state)
