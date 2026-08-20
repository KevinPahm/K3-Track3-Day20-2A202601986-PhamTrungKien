"""LangGraph workflow skeleton."""

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


# Define the state keys used by LangGraph
StateKeys = Literal["supervisor", "researcher", "analyst", "writer", "critic", "done"]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self._graph: StateGraph | None = None
        self._compiled: Any = None

    def build(self) -> StateGraph:
        """Create a LangGraph graph.

        Graph Structure:
        ┌─────────────┐
        │  supervisor │
        └──────┬──────┘
               │
        ┌──────┴──────┬──────────┬──────────┐
        │             │          │          │
        ▼             ▼          ▼          ▼
    researcher    analyst    writer     done
        │             │          │          │
        └──────┬──────┴──────────┴──────────┘
               │  (all route back to supervisor)
               ▼
          supervisor
        """
        workflow = StateGraph(ResearchState)

        # Create agent instances
        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()
        critic = CriticAgent()

        # Define nodes
        workflow.add_node("supervisor", self._supervisor_node(supervisor))
        workflow.add_node("researcher", self._agent_node(researcher))
        workflow.add_node("analyst", self._agent_node(analyst))
        workflow.add_node("writer", self._agent_node(writer))
        workflow.add_node("critic", self._critic_node(critic))

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            source="supervisor",
            path=self._route_decision,
            path_map={
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # Workers route back to supervisor for next decision
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")

        # Add conditional edge from writer to critic or done
        workflow.add_conditional_edges(
            source="writer",
            path=self._should_critique,
            path_map={
                "critic": "critic",
                "done": END,
            },
        )

        # Critic routes back to supervisor
        workflow.add_edge("critic", "supervisor")

        self._graph = workflow
        logger.info("MultiAgentWorkflow graph built successfully")

        return workflow

    def _supervisor_node(self, supervisor: SupervisorAgent):
        """Create the supervisor node function."""
        def node(state: ResearchState) -> dict[str, Any]:
            result = supervisor.run(state)
            return result.model_dump()

        return node

    def _agent_node(self, agent: BaseAgent):
        """Create a generic agent node function."""
        def node(state: dict[str, Any]) -> dict[str, Any]:
            # Convert dict back to ResearchState
            current_state = ResearchState.model_validate(state)
            result = agent.run(current_state)
            return result.model_dump()

        return node

    def _critic_node(self, critic: CriticAgent):
        """Create the critic node function."""
        def node(state: dict[str, Any]) -> dict[str, Any]:
            current_state = ResearchState.model_validate(state)
            result = critic.run(current_state)
            return result.model_dump()

        return node

    def _route_decision(self, state: dict[str, Any]) -> str:
        """Determine next route based on current state.

        This is the routing function for conditional edges.
        """
        # Convert to ResearchState for routing decision
        current_state = ResearchState.model_validate(state)
        supervisor = SupervisorAgent()
        route = supervisor._decide_next_route(current_state)

        logger.debug(f"Routing decision: {route}")
        return route

    def _should_critique(self, state: dict[str, Any]) -> str:
        """Determine if we should run critic or finish.

        Currently always returns 'done' (critic is optional).
        Can be extended to run critic based on certain conditions.
        """
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Args:
            state: Initial ResearchState with the query.

        Returns:
            Final ResearchState with all results populated.
        """
        if self._graph is None:
            self.build()

        if self._compiled is None and self._graph is not None:
            self._compiled = self._graph.compile()

        logger.info(f"Starting multi-agent workflow for query: {state.request.query}")

        try:
            # Run the graph
            initial_state = state.model_dump()
            final_state_dict = self._compiled.invoke(initial_state)

            # Convert back to ResearchState
            final_state = ResearchState.model_validate(final_state_dict)

            logger.info(
                f"Workflow completed: iteration={final_state.iteration}, "
                f"final_route={final_state.route_history[-1] if final_state.route_history else 'none'}"
            )

            return final_state

        except Exception as exc:
            logger.error(f"Workflow execution failed: {exc}")
            state.errors.append(f"Workflow execution failed: {str(exc)}")
            return state
