"""Analyst agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        This agent:
        1. Reviews the research notes and sources
        2. Extracts key claims and evidence
        3. Compares different viewpoints
        4. Identifies weak evidence or gaps
        5. Provides structured analysis
        """
        logger.info("AnalystAgent running")

        if not state.research_notes:
            state.errors.append("Analyst: No research notes available")
            state.analysis_notes = "Cannot analyze: no research notes available."
            return state

        # Build context from research
        context = self._build_analysis_context(state)

        # Generate analysis using LLM
        system_prompt = """You are a critical analysis expert. Given research notes and sources,
perform a deep analysis that includes:

1. **Key Claims**: Extract the most important claims and facts
2. **Evidence Assessment**: Evaluate the strength of evidence for each claim
3. **Viewpoint Comparison**: Compare different perspectives found in the sources
4. **Gaps & Weaknesses**: Identify areas with weak evidence or missing information
5. **Implications**: Discuss the implications of these findings

Be thorough and objective. Flag any claims that seem unsubstantiated."""

        user_prompt = f"""Research Query: {state.request.query}

Research Notes:
{state.research_notes}

Sources Available: {len(state.sources)}
{context}

Please provide a comprehensive analysis of the research."""

        try:
            response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            state.analysis_notes = response.content

            # Record agent result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "research_notes_length": len(state.research_notes),
                        "source_count": len(state.sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )

            logger.info(
                f"AnalystAgent completed: {len(response.content)} chars in analysis"
            )

        except Exception as exc:
            logger.error(f"AnalystAgent LLM call failed: {exc}")
            state.errors.append(f"Analyst: LLM call failed - {str(exc)}")
            state.analysis_notes = f"Analysis failed: {exc}"

        return state

    def _build_analysis_context(self, state: ResearchState) -> str:
        """Build context string for analysis."""
        if not state.sources:
            return "No sources available for analysis."

        source_list = []
        for i, source in enumerate(state.sources[:5], 1):  # Limit to 5 sources for context
            source_list.append(f"  {i}. {source.title} - {source.url or 'N/A'}")
        return "\nSources:\n" + "\n".join(source_list)
