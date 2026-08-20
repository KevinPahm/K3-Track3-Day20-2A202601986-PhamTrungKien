"""Writer agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        This agent synthesizes research notes and analysis into a final answer
        with proper citations and source references.
        """
        logger.info("WriterAgent running")

        if not state.research_notes:
            state.errors.append("Writer: No research notes available")
            state.final_answer = "Cannot write response: no research notes available."
            return state

        # Build context from research and analysis
        context = self._build_final_context(state)

        # Audience-specific guidance
        audience = state.request.audience
        audience_guidance = self._get_audience_guidance(audience)

        # Generate final answer using LLM
        system_prompt = f"""You are a professional technical writer. Given research notes, analysis,
and sources, write a clear, comprehensive response.

Requirements:
- Write for: {audience}
- Include citations for factual claims using [Source N] notation
- Structure your response with clear headings
- Be factual and avoid hallucination
- {audience_guidance}

Response Structure:
1. Brief introduction
2. Main content with cited claims
3. Key takeaways (if applicable)
4. References section with full source information"""

        user_prompt = f"""Research Query: {state.request.query}

Research Notes:
{state.research_notes}

Analysis Notes:
{state.analysis_notes or 'No analysis available.'}

Sources:
{self._format_sources(state)}

Please write a comprehensive response addressing the query, with proper citations."""

        try:
            response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            state.final_answer = response.content

            # Record agent result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "has_research_notes": bool(state.research_notes),
                        "has_analysis_notes": bool(state.analysis_notes),
                        "source_count": len(state.sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )

            logger.info(
                f"WriterAgent completed: {len(response.content)} chars in final answer"
            )

        except Exception as exc:
            logger.error(f"WriterAgent LLM call failed: {exc}")
            state.errors.append(f"Writer: LLM call failed - {str(exc)}")
            state.final_answer = f"Failed to generate final answer: {exc}"

        return state

    def _build_final_context(self, state: ResearchState) -> str:
        """Build context string for final answer generation."""
        parts = []
        parts.append(f"Sources collected: {len(state.sources)}")
        if state.research_notes:
            parts.append(f"Research notes length: {len(state.research_notes)} chars")
        if state.analysis_notes:
            parts.append(f"Analysis notes length: {len(state.analysis_notes)} chars")
        return "\n".join(parts)

    def _format_sources(self, state: ResearchState) -> str:
        """Format sources for inclusion in prompt."""
        if not state.sources:
            return "No sources available."

        source_list = []
        for i, source in enumerate(state.sources, 1):
            source_list.append(
                f"[Source {i}]: {source.title}\n"
                f"URL: {source.url or 'N/A'}\n"
                f"Content: {source.snippet[:300]}..."
            )
        return "\n".join(source_list)

    def _get_audience_guidance(self, audience: str) -> str:
        """Get writing guidance based on audience."""
        guidance_map = {
            "technical learners": "Use appropriate technical depth. Explain jargon when needed.",
            "executives": "Focus on key insights and business implications. Be concise.",
            "researchers": "Include methodological details and cite sources properly.",
            "general": "Use simple language. Avoid unnecessary technical terms.",
        }
        return guidance_map.get(audience.lower(), "Write clearly and accurately.")
