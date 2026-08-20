"""Researcher agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        This agent:
        1. Searches for relevant sources using the search client
        2. Filters and deduplicates results
        3. Uses LLM to synthesize research notes from sources
        """
        logger.info(f"ResearcherAgent running for query: {state.request.query}")

        # Search for relevant sources
        sources = self._search.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = sources

        if not sources:
            logger.warning("No sources found for query")
            state.research_notes = "No sources found for the given query."
            state.errors.append("Researcher: No sources found")
            return state

        # Build context from sources for LLM
        source_context = self._build_source_context(sources)

        # Generate research notes using LLM
        system_prompt = """You are a research assistant. Given a query and relevant sources,
summarize the key findings in a structured format. Include:
- Main topics and themes
- Key facts and data points
- Different perspectives or viewpoints
- Any notable claims or insights

Be concise but comprehensive. Format your response with clear sections."""

        user_prompt = f"""Query: {state.request.query}

Sources:
{source_context}

Please provide a concise summary of the research findings addressing the query above."""

        try:
            response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            state.research_notes = response.content

            # Record agent result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "source_count": len(sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )

            logger.info(
                f"ResearcherAgent completed: {len(sources)} sources, "
                f"{len(response.content)} chars in notes"
            )

        except Exception as exc:
            logger.error(f"ResearcherAgent LLM call failed: {exc}")
            state.errors.append(f"Researcher: LLM call failed - {str(exc)}")
            # Still keep the sources even if LLM fails
            state.research_notes = f"Found {len(sources)} sources but failed to generate notes: {exc}"

        return state

    def _build_source_context(self, sources: list) -> str:
        """Build a context string from sources for LLM consumption."""
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Source {i}]: {source.title}\n"
                f"URL: {source.url or 'N/A'}\n"
                f"Content: {source.snippet}\n"
            )
        return "\n".join(context_parts)
