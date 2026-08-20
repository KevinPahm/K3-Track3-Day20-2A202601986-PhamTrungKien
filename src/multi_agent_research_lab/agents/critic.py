"""Critic agent skeleton for quality assurance."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Performs quality checks on the final answer.

    This agent can:
    - Check fact consistency with sources
    - Verify citation coverage
    - Detect potential hallucinations
    - Suggest improvements
    """

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Perform quality checks and populate critique in agent_results.

        The critique is stored in agent_results rather than a dedicated field,
        as it's supplementary to the final answer.
        """
        logger.info("CriticAgent running")

        if not state.final_answer:
            state.errors.append("Critic: No final answer to review")
            return state

        # Perform various quality checks
        checks = {
            "citation_coverage": self._check_citation_coverage(state),
            "source_relevance": self._check_source_relevance(state),
            "hallucination_risk": self._check_hallucination_risk(state),
            "completeness": self._check_completeness(state),
        }

        # Generate overall critique using LLM
        critique = self._generate_critique(state, checks)

        # Record agent result
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critique,
                metadata={
                    "checks": checks,
                    "has_final_answer": bool(state.final_answer),
                    "source_count": len(state.sources),
                },
            )
        )

        logger.info("CriticAgent completed")

        return state

    def _check_citation_coverage(self, state: ResearchState) -> dict:
        """Check if claims in final answer are cited."""
        final_answer = state.final_answer or ""
        sources = state.sources or []

        # Find citations in the text
        citation_pattern = r'\[Source\s+(\d+)\]|\[(\d+)\]|\[cite:?\s*(\d+)\]'
        found_citations = re.findall(citation_pattern, final_answer, re.IGNORECASE)

        total_sources = len(sources)
        cited_sources = len(set(c for match in found_citations for c in match if c))

        coverage = cited_sources / total_sources if total_sources > 0 else 0

        return {
            "passed": coverage >= 0.5,
            "score": coverage,
            "cited_sources": cited_sources,
            "total_sources": total_sources,
            "has_citations": len(found_citations) > 0,
        }

    def _check_source_relevance(self, state: ResearchState) -> dict:
        """Check if sources are relevant to the query."""
        query = state.request.query.lower()
        sources = state.sources or []

        if not sources:
            return {"passed": False, "score": 0, "message": "No sources available"}

        # Simple keyword overlap check
        query_words = set(query.split())
        relevant_count = 0

        for source in sources:
            source_text = (source.title + " " + source.snippet).lower()
            source_words = set(source_text.split())
            overlap = len(query_words & source_words)
            if overlap >= 2:  # At least 2 common words
                relevant_count += 1

        relevance_score = relevant_count / len(sources) if sources else 0

        return {
            "passed": relevance_score >= 0.5,
            "score": relevance_score,
            "relevant_sources": relevant_count,
            "total_sources": len(sources),
        }

    def _check_hallucination_risk(self, state: ResearchState) -> dict:
        """Estimate hallucination risk based on verifiable factors."""
        final_answer = state.final_answer or ""

        # Check for hedging language (good sign)
        hedging_words = ["may", "might", "could", "possibly", "perhaps", "suggest"]
        has_hedging = any(word in final_answer.lower() for word in hedging_words)

        # Check for definitive claims without citations
        definitive_patterns = [
            r'\b(is|are|was|were)\s+[A-Z][a-z]+',
            r'\b\d+%\s+of',
            r'\bstudies\s+show\b',
            r'\bresearch\s+has\s+(?:found|shown)\b',
        ]

        claims_without_citations = 0
        for pattern in definitive_patterns:
            matches = re.findall(pattern, final_answer)
            claims_without_citations += len(matches)

        # Check for URLs that appear in answer
        url_pattern = r'https?://[^\s\)\]]+'
        urls_in_answer = re.findall(url_pattern, final_answer)
        has_urls = len(urls_in_answer) > 0

        # Lower risk if there's hedging and/or citations/URLs
        risk_score = 0.5  # Base risk
        if not has_hedging:
            risk_score += 0.2
        if claims_without_citations > 3:
            risk_score += 0.2
        if not has_urls:
            risk_score += 0.1

        return {
            "passed": risk_score < 0.6,
            "score": 1 - risk_score,  # Higher is better
            "risk_factors": {
                "has_hedging": has_hedging,
                "claims_without_citations": claims_without_citations,
                "has_urls": has_urls,
            },
        }

    def _check_completeness(self, state: ResearchState) -> dict:
        """Check if the answer addresses the query adequately."""
        query = state.request.query.lower()
        final_answer = state.final_answer or ""

        # Check length (too short might be incomplete)
        min_length = 200
        length_ok = len(final_answer) >= min_length

        # Check if query terms appear in answer
        query_terms = [w for w in query.split() if len(w) > 4]  # Skip short words
        if query_terms:
            coverage = sum(1 for term in query_terms if term in final_answer.lower()) / len(query_terms)
        else:
            coverage = 1.0

        # Check for conclusion/summary
        has_conclusion = any(
            phrase in final_answer.lower()
            for phrase in ["in conclusion", "summary", "in summary", "key takeaway", "to summarize"]
        )

        completeness_score = (
            (1 if length_ok else 0) * 0.3 +
            coverage * 0.5 +
            (0.2 if has_conclusion else 0)
        )

        return {
            "passed": completeness_score >= 0.6,
            "score": completeness_score,
            "length_ok": length_ok,
            "term_coverage": coverage,
            "has_conclusion": has_conclusion,
        }

    def _generate_critique(self, state: ResearchState, checks: dict) -> str:
        """Generate a human-readable critique based on the checks."""
        critique_parts = ["## Quality Critique\n"]

        for check_name, check_result in checks.items():
            status = "✅ PASS" if check_result.get("passed") else "❌ FAIL"
            score = check_result.get("score", 0)
            critique_parts.append(f"**{check_name.replace('_', ' ').title()}:** {status} (score: {score:.2f})")

            # Add specific details
            if check_name == "citation_coverage":
                critique_parts.append(
                    f"  - Cited {check_result.get('cited_sources', 0)}/{check_result.get('total_sources', 0)} sources"
                )
            elif check_name == "completeness":
                if not check_result.get("length_ok"):
                    critique_parts.append("  - Answer is too short")
                if check_result.get("term_coverage", 0) < 0.5:
                    critique_parts.append("  - Missing key terms from query")

        critique_parts.append("\n### Overall Assessment")
        all_passed = all(c.get("passed", False) for c in checks.values())
        if all_passed:
            critique_parts.append("✅ All quality checks passed!")
        else:
            critique_parts.append("⚠️ Some quality checks failed. Consider revisions.")

        return "\n".join(critique_parts)
