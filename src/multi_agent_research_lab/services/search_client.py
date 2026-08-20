"""Search client abstraction for ResearcherAgent."""

import logging
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client.

    Supports:
    - Tavily API (if TAVILY_API_KEY is set)
    - DuckDuckGo (fallback, no API key required)
    - Mock mode (when no search is available)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._tavily_api_key = settings.tavily_api_key
        self._use_tavily = bool(self._tavily_api_key and not self._tavily_api_key.startswith("sk-"))

        if self._use_tavily:
            logger.info("SearchClient initialized with Tavily API")
        else:
            logger.info("SearchClient initialized in DuckDuckGo fallback mode")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 5).

        Returns:
            List of SourceDocument objects with title, URL, and snippet.
        """
        if self._use_tavily:
            return self._search_tavily(query, max_results)
        else:
            return self._search_duckduckgo(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search using Tavily API."""
        import httpx

        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            results: list[SourceDocument] = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("url"),
                        snippet=item.get("content", "")[:500],
                        metadata={"score": item.get("score"), "source": "tavily"},
                    )
                )
            logger.info(f"Tavily search returned {len(results)} results for query: {query}")
            return results
        except Exception as exc:
            logger.warning(f"Tavily search failed: {exc}, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, max_results)

    def _search_duckduckgo(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search using DuckDuckGo HTML (no API key required)."""
        try:
            from duckduckgo_search import DDGS

            results: list[SourceDocument] = []
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=max_results):
                    results.append(
                        SourceDocument(
                            title=result.get("title", "Untitled"),
                            url=result.get("href"),
                            snippet=result.get("body", "")[:500],
                            metadata={"source": "duckduckgo"},
                        )
                    )
            logger.info(f"DuckDuckGo search returned {len(results)} results for query: {query}")
            return results
        except ImportError:
            logger.warning("duckduckgo_search not installed, using mock search")
            return self._mock_search(query, max_results)
        except Exception as exc:
            logger.warning(f"DuckDuckGo search failed: {exc}, using mock search")
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return mock search results when no search provider is available."""
        logger.warning(f"Using mock search results for query: {query}")
        return [
            SourceDocument(
                title=f"Mock Article about {query[:30]}",
                url="https://example.com/mock-article",
                snippet=f"This is a mock search result for the query '{query}'. "
                        f"In production, this would be replaced with real search results "
                        f"from Tavily, Bing, or another search provider.",
                metadata={"source": "mock"},
            )
            for i in range(min(max_results, 3))
        ]
