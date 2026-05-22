import logging
import os

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Performs web search via Tavily when document retrieval yields no relevant results."""

    def __init__(self):
        self._client = None
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=api_key)
                logger.info("Tavily web search is available.")
            except ImportError:
                logger.warning("tavily-python package is not installed. Web search disabled.")
        else:
            logger.info("TAVILY_API_KEY not set. Web search disabled.")

    @property
    def available(self) -> bool:
        return self._client is not None

    def search(self, query: str, max_results: int = 5) -> list:
        """
        Search the web for *query* and return a list of result dicts.
        Each dict: {source, content, title, score}
        Returns an empty list if search is unavailable or fails.
        """
        if not self._client:
            return []

        try:
            response = self._client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
            results = response.get("results", [])
            return [
                {
                    "source": r.get("url", "Web"),
                    "content": r.get("content", ""),
                    "title": r.get("title", ""),
                    "score": float(r.get("score", 0.0)),
                    "chunk_index": i,
                }
                for i, r in enumerate(results)
                if r.get("content")
            ]
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return []
