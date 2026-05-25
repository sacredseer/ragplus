


class WebSearchAgent:
    """Handles fallback web searches via Tavily if local documents lack relevant information."""

    def search(self, query: str, api_key: str, max_results: int = 5) -> list:
        """
        Executes a web search query dynamically using the provided API key.
        
        Returns:
            list[dict]: List of web snippets.
        """
        if not api_key:
            return []

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(
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
            return []
