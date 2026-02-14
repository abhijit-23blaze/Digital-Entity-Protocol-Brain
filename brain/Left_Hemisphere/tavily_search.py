import os
from typing import List, Dict, Optional


class TavilySearchClient:
    """Thin wrapper around the Tavily Python SDK for web search."""

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = None

        if not self.api_key:
            print("[Tavily] Warning: TAVILY_API_KEY not found. Web search disabled.")
            return

        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
        except ImportError:
            print("[Tavily] Warning: tavily-python not installed. Run: pip install tavily-python")
        except Exception as e:
            print(f"[Tavily] Warning: Failed to initialize client: {e}")

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web using Tavily and return structured results.

        Returns a list of dicts with keys: title, url, content
        Returns an empty list on any failure.
        """
        if not self.client:
            return []

        try:
            # Tavily API has a 400-character limit on queries
            if len(query) > 400:
                query = query[:400].rsplit(' ', 1)[0]
                print(f"[Tavily] Warning: Query truncated to {len(query)} chars (max 400).")

            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })

            return results

        except Exception as e:
            print(f"[Tavily] Search error: {e}")
            return []
