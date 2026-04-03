"""Web researcher tool using DuckDuckGo + BeautifulSoup."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_REQUESTS_AVAILABLE = False
try:
    import requests
    from bs4 import BeautifulSoup
    _REQUESTS_AVAILABLE = True
except ImportError:
    pass

_DDGO_URL = "https://html.duckduckgo.com/html/?q={query}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36"
    )
}
_MAX_RESULTS = 5
_MAX_SNIPPET_LEN = 300


class Researcher(BaseTool):
    """Fetches web search results and returns summarised findings."""

    name = "researcher"
    description = "Web research tool: searches DuckDuckGo and summarises results."

    def run(self, query: str, **kwargs: Any) -> str:
        if not _REQUESTS_AVAILABLE:
            return f"[Researcher MOCK] Research on: '{query}'. (Install requests + beautifulsoup4 for live search.)"

        snippets = self._ddgo_search(query)
        if not snippets:
            return f"No results found for: '{query}'"

        combined = "\n\n".join(snippets)

        if self._llm is not None:
            prompt = (
                f"Summarise the following search results for the query '{query}':\n\n"
                f"{combined[:2000]}\n\nSummary:"
            )
            try:
                return self._llm.generate(prompt, max_tokens=256)
            except Exception as exc:
                logger.warning("LLM summarisation failed: %s", exc)

        # Return raw snippets if no LLM
        return f"Research findings for '{query}':\n\n" + combined

    # ------------------------------------------------------------------
    # Search backend
    # ------------------------------------------------------------------

    def _ddgo_search(self, query: str) -> list[str]:
        url = _DDGO_URL.format(query=quote_plus(query))
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("DuckDuckGo fetch failed: %s", exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[str] = []

        for result in soup.select(".result__body"):
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            snippet = re.sub(r"\s+", " ", snippet)[:_MAX_SNIPPET_LEN]
            if title or snippet:
                results.append(f"**{title}**\n{snippet}")
            if len(results) >= _MAX_RESULTS:
                break

        logger.info("Researcher found %d results for %r", len(results), query[:60])
        return results
