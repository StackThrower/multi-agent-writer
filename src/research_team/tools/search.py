"""Web search tool with pluggable backends.

* Tavily — search API built for LLM agents (needs TAVILY_API_KEY).
* DuckDuckGo via `ddgs` — no key, good enough for demos.
"""

from __future__ import annotations

import logging
import os

from langchain.tools import tool

from research_team.config import get_settings

log = logging.getLogger(__name__)


def _search_tavily(query: str, k: int) -> list[dict[str, str]]:
    from langchain_tavily import TavilySearch  # optional extra: pip install .[tavily]

    raw = TavilySearch(max_results=k).invoke({"query": query})
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in raw.get("results", [])
    ]


def _search_ddg(query: str, k: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in DDGS().text(query, max_results=k)
    ]


def search(query: str, k: int | None = None) -> list[dict[str, str]]:
    """Plain function (easy to unit-test and reuse outside of agents)."""
    s = get_settings()
    k = k or s.search_results
    backend = s.search_backend
    if backend == "auto":
        backend = "tavily" if os.getenv("TAVILY_API_KEY") else "duckduckgo"
    log.info("search[%s]: %s", backend, query)
    return _search_tavily(query, k) if backend == "tavily" else _search_ddg(query, k)


@tool
def web_search(query: str) -> str:
    """Search the web. Returns a numbered list of results with title, URL and snippet.

    Use short, specific queries (3-8 words). Call several times with different
    phrasings if the first results are weak.
    """
    try:
        results = search(query)
    except Exception as exc:  # network errors must not kill the agent loop
        return f"Search failed: {exc!r}. Try a different query."
    if not results:
        return "No results. Try a broader or differently worded query."
    return "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1)
    )
