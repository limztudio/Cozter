"""web_search: scrape DuckDuckGo HTML results for the model."""

from __future__ import annotations

import html
import re
import urllib.parse

import aiohttp

from .base import AgentTool, html_to_text, read_bounded_text


class WebSearchTool(AgentTool):
    name = "web_search"
    description = (
        "Search the public internet for current information. Use this"
        " to find relevant pages, then use web_fetch to read a"
        " specific result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum results to return, default 5, max 10."
                ),
            },
        },
        "required": ["query"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path  # web tools don't need the workspace
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: 'query' must be a non-empty string"

        max_results = args.get("max_results") or 5
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 10))

        search_url = (
            "https://duckduckgo.com/html/?"
            + urllib.parse.urlencode({"q": query})
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 compatible; CozterAgent/1.0; "
                "+https://local"
            )
        }

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    search_url,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        return f"Search failed: HTTP {resp.status}"
                    body = await read_bounded_text(resp)
        except Exception as exc:
            return f"Search failed: {exc}"

        matches = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: list[str] = []
        seen: set[str] = set()

        for href, title_html in matches:
            url = _ddg_unwrap_url(html.unescape(href))
            title = html_to_text(title_html)
            if not title or not url or url in seen:
                continue
            seen.add(url)
            results.append(f"{len(results) + 1}. {title}\n   {url}")
            if len(results) >= max_results:
                break

        if not results:
            return "No search results found."

        return "\n".join(results)

    def summarize(self, args: dict) -> str:
        query = args.get("query", "")
        return f"web_search: {query[:200]}" + (
            "..." if len(query) > 200 else ""
        )


def _ddg_unwrap_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return qs["uddg"][0]
    return url
