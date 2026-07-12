"""web_search: scrape DuckDuckGo HTML results for the model."""

from __future__ import annotations

import html
import re
import urllib.parse

from ..base import (
    AgentTool,
    coerce_int_arg,
    html_to_text,
    object_parameters,
    open_http_response,
    read_bounded_text,
    require_nonempty_string_arg,
    summarize_arg,
)


class WebSearchTool(AgentTool):
    name = "web_search"
    description = (
        "Search the public internet for current information. Use this"
        " to find relevant pages, then use web_fetch to read a"
        " specific result."
    )
    parameters = object_parameters(
        {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum results to return, default 5, max 10."
                ),
            },
        },
        ["query"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path  # web tools don't need the workspace
        query, error = require_nonempty_string_arg(args, "query", strip=True)
        if error:
            return error

        max_results = coerce_int_arg(
            args.get("max_results") or 5,
            default=5,
            minimum=1,
            maximum=10,
        )

        search_url = (
            "https://duckduckgo.com/html/?"
            + urllib.parse.urlencode({"q": query})
        )

        try:
            async with open_http_response(
                search_url, timeout=20,
            ) as response:
                if response.status != 200:
                    return f"Search failed: HTTP {response.status}"
                body = await read_bounded_text(response)
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
        return summarize_arg("web_search", args, "query")


def _ddg_unwrap_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if qs.get("uddg"):
        return qs["uddg"][0]
    return url
