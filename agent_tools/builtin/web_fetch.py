"""web_fetch: download an HTTP/HTTPS URL and return readable text."""

from __future__ import annotations

import re
import urllib.parse

import aiohttp

from ..base import (
    HTTP_USER_AGENT_HEADERS,
    AgentTool,
    coerce_int_arg,
    html_to_text,
    object_parameters,
    read_bounded_text,
    require_nonempty_string_arg,
    summarize_arg,
)


class WebFetchTool(AgentTool):
    name = "web_fetch"
    description = (
        "Fetch a public HTTP/HTTPS URL and return readable text. Use"
        " this after web_search to inspect a page."
    )
    parameters = object_parameters(
        {
            "url": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "description": (
                    "Maximum characters to return, default 12000."
                ),
            },
        },
        ["url"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path  # web tools don't need the workspace
        url, error = require_nonempty_string_arg(args, "url", strip=True)
        if error:
            return error
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Error: only http:// and https:// URLs are allowed"

        max_chars = coerce_int_arg(
            args.get("max_chars") or 12_000,
            default=12_000,
            minimum=1_000,
            maximum=30_000,
        )

        try:
            async with aiohttp.ClientSession(
                headers=HTTP_USER_AGENT_HEADERS,
            ) as session:
                async with session.get(
                    url,
                    allow_redirects=True,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    final_url = str(resp.url)
                    content_type = resp.headers.get("content-type", "")

                    if resp.status >= 400:
                        return (
                            f"Fetch failed: HTTP {resp.status} for"
                            f" {final_url}"
                        )

                    if not (
                        content_type.startswith("text/")
                        or "html" in content_type
                        or "json" in content_type
                        or "xml" in content_type
                        or content_type == ""
                    ):
                        return (
                            f"Fetched {final_url}, but content type is "
                            f"'{content_type}', not readable text."
                        )

                    body = await read_bounded_text(resp)
        except Exception as exc:
            return f"Fetch failed: {exc}"

        title = ""
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if title_match:
            title = html_to_text(title_match.group(1))

        is_html = "html" in content_type.lower()
        text = html_to_text(body) if is_html else body
        text = text.strip()

        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n... [truncated, {len(text)} chars total]"
            )

        header = f"URL: {final_url}"
        if title:
            header += f"\nTitle: {title}"

        return f"{header}\n\n{text}"

    def summarize(self, args: dict) -> str:
        return summarize_arg("web_fetch", args, "url")
