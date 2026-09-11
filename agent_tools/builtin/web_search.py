"""web_search: scrape DuckDuckGo HTML results for the model."""

from __future__ import annotations

import asyncio
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


# DuckDuckGo serves the same results from two independent frontends. Their
# rate limits and outage profiles differ, so walking the chain turns a
# single flaky host (a recurring failure in practice) into a per-query
# internal retry instead of a failed tool call.
_SEARCH_ENDPOINTS = (
    "https://html.duckduckgo.com/html/?{qs}",
    "https://lite.duckduckgo.com/lite/?{qs}",
)
# A 200 response carrying an empty shell (no result anchors) is common while
# the service sheds load, so each endpoint is tried twice before moving on.
_ATTEMPTS_PER_ENDPOINT = 2
_ATTEMPT_TIMEOUT_SECONDS = 15
_RETRY_DELAY_SECONDS = 0.5

# One scan bound so a pathological response cannot make the parser churn.
_MAX_ANCHORS_SCANNED = 200
_ANCHOR_RE = re.compile(
    r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
# DuckDuckGo tags its sponsored links with these query parameters on its
# own /l/ redirector; real results never carry them.
_AD_PARAMS = frozenset({"ad_provider", "ad_domain", "ad_tool"})
_DDG_HOST_SUFFIX = "duckduckgo.com"


class WebSearchTool(AgentTool):
    name = "web_search"
    description = "Web search for current info."
    parameters = object_parameters(
        {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "description": "max 10.",
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

        encoded = urllib.parse.urlencode({"q": query})
        failures: list[str] = []
        saw_page = False
        attempts_left = len(_SEARCH_ENDPOINTS) * _ATTEMPTS_PER_ENDPOINT

        for template in _SEARCH_ENDPOINTS:
            url = template.format(qs=encoded)
            host = urllib.parse.urlsplit(url).netloc
            for _attempt in range(_ATTEMPTS_PER_ENDPOINT):
                attempts_left -= 1
                try:
                    async with open_http_response(
                        url, timeout=_ATTEMPT_TIMEOUT_SECONDS,
                    ) as response:
                        if response.status != 200:
                            failures.append(
                                f"{host}: HTTP {response.status}",
                            )
                        else:
                            body = await read_bounded_text(response)
                            saw_page = True
                            results = _parse_results(body, max_results)
                            if results:
                                if len(results) < max_results + (
                                    1 if results[-1].startswith("(…parser") else 0
                                ) and results[-1].startswith("(…parser"):
                                    pass  # scan-cap footnote already appended
                                elif len(
                                    [r for r in results if r[:1].isdigit()]
                                ) >= max_results:
                                    results.append(
                                        "(showing first"
                                        f" {max_results} result(s); raise"
                                        " max_results up to 10 for more;"
                                        " never treat this preview as full"
                                        " coverage; say PARTIAL + remainder"
                                        " when coverage is unclear)"
                                    )
                                return "\n".join(results)
                            failures.append(f"{host}: no results parsed")
                except Exception as exc:
                    failures.append(f"{host}: {exc}")
                if attempts_left:
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)

        if saw_page:
            # At least one endpoint answered 200; an empty parse then most
            # likely means the query genuinely has no results.
            return "No search results found."
        return "Search failed: " + "; ".join(dict.fromkeys(failures))

    def summarize(self, args: dict) -> str:
        return summarize_arg("web_search", args, "query")


def _parse_results(body: str, max_results: int) -> list[str]:
    """Extract numbered result lines from either DuckDuckGo frontend.

    The ``html`` frontend marks results with ``class="result__a"`` anchors;
    the ``lite`` frontend renders plain anchors inside its results table.
    Both are parsed with one generic anchor scan plus filters, because every
    non-result link on either page points back at a DuckDuckGo host.
    """
    results: list[str] = []
    seen: set[str] = set()
    scan_capped = False
    for index, match in enumerate(_ANCHOR_RE.finditer(body)):
        if index >= _MAX_ANCHORS_SCANNED:
            scan_capped = True
            break
        raw_href = html.unescape(match.group(1))
        if _is_ad_or_internal(raw_href):
            continue
        title = html_to_text(match.group(2))
        if not title:
            continue
        url = _ddg_unwrap_url(raw_href)
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(f"{len(results) + 1}. {title}\n   {url}")
        if len(results) >= max_results:
            break
    if scan_capped and len(results) < max_results:
        results.append(
            "(…parser scan capped at 200 anchors — further page links"
            " omitted; narrow the query; never treat this preview as full"
            " coverage; say PARTIAL + remainder when coverage is unclear)"
        )
    return results


def _is_ad_or_internal(raw_href: str) -> bool:
    """Whether *raw_href* is a sponsored link or points at DuckDuckGo."""
    try:
        parsed = urllib.parse.urlsplit(raw_href)
        query = urllib.parse.parse_qs(parsed.query)
    except ValueError:
        return True
    if _AD_PARAMS.intersection(query):
        return True
    if parsed.scheme == "" and parsed.netloc == "" and parsed.path == "":
        return True
    target = query.get("uddg", [""])[0] or raw_href
    try:
        target_host = urllib.parse.urlsplit(target).hostname or ""
    except ValueError:
        return True
    return target_host.lower().endswith(_DDG_HOST_SUFFIX)


def _ddg_unwrap_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if qs.get("uddg"):
        return qs["uddg"][0]
    return url
