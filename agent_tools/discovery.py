"""Bounded discovery pages that retain complete matches and a usable offset."""

from __future__ import annotations


DISCOVERY_RESULT_MAX = 32 * 1024


def render_discovery_page(
    results: list[str], *, offset: int, max_results: int,
) -> str:
    """Render matches, including at most one lookahead, without losing rows.

    Offsets count collected matches, not display lines: filenames themselves
    can contain newlines. A page is stable only while the searched files are
    unchanged; no cached index or snapshot is retained between tool calls.
    """
    count = min(len(results), max_results)
    content_chars = sum(len(row) for row in results[:count]) + max(0, count - 1)
    while count:
        has_more = count < len(results)
        footer = (
            f"\n[More matches remain; repeat the same search with offset={offset + count}"
            " and the same max_results. Offsets count matches in deterministic"
            " directory traversal order; restart at offset=0 if files change.]"
            if has_more else ""
        )
        if content_chars + len(footer) <= DISCOVERY_RESULT_MAX:
            return "\n".join(results[:count]) + footer
        content_chars -= len(results[count - 1]) + (1 if count > 1 else 0)
        count -= 1
    if results:
        return (
            "A single match exceeds the discovery output budget. Narrow the"
            " search path or pattern to inspect it; no matches were skipped."
        )
    return ""
