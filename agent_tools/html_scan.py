"""Forward-only extraction of HTML titles and search-result anchors.

Only literal tag-name searches use regex. Attribute and tag-content scans
advance their cursors monotonically, including for unfinished HTML, instead
of trying a greedy whole-element regex again at every opening tag.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .base import _scan_html_tag_end


_TITLE_START = re.compile(r"<title(?=[\s/>])", re.IGNORECASE)
_TITLE_END = re.compile(r"</title>", re.IGNORECASE)
_ANCHOR_START = re.compile(r"<a(?=[\s/>])", re.IGNORECASE)
_ANCHOR_END = re.compile(r"</a>", re.IGNORECASE)
_WHITESPACE = frozenset(" \t\n\r\f")


def first_title_contents(value: str) -> str:
    """Return the first complete title's contents, or an empty string."""
    opening = _TITLE_START.search(value)
    if opening is None:
        return ""
    start = _scan_html_tag_end(value, opening.end())
    if start is None:
        return ""
    closing = _TITLE_END.search(value, start)
    return "" if closing is None else value[start:closing.start()]


def _href_attribute(value: str, start: int, end: int) -> str | None:
    """Read an href without rescanning suffixes or matching quoted text."""
    index = start
    while index < end:
        while index < end and (value[index] in _WHITESPACE or value[index] == "/"):
            index += 1
        name_start = index
        while index < end and value[index] not in _WHITESPACE and value[index] not in "=/>\"'":
            index += 1
        name = value[name_start:index].lower()
        while index < end and value[index] in _WHITESPACE:
            index += 1
        if index >= end:
            break
        if value[index] != "=":
            if index == name_start:
                index += 1
            continue
        index += 1
        while index < end and value[index] in _WHITESPACE:
            index += 1
        if index >= end:
            break
        if value[index] in "\"'":
            quote = value[index]
            content_start = index + 1
            closing = value.find(quote, content_start, end)
            if closing < 0:
                return None
            index = closing + 1
        else:
            content_start = index
            while index < end and value[index] not in _WHITESPACE and value[index] != ">":
                index += 1
            closing = index
        if name == "href":
            return value[content_start:closing] or None
    return None


def iter_anchor_contents(value: str) -> Iterator[tuple[str, str]]:
    """Yield complete href/title pairs with a single forward traversal.

    An unfinished opening tag consumes the remaining suffix once. An opening
    anchor without any later closing anchor also ends the scan: no complete
    later anchor could exist without such a closing marker.
    """
    index = 0
    while opening := _ANCHOR_START.search(value, index):
        start = _scan_html_tag_end(value, opening.end())
        if start is None:
            return
        href = _href_attribute(value, opening.end(), start - 1)
        if href is None:
            index = start
            continue
        closing = _ANCHOR_END.search(value, start)
        if closing is None:
            return
        yield href, value[start:closing.start()]
        index = closing.end()
