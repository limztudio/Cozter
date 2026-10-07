"""Shared formatting helpers for chat platform adapters."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_CODE_RE = re.compile(r"`([^`]+?)`")


LineRenderer = Callable[[str], str]
CodeBlockRenderer = Callable[[list[str]], list[str]]


def render_inline_code(
    text: str, *, render_prose: LineRenderer, render_code: LineRenderer,
) -> str:
    """Keep inline code literal while formatting the surrounding Markdown."""
    if "`" not in text:
        return render_prose(text)
    marker = "\x00CODE\x00"
    while marker in text:
        marker += "\x00"
    blocks: list[str] = []

    def protect(match: re.Match[str]) -> str:
        token = f"{marker}{len(blocks)}{marker}"
        blocks.append(render_code(match.group(1)))
        return token

    rendered = render_prose(_INLINE_CODE_RE.sub(protect, text))
    if not blocks:
        return rendered
    return re.sub(
        re.escape(marker) + r"([0-9]+)" + re.escape(marker),
        lambda match: blocks[int(match.group(1))], rendered,
    )


def iter_fenced_markdown(
    text: str,
) -> Iterator[tuple[bool, list[str]]]:
    """Yield normal lines and grouped fenced-code lines.

    The boolean is true for code blocks. Normal lines are yielded one at a
    time so callers can apply line-oriented Markdown rules without rebuilding
    the fence state machine.
    """
    in_code_block = False
    code_buf: list[str] = []

    if not isinstance(text, str):
        return
    for source_line in text.split("\n"):
        if source_line.strip().startswith("```"):
            if in_code_block:
                yield True, code_buf
                code_buf = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buf.append(source_line)
        else:
            yield False, [source_line]

    if in_code_block and code_buf:
        yield True, code_buf


def render_fenced_markdown(
    text: str,
    *,
    render_line: LineRenderer,
    render_code_block: CodeBlockRenderer,
) -> str:
    """Render Markdown lines while preserving fenced code block grouping."""
    if not isinstance(text, str):
        return ""
    result: list[str] = []
    for is_code, lines in iter_fenced_markdown(text):
        if is_code:
            result.extend(render_code_block(lines))
        else:
            result.append(render_line(lines[0]))

    return "\n".join(result)


def escape_html_entities(text: str) -> str:
    """Escape ``&``, ``<``, ``>`` for HTML/mrkdwn-safe output.

    Shared by Telegram (HTML) and Slack (mrkdwn uses the same escapes).
    Pairs with :func:`strip_html_markup`, which reverses it.
    """
    if not isinstance(text, str):
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def strip_html_markup(text: str) -> str:
    """Drop simple HTML tags and unescape the entities we emit."""
    if not isinstance(text, str):
        return ""
    plain = _HTML_TAG_RE.sub("", text)
    return (
        plain.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )
