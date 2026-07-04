"""Shared formatting helpers for chat platform adapters."""

from __future__ import annotations

from collections.abc import Callable


LineRenderer = Callable[[str], str]
CodeBlockRenderer = Callable[[list[str]], list[str]]


def render_fenced_markdown(
    text: str,
    *,
    render_line: LineRenderer,
    render_code_block: CodeBlockRenderer,
) -> str:
    """Render Markdown lines while preserving fenced code block grouping."""
    result: list[str] = []
    in_code_block = False
    code_buf: list[str] = []

    for source_line in text.split("\n"):
        if source_line.strip().startswith("```"):
            if in_code_block:
                result.extend(render_code_block(code_buf))
                code_buf.clear()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buf.append(source_line)
            continue

        result.append(render_line(source_line))

    if in_code_block and code_buf:
        result.extend(render_code_block(code_buf))

    return "\n".join(result)
