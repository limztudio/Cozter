"""read_file: return bounded UTF-8 file contents, optionally a line range."""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    object_parameters,
    path_property,
    resolve_inside_workspace,
    summarize_path,
)


# Cap the read.
_READ_FILE_MAX_CHARS = 128 * 1024
_READ_FILE_SKIP_CHUNK_CHARS = 64 * 1024
# Bound the skip.
_READ_FILE_MAX_SKIP_CHARS = 16 * 1024 * 1024


class _OffsetScanLimitExceededError(Exception):
    """The requested line offset needs an excessive sequential scan."""


def _validated_read_bound(
    value: Any, name: str, *, default: int | None,
) -> int | None:
    """Validate an offset/limit tool arg as a real integer bound.

    Tool args are loosely-typed JSON: reject bools, non-integral floats,
    and non-numeric types instead of silently coercing them.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"'{name}' must be an integer")
        value = int(value)
    elif isinstance(value, int):
        pass
    else:
        raise ValueError(f"'{name}' must be an integer")
    return max(0, value) if name == "offset" else value


class ReadFileTool(AgentTool):
    name = "read_file"
    max_result_chars = 32 * 1024
    description = (
        "Read a UTF-8 text file (up to 32K characters/call). offset skips"
        " lines; limit bounds lines. Follow any returned char_offset cursor"
        " with the same offset/limit to recover even a long single line."
        " For image files (png/jpg/gif/webp/bmp), returns"
        " verified dimensions/format/size instead of binary noise —"
        " vision-capable backends already receive the pixels natively."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "path": path_property(),
            "offset": {
                "type": "integer",
            },
            "limit": {
                "type": "integer",
            },
            "char_offset": {
                "type": "integer", "minimum": 0,
                "description": "Characters already read after the line offset; keep offset/limit unchanged when paging.",
            },
        },
        ["path"],
    )

    def __init__(self, result_budget: int = 32 * 1024) -> None:
        # Batch callers give each file its final budget before reading, so
        # neither their formatter nor the dispatcher can discard its cursor.
        self._result_budget = max(256, min(result_budget, 32 * 1024))

    async def run(self, workspace_path: str, args: dict) -> str:
        try:
            target = resolve_inside_workspace(
                workspace_path, args.get("path", ""),
            )
        except ValueError as exc:
            return f"Error: {exc}"
        if not os.path.isfile(target):
            return f"File not found: {args.get('path')}"

        # Binary images: return metadata.
        if os.path.splitext(target)[1].lower() in {
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
        }:
            return await asyncio.to_thread(
                _describe_image_file, target, args.get("path"),
            )

        offset = args.get("offset")
        limit = args.get("limit")
        try:
            start = _validated_read_bound(offset, "offset", default=0)
            assert start is not None  # offset default is 0, never None
        except ValueError as exc:
            return f"Error: {exc}"
        try:
            count = _validated_read_bound(limit, "limit", default=None)
        except ValueError as exc:
            return f"Error: {exc}"
        if count is not None and count < 0:
            return "Error: 'limit' must be >= 0"
        try:
            position = _validated_read_bound(args.get("char_offset"), "char_offset", default=0)
            assert position is not None
        except ValueError as exc:
            return f"Error: {exc}"
        if position < 0:
            return "Error: 'char_offset' must be >= 0"

        try:
            text, truncated = await asyncio.to_thread(
                _read_text_range, target, start, count, position,
                min(_READ_FILE_MAX_CHARS, self._result_budget),
            )
        except _OffsetScanLimitExceededError:
            return (
                "Error: offset requires scanning more than "
                f"{_READ_FILE_MAX_SKIP_CHARS:,} characters; use a smaller "
                "offset"
            )
        except OSError as exc:
            return f"Read failed: {exc}"

        if truncated:
            def marker(length: int) -> str:
                return (
                    f"\n… [truncated file preview at {length} characters;"
                    f" continue with offset={start}, char_offset={position + length}"
                    "; keep offset/limit unchanged; never treat"
                    " this preview as full content; say PARTIAL + remainder"
                    " when work is left]"
                )

            suffix = marker(len(text))
            while len(text) + len(suffix) > self._result_budget:
                text = text[:max(0, self._result_budget - len(suffix))]
                suffix = marker(len(text))
            text += suffix
        return text

    def summarize(self, args: dict) -> str:
        return summarize_path("read_file", args)


def _describe_image_file(target: str, shown_path: object) -> str:
    """Return verified metadata for an image instead of binary noise."""
    from ...utils import probe_image_dimensions
    shown = shown_path if isinstance(shown_path, str) else target
    try:
        size = os.path.getsize(target)
    except OSError as exc:
        return f"Read failed: {exc}"
    dims = probe_image_dimensions(target)
    if dims is not None:
        width, height, fmt = dims
        return (
            f"[Image: {shown} is a {width}x{height} {fmt} ({size:,} bytes)."
            " Vision-capable backends receive these pixels natively —"
            " describe what is actually in the image.]"
        )
    return (
        f"[Image: {shown} ({size:,} bytes)."
        " Binary pixels are not text — vision-capable backends see"
        " them natively; otherwise say what you verified and ask the"
        " user to describe the content.]"
    )


def _read_text_range(
    path: str, start: int, count: int | None, char_offset: int = 0,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    """Read one bounded text range without blocking the event loop.

    The helper runs in a worker thread. Its character cap is enforced while
    reading rather than after assembling the result, including when a single
    unbroken line is much larger than the cap.
    """
    if count == 0:
        return "", False
    if max_chars is None:
        max_chars = _READ_FILE_MAX_CHARS

    with open(path, encoding="utf-8", errors="replace") as file_handle:
        if not _skip_lines(file_handle, start):
            return "", False

        if char_offset > _READ_FILE_MAX_SKIP_CHARS:
            raise _OffsetScanLimitExceededError
        # Character cursors are relative to the original line range. Account
        # for skipped newlines so paging cannot escape an explicit line limit.
        while char_offset:
            chunk = file_handle.read(min(_READ_FILE_SKIP_CHUNK_CHARS, char_offset))
            if not chunk:
                return "", False
            char_offset -= len(chunk)
            if count is not None:
                count -= chunk.count("\n")
                if count <= 0:
                    return "", False

        if count is None:
            text = file_handle.read(max_chars + 1)
            return (
                text[:max_chars],
                len(text) > max_chars,
            )

        chunks: list[str] = []
        remaining = max_chars
        for _ in range(count):
            if remaining == 0:
                return "".join(chunks), bool(file_handle.read(1))
            line = file_handle.readline(remaining + 1)
            if not line:
                break
            if len(line) > remaining:
                chunks.append(line[:remaining])
                return "".join(chunks), True
            chunks.append(line)
            remaining -= len(line)
        return "".join(chunks), False


def _skip_lines(file, count: int) -> bool:
    """Discard *count* lines without materializing or scanning too much.

    Returns False at EOF before the requested offset. Raises
    :class:`_OffsetScanLimitExceededError` when reaching the offset would require
    reading more than the bounded skip budget.
    """
    if count > 128:
        seekable = getattr(file, "seekable", None)
        if callable(seekable) and seekable():
            return _skip_lines_in_chunks(file, count)

    # Preserve nonseekable/readline-only streams and avoid a cookie/rewind
    # for the common small-offset case.
    remaining = _READ_FILE_MAX_SKIP_CHARS
    for _ in range(count):
        while True:
            if remaining <= 0:
                raise _OffsetScanLimitExceededError
            chunk = file.readline(
                min(_READ_FILE_SKIP_CHUNK_CHARS, remaining),
            )
            if not chunk:
                return False
            remaining -= len(chunk)
            if chunk.endswith("\n"):
                break
    return True


def _skip_lines_in_chunks(file, count: int) -> bool:
    """Count large offsets in bounded decoded chunks instead of Python lines.

    TextIO cookies preserve UTF-8 decoder and universal-newline state. Rewind
    only the final chunk and consume its exact prefix, leaving the requested
    range in the same position as readline, including replacement characters.
    """
    remaining = _READ_FILE_MAX_SKIP_CHARS
    while count:
        if remaining <= 0:
            raise _OffsetScanLimitExceededError
        cookie = file.tell()
        chunk = file.read(min(_READ_FILE_SKIP_CHUNK_CHARS, remaining))
        if not chunk:
            return False
        line_count = chunk.count("\n")
        if line_count < count:
            count -= line_count
            remaining -= len(chunk)
            continue

        end = 0
        for _ in range(count):
            end = chunk.find("\n", end) + 1
        file.seek(cookie)
        file.read(end)
        return True
    return True
