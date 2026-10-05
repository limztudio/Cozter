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


# Cap the read: one request must not allocate multi-GB files.
_READ_FILE_MAX_CHARS = 128 * 1024
_READ_FILE_SKIP_CHUNK_CHARS = 64 * 1024
# Line offsets scan data; bound the skip (cancel won't stop the worker).
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
    description = (
        "Read a UTF-8 text file (128 KiB/call; page remainder via"
        " offset/limit). For image files (png/jpg/gif/webp/bmp), returns"
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
        },
        ["path"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        try:
            target = resolve_inside_workspace(
                workspace_path, args.get("path", ""),
            )
        except ValueError as exc:
            return f"Error: {exc}"
        if not os.path.isfile(target):
            return f"File not found: {args.get('path')}"

        # Binary images: return metadata, not noise.
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
            text, truncated = await asyncio.to_thread(
                _read_text_range, target, start, count,
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
            text += (
                f"\n… [truncated at {_READ_FILE_MAX_CHARS} characters;"
                " use offset and limit to read another range; never treat"
                " this preview as full content; say PARTIAL + remainder"
                " when work is left]"
            )
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
    path: str, start: int, count: int | None,
) -> tuple[str, bool]:
    """Read one bounded text range without blocking the event loop.

    The helper runs in a worker thread. Its character cap is enforced while
    reading rather than after assembling the result, including when a single
    unbroken line is much larger than the cap.
    """
    if count == 0:
        return "", False

    with open(path, encoding="utf-8", errors="replace") as file_handle:
        if not _skip_lines(file_handle, start):
            return "", False

        if count is None:
            text = file_handle.read(_READ_FILE_MAX_CHARS + 1)
            return (
                text[:_READ_FILE_MAX_CHARS],
                len(text) > _READ_FILE_MAX_CHARS,
            )

        chunks: list[str] = []
        remaining = _READ_FILE_MAX_CHARS
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
