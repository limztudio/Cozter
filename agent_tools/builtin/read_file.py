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


# ``execute_tool`` limits the result sent back to the model, but applying
# that limit after ``read()`` still lets one request allocate a multi-GB log
# or disk image in the bot process. Keep this comfortably above the visible
# result cap so ordinary source files remain useful while bounding both
# memory and synchronous disk work.
_READ_FILE_MAX_CHARS = 128 * 1024
_READ_FILE_SKIP_CHUNK_CHARS = 64 * 1024
# An offset is expressed in lines, so reaching it can require scanning a
# great deal of data (especially in generated files with very long lines).
# ``asyncio.to_thread`` keeps that scan off the event loop, but cancelling a
# timed-out await does not stop the underlying worker thread. Bound the skip
# itself so an untrusted tool argument cannot leave a thread reading a huge
# file long after the tool call has returned.
_READ_FILE_MAX_SKIP_CHARS = 16 * 1024 * 1024


class _OffsetScanLimitExceeded(Exception):
    """The requested line offset needs an excessive sequential scan."""


def _validated_read_bound(
    value: Any, name: str, *, default: int | None,
) -> int | None:
    """Validate an offset/limit tool arg as a real integer bound.

    Tool args arrive as loosely-typed JSON: ``int(True) == 1`` and
    ``int(1.9) == 1`` would silently turn junk into a different read
    range, and a raw float reaching ``range()`` raises ``TypeError``
    instead of the documented error string. Reject bools, non-integral
    floats, and non-numeric types with a ValueError the caller renders.
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

        # Images are binary: decoding them as text yields noise. Return
        # verified metadata instead so the model stops guessing from
        # garbage bytes; vision-capable backends see the pixels natively.
        if os.path.splitext(target)[1].lower() in {
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
        }:
            describe = await asyncio.to_thread(
                _describe_image_file, target, args.get("path"),
            )
            return describe

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
        except _OffsetScanLimitExceeded:
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
        w, h, fmt = dims
        return (
            f"[Image: {shown} is a {w}x{h} {fmt} ({size:,} bytes)."
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

    with open(path, encoding="utf-8", errors="replace") as f:
        if not _skip_lines(f, start):
            return "", False

        if count is None:
            text = f.read(_READ_FILE_MAX_CHARS + 1)
            return (
                text[:_READ_FILE_MAX_CHARS],
                len(text) > _READ_FILE_MAX_CHARS,
            )

        chunks: list[str] = []
        remaining = _READ_FILE_MAX_CHARS
        for _ in range(count):
            if remaining == 0:
                return "".join(chunks), bool(f.read(1))
            line = f.readline(remaining + 1)
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
    :class:`_OffsetScanLimitExceeded` when reaching the offset would require
    reading more than the bounded skip budget.
    """
    remaining = _READ_FILE_MAX_SKIP_CHARS
    for _ in range(count):
        while True:
            if remaining <= 0:
                raise _OffsetScanLimitExceeded
            chunk = file.readline(
                min(_READ_FILE_SKIP_CHUNK_CHARS, remaining),
            )
            if not chunk:
                return False
            remaining -= len(chunk)
            if chunk.endswith("\n"):
                break
    return True
