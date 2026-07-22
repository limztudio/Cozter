"""read_file: return bounded UTF-8 file contents, optionally a line range."""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar

from ..base import AgentTool, resolve_inside_workspace, summarize_path


# ``execute_tool`` limits the result sent back to the model, but applying
# that limit after ``read()`` still lets one request allocate a multi-GB log
# or disk image in the bot process. Keep this comfortably above the visible
# result cap so ordinary source files remain useful while bounding both
# memory and synchronous disk work.
_READ_FILE_MAX_CHARS = 128 * 1024
_READ_FILE_SKIP_CHUNK_CHARS = 64 * 1024


class ReadFileTool(AgentTool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace (up to 128 KiB per"
        " call). Pass *offset* and *limit* to read only a line range,"
        " which is useful for large files."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path relative to the workspace root, or an absolute"
                    " path inside the workspace."
                ),
            },
            "offset": {
                "type": "integer",
                "description": "0-based line index to start at. Default 0.",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of lines to return. Omit for the rest"
                    " of the file."
                ),
            },
        },
        "required": ["path"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        target = resolve_inside_workspace(workspace_path, args.get("path", ""))
        if not os.path.isfile(target):
            return f"File not found: {args.get('path')}"

        offset = args.get("offset")
        limit = args.get("limit")
        try:
            start = max(0, int(offset)) if offset is not None else 0
        except (TypeError, ValueError):
            return "Error: 'offset' must be an integer"
        try:
            count = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            return "Error: 'limit' must be an integer"
        if count is not None and count < 0:
            return "Error: 'limit' must be >= 0"

        try:
            text, truncated = await asyncio.to_thread(
                _read_text_range, target, start, count,
            )
        except OSError as exc:
            return f"Read failed: {exc}"

        if truncated:
            text += (
                f"\n... [truncated at {_READ_FILE_MAX_CHARS} characters;"
                " use offset and limit to read another range]"
            )
        return text

    def summarize(self, args: dict) -> str:
        return summarize_path("read_file", args)


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
    """Discard *count* lines without materializing an oversized line."""
    for _ in range(count):
        while True:
            chunk = file.readline(_READ_FILE_SKIP_CHUNK_CHARS)
            if not chunk:
                return False
            if chunk.endswith("\n"):
                break
    return True
