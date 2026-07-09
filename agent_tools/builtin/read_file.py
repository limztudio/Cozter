"""read_file: return UTF-8 file contents, optionally a line range."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..base import AgentTool, resolve_inside_workspace, summarize_path


class ReadFileTool(AgentTool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Without arguments,"
        " returns the full file. Pass *offset* and *limit* to read only"
        " a line range (useful for large files)."
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
        if offset is None and limit is None:
            with open(target, encoding="utf-8", errors="replace") as f:
                return f.read()

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

        lines: list[str] = []
        with open(target, encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx < start:
                    continue
                if count is not None and len(lines) >= count:
                    break
                lines.append(line)
        return "".join(lines)

    def summarize(self, args: dict) -> str:
        return summarize_path("read_file", args)
