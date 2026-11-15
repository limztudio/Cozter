"""list_dir: list the entries of a directory inside the workspace."""

from __future__ import annotations

import asyncio
import heapq
import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    path_property,
    resolve_tool_path,
    summarize_path,
)
from ..discovery import DISCOVERY_RESULT_MAX, render_discovery_page


class ListDirTool(AgentTool):
    name = "list_dir"
    description = "List a directory (page remainder when truncated)."
    max_result_chars = DISCOVERY_RESULT_MAX
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": path_property(),
            "max_results": {
                "type": "integer",
                "description": "max 1000.",
            },
            "offset": {
                "type": "integer",
                "description": "Entries to skip; restart at zero if the directory changes.",
            },
        },
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path") or "."
        target, raw_path, path_error = resolve_tool_path(
            workspace_path, raw_path,
        )
        if path_error:
            return path_error
        assert target is not None
        if not os.path.isdir(target):
            return f"Not a directory: {raw_path}"

        max_results = coerce_int_arg(
            args.get("max_results", 200),
            default=200,
            minimum=1,
            maximum=1000,
        )
        offset = coerce_int_arg(args.get("offset", 0), default=0, minimum=0, maximum=2**63 - 2001)

        try:
            lines, entry_count = await asyncio.to_thread(
                self._list_entries, target, offset + max_results,
            )
        except OSError as exc:
            return f"List failed: {exc}"

        lines = lines[offset:]
        if not lines:
            return f"No more entries at offset {offset}: {raw_path}" if offset else f"Directory is empty: {raw_path}"

        if entry_count > offset + max_results:
            lines.append("")

        return render_discovery_page(lines, offset=offset, max_results=max_results)

    @staticmethod
    def _list_entries(
        target: str, max_results: int,
    ) -> tuple[list[str], int]:
        """Return sorted leading entries without materializing a huge directory.

        We still scan every entry to report the exact omitted count, but a
        bounded heap keeps memory at ``O(max_results)`` and reduces selection
        work from sorting every name to ``O(n log max_results)``.
        """
        entry_count = 0

        def entries():
            nonlocal entry_count
            with os.scandir(target) as scan:
                for entry in scan:
                    entry_count += 1
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        # One bad entry must not abort.
                        is_dir = False
                    yield entry.name, is_dir

        selected = heapq.nsmallest(
            max_results, entries(), key=lambda entry: entry[0],
        )
        lines = [
            f"{name}/" if is_dir else name for name, is_dir in selected
        ]
        return lines, entry_count

    def summarize(self, args: dict) -> str:
        return summarize_path("list_dir", args, ".")
