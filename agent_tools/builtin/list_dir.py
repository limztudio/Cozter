"""list_dir: list the entries of a directory inside the workspace."""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    path_property,
    resolve_inside_workspace,
    summarize_path,
)


class ListDirTool(AgentTool):
    name = "list_dir"
    description = (
        "List the entries of a directory in the workspace. Directories"
        " are shown with a trailing slash."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": path_property(
                "Directory path. Defaults to the workspace root if omitted.",
            ),
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum entries to return, default 200, max 1000."
                ),
            },
        },
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path") or "."
        if not isinstance(raw_path, str):
            return "Error: 'path' must be a string"
        target = resolve_inside_workspace(workspace_path, raw_path)
        if not os.path.isdir(target):
            return f"Not a directory: {raw_path}"

        max_results = coerce_int_arg(
            args.get("max_results") or 200,
            default=200,
            minimum=1,
            maximum=1000,
        )

        try:
            lines, entry_count = await asyncio.to_thread(
                self._list_entries, target, max_results,
            )
        except OSError as exc:
            return f"List failed: {exc}"

        if not lines:
            return f"Directory is empty: {raw_path}"

        if entry_count > max_results:
            lines.append(f"... ({entry_count - max_results} more entries)")

        return "\n".join(lines)

    @staticmethod
    def _list_entries(
        target: str, max_results: int,
    ) -> tuple[list[str], int]:
        """Enumerate, sort, and classify entries off the event loop."""
        entries = sorted(os.listdir(target))
        lines = [
            f"{entry}/" if os.path.isdir(os.path.join(target, entry)) else entry
            for entry in entries[:max_results]
        ]
        return lines, len(entries)

    def summarize(self, args: dict) -> str:
        return summarize_path("list_dir", args, ".")
