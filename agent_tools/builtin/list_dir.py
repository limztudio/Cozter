"""list_dir: list the entries of a directory inside the workspace."""

from __future__ import annotations

import os

from ..base import AgentTool, resolve_inside_workspace


class ListDirTool(AgentTool):
    name = "list_dir"
    description = (
        "List the entries of a directory in the workspace. Directories"
        " are shown with a trailing slash."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory path. Defaults to the workspace root if"
                    " omitted."
                ),
            },
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

        max_results = args.get("max_results") or 200
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 200
        max_results = max(1, min(max_results, 1000))

        try:
            entries = sorted(os.listdir(target))
        except OSError as exc:
            return f"List failed: {exc}"

        if not entries:
            return f"Directory is empty: {raw_path}"

        lines: list[str] = []
        for entry in entries[:max_results]:
            full = os.path.join(target, entry)
            lines.append(f"{entry}/" if os.path.isdir(full) else entry)

        if len(entries) > max_results:
            lines.append(f"... ({len(entries) - max_results} more entries)")

        return "\n".join(lines)

    def summarize(self, args: dict) -> str:
        return f"list_dir: {args.get('path', '.')}"
