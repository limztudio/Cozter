"""delete_file: remove a single file (refuses directories)."""

from __future__ import annotations

import os

from ..base import AgentTool, resolve_inside_workspace, summarize_path


class DeleteFileTool(AgentTool):
    name = "delete_file"
    file_action = "delete"
    description = (
        "Delete a file in the workspace. Refuses to delete directories;"
        " use bash 'rm -r' if that's intended."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path", "")
        target = resolve_inside_workspace(workspace_path, raw_path)
        if not os.path.exists(target):
            return f"File not found: {raw_path}"
        if not os.path.isfile(target):
            return f"Not a file (refusing to delete): {raw_path}"
        try:
            os.remove(target)
        except OSError as exc:
            return f"Delete failed: {exc}"
        return f"Deleted: {raw_path}"

    def summarize(self, args: dict) -> str:
        return summarize_path("delete_file", args)
