"""delete_file: remove a single file (refuses directories)."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    path_parameters,
    resolve_workspace_entry,
    summarize_path,
)


class DeleteFileTool(AgentTool):
    name = "delete_file"
    file_action = "delete"
    description = (
        "Delete a file in the workspace. Refuses to delete directories;"
        " use bash 'rm -r' if that's intended."
    )
    parameters = path_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path", "")
        # Keep the final pathname intact: unlinking a symlink must remove the
        # link rather than its in-workspace target. The resolver still checks
        # that an existing link cannot point outside the workspace.
        target = resolve_workspace_entry(workspace_path, raw_path)
        if not os.path.lexists(target):
            return f"File not found: {raw_path}"
        if not (os.path.isfile(target) or os.path.islink(target)):
            return f"Not a file (refusing to delete): {raw_path}"
        try:
            os.remove(target)
        except OSError as exc:
            return f"Delete failed: {exc}"
        return f"Deleted: {raw_path}"

    def summarize(self, args: dict) -> str:
        return summarize_path("delete_file", args)
