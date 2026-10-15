"""delete_file: remove a single file (refuses directories)."""

from __future__ import annotations

import os

from ..owned_worker import run_owned_worker
from ..base import (
    AgentTool,
    path_parameters,
    resolve_workspace_entry,
    summarize_path,
)


class DeleteFileTool(AgentTool):
    name = "delete_file"
    file_action = "delete"
    description = "Delete a file (bash `rm -r` for dirs)."
    parameters = path_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        return await run_owned_worker(self._run_sync, workspace_path, args)

    def _run_sync(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path", "")
        # Unlink the link, not its target.
        try:
            target = resolve_workspace_entry(workspace_path, raw_path)
        except ValueError as exc:
            return f"Error: {exc}"
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
