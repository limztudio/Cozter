"""make_dir: create a directory (and missing parents) in the workspace."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    path_parameters,
    resolve_inside_workspace,
    summarize_path,
)


class MakeDirTool(AgentTool):
    name = "make_dir"
    description = "Create a directory incl. parents."
    parameters = path_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path", "")
        if not isinstance(raw_path, str):
            return "Error: 'path' must be a string"
        try:
            target = resolve_inside_workspace(workspace_path, raw_path)
        except ValueError as exc:
            return f"Error: {exc}"
        if os.path.exists(target) and not os.path.isdir(target):
            return f"Path already exists as a file: {raw_path}"
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as exc:
            return f"Create directory failed: {exc}"
        return f"Directory ready: {raw_path}"

    def summarize(self, args: dict) -> str:
        return summarize_path("make_dir", args)
