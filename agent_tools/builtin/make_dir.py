"""make_dir: create a directory (and missing parents) in the workspace."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    path_parameters,
    resolve_tool_path,
    summarize_path,
)


class MakeDirTool(AgentTool):
    name = "make_dir"
    description = "Create a directory incl. parents."
    parameters = path_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        target, raw_path, path_error = resolve_tool_path(
            workspace_path, args.get("path", ""),
        )
        if path_error:
            return path_error
        assert target is not None  # non-None once error is empty
        if os.path.exists(target) and not os.path.isdir(target):
            return f"Path already exists as a file: {raw_path}"
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as exc:
            return f"Create directory failed: {exc}"
        return f"Directory ready: {raw_path}"

    def summarize(self, args: dict) -> str:
        return summarize_path("make_dir", args)
