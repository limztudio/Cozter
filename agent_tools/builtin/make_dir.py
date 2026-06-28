"""make_dir: create a directory (and missing parents) in the workspace."""

from __future__ import annotations

import os

from ..base import AgentTool, resolve_inside_workspace, summarize_path


class MakeDirTool(AgentTool):
    name = "make_dir"
    description = (
        "Create an empty directory in the workspace, including any"
        " missing parent directories. Idempotent: succeeds even if the"
        " directory already exists. Fails if the path already exists"
        " as a file."
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
        if os.path.exists(target) and not os.path.isdir(target):
            return f"Path already exists as a file: {raw_path}"
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as exc:
            return f"Create directory failed: {exc}"
        return f"Directory ready: {raw_path}"

    def summarize(self, args: dict) -> str:
        return summarize_path("make_dir", args)
