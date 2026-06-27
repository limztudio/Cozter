"""copy_file: duplicate a file within the workspace."""

from __future__ import annotations

import os
import shutil

from ..base import (
    AgentTool,
    ensure_parent_dir,
    resolve_inside_workspace,
    summarize_path_pair,
)


class CopyFileTool(AgentTool):
    name = "copy_file"
    description = (
        "Copy a file within the workspace, preserving its bytes and"
        " metadata. Refuses to copy directories (use bash 'cp -r' for"
        " those). Fails if the destination already exists; parent"
        " directories of the destination are created automatically."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_src = args.get("source", "")
        raw_dst = args.get("destination", "")
        src = resolve_inside_workspace(workspace_path, raw_src)
        dst = resolve_inside_workspace(workspace_path, raw_dst)
        if not os.path.exists(src):
            return f"Source not found: {raw_src}"
        if not os.path.isfile(src):
            return f"Not a file (refusing to copy): {raw_src}"
        if os.path.exists(dst):
            return f"Destination already exists: {raw_dst}"
        try:
            ensure_parent_dir(dst)
            shutil.copy2(src, dst)
        except OSError as exc:
            return f"Copy failed: {exc}"
        return f"Copied: {raw_src} -> {raw_dst}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("copy_file", args)
