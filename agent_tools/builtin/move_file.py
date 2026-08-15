"""move_file: rename or move files / directories within the workspace."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    ensure_parent_dir,
    move_path_no_clobber,
    prepare_source_destination,
    source_destination_parameters,
    summarize_path_pair,
)
from ...utils import is_path_within


class MoveFileTool(AgentTool):
    name = "move_file"
    description = (
        "Move or rename a file or directory within the workspace."
        " Fails if the destination already exists; parent directories"
        " of the destination are created automatically."
    )
    parameters = source_destination_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        paths = prepare_source_destination(workspace_path, args)
        if isinstance(paths, str):
            return paths
        raw_src, raw_dst, src, dst = paths
        if os.path.isdir(src) and is_path_within(dst, src):
            return (
                "Error: destination cannot be inside the source directory: "
                f"{raw_dst}"
            )
        try:
            ensure_parent_dir(dst)
            if not move_path_no_clobber(src, dst):
                return f"Destination already exists: {raw_dst}"
        except OSError as exc:
            return f"Move failed: {exc}"
        return f"Moved: {raw_src} -> {raw_dst}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("move_file", args)
