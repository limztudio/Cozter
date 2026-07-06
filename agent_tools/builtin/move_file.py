"""move_file: rename or move files / directories within the workspace."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    ensure_parent_dir,
    resolve_source_destination,
    source_destination_parameters,
    summarize_path_pair,
    validate_source_destination,
)


class MoveFileTool(AgentTool):
    name = "move_file"
    description = (
        "Move or rename a file or directory within the workspace."
        " Fails if the destination already exists; parent directories"
        " of the destination are created automatically."
    )
    parameters = source_destination_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_src, raw_dst, src, dst = resolve_source_destination(
            workspace_path, args,
        )
        if error := validate_source_destination(raw_src, raw_dst, src, dst):
            return error
        try:
            ensure_parent_dir(dst)
            os.rename(src, dst)
        except OSError as exc:
            return f"Move failed: {exc}"
        return f"Moved: {raw_src} -> {raw_dst}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("move_file", args)
