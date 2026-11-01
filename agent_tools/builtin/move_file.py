"""move_file: rename or move files / directories within the workspace."""

from __future__ import annotations

import os

from ..owned_worker import run_owned_worker
from ..base import (
    AgentTool,
    ensure_parent_dir,
    move_path_no_clobber,
    no_clobber_result,
    prepare_source_destination,
    source_destination_parameters,
    summarize_path_pair,
)
from ...utils import is_path_within


class MoveFileTool(AgentTool):
    name = "move_file"
    description = "Move/rename a file or directory."
    parameters = source_destination_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        return await run_owned_worker(self._run_sync, workspace_path, args)

    def _run_sync(self, workspace_path: str, args: dict) -> str:
        paths = prepare_source_destination(
            workspace_path, args, preserve_final_symlinks=True,
        )
        if isinstance(paths, str):
            return paths
        raw_source, raw_destination, source_path, destination_path = paths
        if (
            not os.path.islink(source_path)
            and os.path.isdir(source_path)
            and is_path_within(destination_path, source_path)
        ):
            return (
                "Error: destination cannot be inside the source directory: "
                f"{raw_destination}"
            )
        try:
            ensure_parent_dir(destination_path)
            if not move_path_no_clobber(source_path, destination_path):
                return no_clobber_result(raw_destination)
        except OSError as exc:
            return f"Move failed: {exc}"
        return f"Moved: {raw_source} -> {raw_destination}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("move_file", args)
