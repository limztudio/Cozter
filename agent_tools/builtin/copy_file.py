"""copy_file: duplicate a file within the workspace."""

from __future__ import annotations

from ..owned_worker import run_owned_worker
from ..base import (
    AgentTool,
    copy_file_atomically,
    ensure_parent_dir,
    no_clobber_result,
    prepare_source_destination,
    source_destination_parameters,
    summarize_path_pair,
)


class CopyFileTool(AgentTool):
    name = "copy_file"
    description = "Copy a file. No dirs, no clobbering."
    parameters = source_destination_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        return await run_owned_worker(self._run_sync, workspace_path, args)

    def _run_sync(self, workspace_path: str, args: dict) -> str:
        paths = prepare_source_destination(
            workspace_path, args, file_action="copy",
        )
        if isinstance(paths, str):
            return paths
        raw_source, raw_destination, source_path, destination_path = paths
        try:
            ensure_parent_dir(destination_path)
            if not copy_file_atomically(source_path, destination_path):
                return no_clobber_result(raw_destination)
        except OSError as exc:
            return f"Copy failed: {exc}"
        return f"Copied: {raw_source} -> {raw_destination}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("copy_file", args)
