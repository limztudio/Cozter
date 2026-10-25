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
        raw_src, raw_dst, src, dst = paths
        try:
            ensure_parent_dir(dst)
            if not copy_file_atomically(src, dst):
                return no_clobber_result(raw_dst)
        except OSError as exc:
            return f"Copy failed: {exc}"
        return f"Copied: {raw_src} -> {raw_dst}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("copy_file", args)
