"""copy_file: duplicate a file within the workspace."""

from __future__ import annotations

from ..base import (
    AgentTool,
    copy_file_atomically,
    ensure_parent_dir,
    prepare_source_destination,
    source_destination_parameters,
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
    parameters = source_destination_parameters()

    async def run(self, workspace_path: str, args: dict) -> str:
        paths = prepare_source_destination(
            workspace_path, args, file_action="copy",
        )
        if isinstance(paths, str):
            return paths
        raw_src, raw_dst, src, dst = paths
        try:
            ensure_parent_dir(dst)
            # Do not trust the preflight absence check alone: another writer
            # can create a file or symlink before a normal copy opens dst.
            # The shared helper publishes only a completed, no-clobber copy.
            if not copy_file_atomically(src, dst):
                return f"Destination already exists: {raw_dst}"
        except OSError as exc:
            return f"Copy failed: {exc}"
        return f"Copied: {raw_src} -> {raw_dst}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("copy_file", args)
