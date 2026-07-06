"""copy_file: duplicate a file within the workspace."""

from __future__ import annotations

import shutil

from ..base import (
    AgentTool,
    ensure_parent_dir,
    resolve_source_destination,
    source_destination_parameters,
    summarize_path_pair,
    validate_source_destination,
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
        raw_src, raw_dst, src, dst = resolve_source_destination(
            workspace_path, args,
        )
        if error := validate_source_destination(
            raw_src, raw_dst, src, dst, file_action="copy",
        ):
            return error
        try:
            ensure_parent_dir(dst)
            shutil.copy2(src, dst)
        except OSError as exc:
            return f"Copy failed: {exc}"
        return f"Copied: {raw_src} -> {raw_dst}"

    def summarize(self, args: dict) -> str:
        return summarize_path_pair("copy_file", args)
