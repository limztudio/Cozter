"""write_file: overwrite a file with new content, creating parent dirs."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    ensure_parent_dir,
    object_parameters,
    resolve_inside_workspace,
    summarize_path,
    write_text_after_edit,
)


class WriteFileTool(AgentTool):
    name = "write_file"
    file_action = "write"
    description = "Write content to *path*."
    parameters = object_parameters(
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        try:
            target = resolve_inside_workspace(
                workspace_path, args.get("path", ""),
            )
        except ValueError as exc:
            return f"Error: {exc}"
        content = args.get("content")
        if not isinstance(content, str):
            return "Error: 'content' must be a string"
        # Refuse FIFOs/devices: opening them for writing can block the loop.
        if os.path.exists(target) and not os.path.isfile(target):
            return f"Error: not a regular file: {args.get('path')}"
        ensure_parent_dir(target)
        # Atomic replace keeps the old file (and its perms) on write failure.
        write_text_after_edit(target, content, uses_crlf=False)
        return f"Wrote {len(content)} chars to {args.get('path')}"

    def summarize(self, args: dict) -> str:
        return summarize_path("write_file", args)
