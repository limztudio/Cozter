"""write_file: overwrite a file with new content, creating parent dirs."""

from __future__ import annotations

import os

from ..base import (
    AgentTool,
    ensure_parent_dir,
    object_parameters,
    resolve_inside_workspace,
    summarize_path,
)


class WriteFileTool(AgentTool):
    name = "write_file"
    file_action = "write"
    description = (
        "Write *content* to *path*, creating parent dirs as needed."
        " Overwrites any existing file."
    )
    parameters = object_parameters(
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        target = resolve_inside_workspace(workspace_path, args.get("path", ""))
        content = args.get("content")
        if not isinstance(content, str):
            return "Error: 'content' must be a string"
        # Opening a FIFO or device for writing can block the bot's event loop
        # indefinitely. This tool is intentionally for ordinary workspace
        # files, so refuse existing special paths rather than treating them
        # as a writable text file.
        if os.path.exists(target) and not os.path.isfile(target):
            return f"Error: not a regular file: {args.get('path')}"
        ensure_parent_dir(target)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {args.get('path')}"

    def summarize(self, args: dict) -> str:
        return summarize_path("write_file", args)
