"""write_file: overwrite a file with new content, creating parent dirs."""

from __future__ import annotations

import os

from .base import AgentTool, resolve_inside_workspace


class WriteFileTool(AgentTool):
    name = "write_file"
    file_action = "write"
    description = (
        "Write *content* to *path*, creating parent dirs as needed."
        " Overwrites any existing file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        target = resolve_inside_workspace(workspace_path, args.get("path", ""))
        content = args.get("content")
        if not isinstance(content, str):
            return "Error: 'content' must be a string"
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {args.get('path')}"

    def summarize(self, args: dict) -> str:
        return f"write_file: {args.get('path', '?')}"
