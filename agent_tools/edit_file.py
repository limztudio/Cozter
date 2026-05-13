"""edit_file: in-place string replacement, single or bulk."""

from __future__ import annotations

import os

from .base import AgentTool, resolve_inside_workspace


class EditFileTool(AgentTool):
    name = "edit_file"
    file_action = "edit"
    description = (
        "In-place string replacement: replace *old_string* with"
        " *new_string* in *path*. By default requires a unique match;"
        " pass *replace_all*=true to replace every occurrence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {
                "type": "boolean",
                "description": (
                    "When true, replace every occurrence. When false"
                    " (default), require a unique match."
                ),
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        target = resolve_inside_workspace(workspace_path, args.get("path", ""))
        old = args.get("old_string")
        new = args.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            return "Error: old_string and new_string must be strings"
        if old == "":
            # Replacing the empty string inserts `new` between every char,
            # which is never what the model wants and corrupts the file.
            return "Error: 'old_string' must not be empty"
        if not os.path.isfile(target):
            return f"File not found: {args.get('path')}"
        replace_all = bool(args.get("replace_all", False))
        with open(target, encoding="utf-8", errors="replace") as f:
            original = f.read()
        count = original.count(old)
        if count == 0:
            return f"old_string not found in {args.get('path')}"
        if count > 1 and not replace_all:
            return (
                f"old_string appears {count} times in {args.get('path')};"
                " include more context or set replace_all=true."
            )
        updated = (
            original.replace(old, new) if replace_all
            else original.replace(old, new, 1)
        )
        with open(target, "w", encoding="utf-8") as f:
            f.write(updated)
        n = count if replace_all else 1
        suffix = "s" if n != 1 else ""
        return f"Replaced {n} occurrence{suffix} in {args.get('path')}"

    def summarize(self, args: dict) -> str:
        return f"edit_file: {args.get('path', '?')}"
