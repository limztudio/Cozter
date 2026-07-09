"""edit_file: in-place string replacement, single or bulk."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    apply_string_replacement,
    resolve_inside_workspace,
    summarize_path,
    validate_replacement_strings,
)


class EditFileTool(AgentTool):
    name = "edit_file"
    file_action = "edit"
    description = (
        "In-place string replacement: replace *old_string* with"
        " *new_string* in *path*. By default requires a unique match;"
        " pass *replace_all*=true to replace every occurrence."
    )
    parameters: ClassVar[dict[str, Any]] = {
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
        replacement = validate_replacement_strings(
            args.get("old_string"), args.get("new_string"),
        )
        if isinstance(replacement, str):
            return f"Error: {replacement}"
        old, new = replacement
        if not os.path.isfile(target):
            return f"File not found: {args.get('path')}"
        replace_all = bool(args.get("replace_all", False))
        with open(target, encoding="utf-8", errors="replace") as f:
            original = f.read()
        updated, count, n = apply_string_replacement(
            original, old, new, replace_all=replace_all,
        )
        if count == 0:
            return f"old_string not found in {args.get('path')}"
        if n == 0:
            return (
                f"old_string appears {count} times in {args.get('path')};"
                " include more context or set replace_all=true."
            )
        with open(target, "w", encoding="utf-8") as f:
            f.write(updated)
        suffix = "s" if n != 1 else ""
        return f"Replaced {n} occurrence{suffix} in {args.get('path')}"

    def summarize(self, args: dict) -> str:
        return summarize_path("edit_file", args)
