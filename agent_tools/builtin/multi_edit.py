"""multi_edit: apply several in-place edits to one file atomically."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    apply_string_replacement,
    read_text_for_edit,
    resolve_inside_workspace,
    summarize_path,
    validate_replacement_strings,
    write_text_after_edit,
)


class MultiEditTool(AgentTool):
    name = "multi_edit"
    file_action = "edit"
    description = (
        "Apply multiple in-place string replacements to one file in a"
        " single atomic operation. Each edit is applied to the result"
        " of the previous edit. If any edit fails (string missing or"
        " ambiguous without replace_all), no changes are written - so"
        " the file is never left in a partially-edited state."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "description": (
                    "Ordered list of edits. Each edit is applied to the"
                    " result of the previous one."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {
                            "type": "boolean",
                            "description": (
                                "When true, replace every occurrence."
                                " When false (default), require a"
                                " unique match for this edit."
                            ),
                        },
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        raw_path = args.get("path", "")
        target = resolve_inside_workspace(workspace_path, raw_path)
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return "Error: 'edits' must be a non-empty list"
        if not os.path.isfile(target):
            return f"File not found: {raw_path}"

        # Validate every edit up-front so we don't start applying partial
        # edits and then discover a malformed one halfway through.
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return f"Edit {i}: must be an object"
            replacement = validate_replacement_strings(
                edit.get("old_string"), edit.get("new_string"),
            )
            if isinstance(replacement, str):
                return f"Edit {i}: {replacement}"

        loaded = read_text_for_edit(target)
        if isinstance(loaded, str):
            return f"Error: {loaded}"
        content, uses_crlf = loaded

        total_replacements = 0
        for i, edit in enumerate(edits):
            old = edit["old_string"]
            new = edit["new_string"]
            replace_all = bool(edit.get("replace_all", False))
            content, count, replacements = apply_string_replacement(
                content, old, new, replace_all=replace_all,
            )
            if count == 0:
                return f"Edit {i}: old_string not found"
            if replacements == 0:
                return (
                    f"Edit {i}: old_string appears {count} times;"
                    " include more context or set replace_all=true."
                )
            total_replacements += replacements

        write_text_after_edit(target, content, uses_crlf=uses_crlf)

        n = total_replacements
        rsuffix = "s" if n != 1 else ""
        esuffix = "s" if len(edits) != 1 else ""
        return (
            f"Applied {len(edits)} edit{esuffix}"
            f" ({n} replacement{rsuffix}) in {raw_path}"
        )

    def summarize(self, args: dict) -> str:
        edits = args.get("edits") or []
        n = len(edits) if isinstance(edits, list) else 0
        return f"{summarize_path('multi_edit', args)} ({n} edits)"
