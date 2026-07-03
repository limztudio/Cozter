"""grep: regex search across workspace file contents."""

from __future__ import annotations

import os
import re

from ..base import (
    AgentTool,
    coerce_int_arg,
    iter_workspace_files,
    object_parameters,
    require_nonempty_string_arg,
    resolve_inside_workspace,
)

# Skip grep on files bigger than this - usually binary or generated.
_GREP_MAX_FILE_BYTES = 1_000_000  # 1 MB

# Per-match-line truncation so one giant minified line can't blow past
# the agent's tool-result cap and hide every other match.
_GREP_MAX_LINE_CHARS = 200


class GrepTool(AgentTool):
    name = "grep"
    description = (
        "Search file contents in the workspace for a regex pattern."
        " Returns matching lines as 'path:lineno: line'. Binary files"
        " and files larger than 1 MB are skipped."
    )
    parameters = object_parameters(
        {
            "pattern": {
                "type": "string",
                "description": "Python regex to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory to search in. Defaults to workspace root."
                ),
            },
            "glob": {
                "type": "string",
                "description": (
                    "Glob restricting which files to search, e.g."
                    " '**/*.py'. Defaults to '**/*'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum matching lines, default 50, max 200."
                ),
            },
        },
        ["pattern"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        pattern_str, error = require_nonempty_string_arg(args, "pattern")
        if error:
            return error
        assert pattern_str is not None  # non-None once error is None
        try:
            regex = re.compile(pattern_str)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        raw_path = args.get("path") or "."
        if not isinstance(raw_path, str):
            raw_path = "."
        search_root = resolve_inside_workspace(workspace_path, raw_path)
        if not os.path.isdir(search_root):
            return f"Not a directory: {raw_path}"

        file_glob = args.get("glob") or "**/*"
        if not isinstance(file_glob, str) or not file_glob:
            file_glob = "**/*"

        max_results = coerce_int_arg(
            args.get("max_results") or 50,
            default=50,
            minimum=1,
            maximum=200,
        )

        results: list[str] = []
        try:
            for fpath, rel, _root_rel in iter_workspace_files(
                workspace_path, search_root, file_glob,
            ):
                try:
                    if os.path.getsize(fpath) > _GREP_MAX_FILE_BYTES:
                        continue
                    with open(fpath, "rb") as f:
                        raw = f.read()
                except OSError:
                    continue
                if b"\x00" in raw[:8192]:
                    continue  # likely binary
                content = raw.decode("utf-8", errors="replace")
                for lineno, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        if len(line) > _GREP_MAX_LINE_CHARS:
                            line = line[:_GREP_MAX_LINE_CHARS] + "..."
                        results.append(f"{rel}:{lineno}: {line}")
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
        except Exception as exc:
            return f"Grep failed: {exc}"

        if not results:
            return f"No matches for pattern: {pattern_str}"

        summary = "\n".join(results)
        if len(results) >= max_results:
            summary += f"\n(stopped at {max_results} matches)"
        return summary

    def summarize(self, args: dict) -> str:
        return f"grep: {args.get('pattern', '?')}"
