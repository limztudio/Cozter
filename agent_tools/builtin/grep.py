"""grep: regex search across workspace file contents."""

from __future__ import annotations

import os
import pathlib
import re

from ..base import AgentTool, resolve_inside_workspace

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
    parameters = {
        "type": "object",
        "properties": {
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
        "required": ["pattern"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        pattern_str = args.get("pattern")
        if not isinstance(pattern_str, str) or not pattern_str.strip():
            return "Error: 'pattern' must be a non-empty string"
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

        max_results = args.get("max_results") or 50
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 50
        max_results = max(1, min(max_results, 200))

        abs_ws = os.path.realpath(workspace_path)
        results: list[str] = []
        try:
            for fpath in pathlib.Path(search_root).glob(file_glob):
                if not fpath.is_file():
                    continue
                real = os.path.realpath(str(fpath))
                if not (real == abs_ws or real.startswith(abs_ws + os.sep)):
                    continue
                try:
                    if fpath.stat().st_size > _GREP_MAX_FILE_BYTES:
                        continue
                    raw = fpath.read_bytes()
                except OSError:
                    continue
                if b"\x00" in raw[:8192]:
                    continue  # likely binary
                content = raw.decode("utf-8", errors="replace")
                rel = os.path.relpath(str(fpath), abs_ws)
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
