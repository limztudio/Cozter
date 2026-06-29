"""glob: find workspace files matching a glob pattern."""

from __future__ import annotations

import os

from ..base import AgentTool, coerce_int_arg, iter_workspace_files


class GlobTool(AgentTool):
    name = "glob"
    description = (
        "Find files in the workspace matching a glob pattern. Supports"
        " ** for recursive matching (e.g. '**/*.py'). Returns sorted"
        " relative paths."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Glob pattern, e.g. '**/*.py' or 'src/*.ts'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum paths to return, default 100, max 500."
                ),
            },
        },
        "required": ["pattern"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return "Error: 'pattern' must be a non-empty string"

        max_results = coerce_int_arg(
            args.get("max_results") or 100,
            default=100,
            minimum=1,
            maximum=500,
        )

        abs_ws = os.path.realpath(workspace_path)
        matches: list[str] = []
        try:
            for _path, rel, _root_rel in iter_workspace_files(
                abs_ws, abs_ws, pattern,
            ):
                matches.append(rel)
                if len(matches) >= max_results:
                    break
        except Exception as exc:
            # Filesystem trouble should surface as a clean model-facing error.
            return f"Glob failed: {exc}"

        if not matches:
            return f"No files matched: {pattern}"

        matches.sort()
        summary = "\n".join(matches)
        if len(matches) >= max_results:
            summary += f"\n(stopped at {max_results} matches)"
        return summary

    def summarize(self, args: dict) -> str:
        return f"glob: {args.get('pattern', '?')}"
