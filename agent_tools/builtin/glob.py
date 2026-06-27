"""glob: find workspace files matching a glob pattern."""

from __future__ import annotations

import os
import pathlib

from ..base import AgentTool, coerce_int_arg


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
            for p in pathlib.Path(abs_ws).glob(pattern):
                # Drop matches that escape the workspace via symlinks.
                real = os.path.realpath(str(p))
                if not (real == abs_ws or real.startswith(abs_ws + os.sep)):
                    continue
                matches.append(os.path.relpath(str(p), abs_ws))
                if len(matches) >= max_results:
                    break
        except Exception as exc:
            # pathlib.Path.glob can raise NotImplementedError on absolute
            # patterns in 3.11/3.12, plus OSError from filesystem trouble;
            # surface either as a clean error to the model.
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
