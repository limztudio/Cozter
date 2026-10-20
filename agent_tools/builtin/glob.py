"""glob: find workspace files matching a glob pattern."""

from __future__ import annotations

import asyncio
import os
from glob import has_magic

from ..base import (
    AgentTool,
    coerce_int_arg,
    git_discovery_files,
    iter_workspace_files,
    object_parameters,
    require_nonempty_string_arg,
    resolve_inside_workspace,
    summarize_arg,
)
from ..discovery import render_discovery_page


class GlobTool(AgentTool):
    name = "glob"
    description = (
        "Find files by glob (`**` recurses). Use the returned offset to page"
        " remaining matches; restart at offset=0 if files change. Broad"
        " workspace searches use the root repository's Git files; use a"
        " specific path or directory pattern for ignored files or nested repositories."
    )
    parameters = object_parameters(
        {
            "pattern": {"type": "string"},
            "path": {
                "type": "string",
                "description": "Search root; defaults to the workspace.",
            },
            "max_results": {
                "type": "integer",
                "description": "max 500.",
            },
            "offset": {
                "type": "integer", "minimum": 0,
                "description": "Matches to skip; use the next offset returned by the tool.",
            },
        },
        ["pattern"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        pattern, error = require_nonempty_string_arg(args, "pattern")
        if error:
            return error
        assert pattern is not None

        max_results = coerce_int_arg(
            args.get("max_results", 100),
            default=100,
            minimum=1,
            maximum=500,
        )
        offset = coerce_int_arg(args.get("offset", 0), default=0, minimum=0)

        abs_ws = os.path.realpath(workspace_path)
        try:
            root = resolve_inside_workspace(
                abs_ws, args.get("path") or ".",
            )
            source_files = await git_discovery_files(abs_ws, root, pattern)
            matches = await asyncio.to_thread(
                self._find_matches, abs_ws, pattern, max_results + 1, root,
                offset, sort_entries=True, source_files=source_files,
            )
        except Exception as exc:
            return f"Glob failed: {exc}"

        if not matches:
            if offset:
                return f"No more files matched at offset {offset}: {pattern}"
            return f"No files matched: {pattern}"

        return render_discovery_page(matches, offset=offset, max_results=max_results)

    @staticmethod
    def _find_matches(
        workspace_path: str, pattern: str, max_results: int,
        search_root: str | None = None,
        offset: int = 0,
        *, sort_entries: bool = False,
        source_files: list[str] | None = None,
    ) -> list[str]:
        """Synchronously scan for matches; called in a worker thread."""
        # Split off a literal prefix (anchors literal; never split magical ones).
        drive, tail = os.path.splitdrive(pattern)
        prefix, search_pattern = os.path.split(tail)
        while has_magic(prefix):
            prefix, segment = os.path.split(prefix)
            search_pattern = os.path.join(segment, search_pattern)
        root = resolve_inside_workspace(
            workspace_path,
            os.path.join(search_root or workspace_path, drive + prefix or "."),
        )
        matches: list[str] = []
        for index, (_path, rel, _root_rel) in enumerate(iter_workspace_files(
            workspace_path, root, search_pattern, discovery_pattern=pattern,
            sort_entries=sort_entries,
            source_files=source_files,
        )):
            if index < offset:
                continue
            matches.append(rel)
            if len(matches) >= max_results:
                break
        return matches

    def summarize(self, args: dict) -> str:
        return summarize_arg("glob", args, "pattern", default="?")
