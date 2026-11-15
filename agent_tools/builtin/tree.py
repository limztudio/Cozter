"""tree: bounded directory-structure overview for codebase orientation."""

from __future__ import annotations

import asyncio
import heapq
import os

from ..base import (
    DISCOVERY_SKIP_DIRS,
    AgentTool,
    coerce_int_arg,
    get_tool_permission,
    object_parameters,
    resolve_inside_workspace,
    summarize_path,
)
from ..discovery import DISCOVERY_RESULT_MAX, render_discovery_page


class TreeTool(AgentTool):
    name = "tree"
    order = 30  # group with the discovery tools (list_dir/glob/grep)
    description = "Directory tree (dirs first; page remainder when truncated)."
    max_result_chars = DISCOVERY_RESULT_MAX
    parameters = object_parameters(
        {
            "path": {"type": "string"},
            "depth": {"type": "integer"},
            "max_entries": {
                "type": "integer",
                "description": "max 2000.",
            },
            "offset": {
                "type": "integer",
                "description": "Entries to skip; restart at zero if the tree changes.",
            },
        },
        [],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        try:
            root = resolve_inside_workspace(
                workspace_path, args.get("path") or ".",
            )
        except ValueError as exc:
            return f"Error: {exc}"
        if not os.path.isdir(root):
            return f"Not a directory: {args.get('path') or '.'}"
        depth = coerce_int_arg(
            args.get("depth", 3), default=3, minimum=1, maximum=10,
        )
        max_entries = coerce_int_arg(
            args.get("max_entries", 200), default=200, minimum=1, maximum=2000,
        )
        offset = coerce_int_arg(args.get("offset", 0), default=0, minimum=0, maximum=2**63 - 2001)

        lines: list[str] = []
        workspace_root = (
            os.path.realpath(workspace_path)
            if get_tool_permission() != "full" else None
        )
        truncated = await asyncio.to_thread(
            self._walk, root, 0, depth, "", lines, offset + max_entries,
            workspace_root=workspace_root,
        )
        lines = lines[offset:]
        if not lines:
            return f"No more entries at offset {offset}" if offset else "(empty)"
        if truncated:
            lines.append("")
        return render_discovery_page(lines, offset=offset, max_results=max_entries)

    def _walk(
        self, path: str, level: int, max_depth: int,
        indent: str, lines: list[str], max_entries: int,
        *, workspace_root: str | None = None,
    ) -> bool:
        """Append *path*'s tree to *lines*; return True if the cap was hit."""
        try:
            remaining = max_entries - len(lines)

            def entries():
                with os.scandir(path) as scan:
                    for entry in scan:
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError:
                            is_dir = False
                        if is_dir and entry.name in DISCOVERY_SKIP_DIRS:
                            continue
                        yield is_dir, entry.name, entry.path

            # Cap entries; keep one extra for the marker.
            selected = heapq.nsmallest(
                remaining + 1,
                entries(),
                key=lambda entry: (not entry[0], entry[1]),
            )
        except OSError:
            return False
        truncated = len(selected) > remaining
        for is_dir, name, entry_path in selected[:remaining]:
            if len(lines) >= max_entries:
                return True
            lines.append(f"{indent}{name}{'/' if is_dir else ''}")
            if is_dir and level + 1 < max_depth:
                if workspace_root is not None:
                    # Show local entries; check recursion targets.
                    try:
                        if os.path.commonpath((os.path.realpath(entry_path), workspace_root)) != workspace_root:
                            continue
                    except (OSError, ValueError):
                        continue
                if self._walk(
                    entry_path, level + 1, max_depth,
                    indent + "  ", lines, max_entries,
                    workspace_root=workspace_root,
                ):
                    return True
        return truncated

    def summarize(self, args: dict) -> str:
        return summarize_path("tree", args)
