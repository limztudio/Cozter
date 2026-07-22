"""tree: bounded directory-structure overview for codebase orientation."""

from __future__ import annotations

import asyncio
import os

from ..base import (
    DISCOVERY_SKIP_DIRS,
    AgentTool,
    coerce_int_arg,
    object_parameters,
    resolve_inside_workspace,
    summarize_path,
)


class TreeTool(AgentTool):
    name = "tree"
    order = 30  # group with the discovery tools (list_dir/glob/grep)
    description = (
        "Show the workspace's directory structure as an indented tree"
        " (directories first, then files), for quickly orienting in a"
        " codebase. Noise dirs (.git, node_modules, __pycache__, ...) are"
        " skipped and symlinks are not followed. Bounded by *depth* and"
        " *max_entries*; read-only."
    )
    parameters = object_parameters(
        {
            "path": {
                "type": "string",
                "description": (
                    "Subdirectory to root the tree at. Default: the"
                    " workspace root."
                ),
            },
            "depth": {
                "type": "integer",
                "description": "Maximum directory depth to descend. Default 3.",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum entries to list. Default 200.",
            },
        },
        [],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        root = resolve_inside_workspace(workspace_path, args.get("path") or ".")
        if not os.path.isdir(root):
            return f"Not a directory: {args.get('path') or '.'}"
        depth = coerce_int_arg(
            args.get("depth", 3), default=3, minimum=1, maximum=10,
        )
        max_entries = coerce_int_arg(
            args.get("max_entries", 200), default=200, minimum=1, maximum=2000,
        )

        lines: list[str] = []
        truncated = await asyncio.to_thread(
            self._walk, root, 0, depth, "", lines, max_entries,
        )
        if not lines:
            return "(empty)"
        if truncated:
            lines.append(f"... (truncated at {max_entries} entries)")
        return "\n".join(lines)

    def _walk(
        self, path: str, level: int, max_depth: int,
        indent: str, lines: list[str], max_entries: int,
    ) -> bool:
        """Append *path*'s tree to *lines*; return True if the cap was hit."""
        try:
            with os.scandir(path) as it:
                # Directories first, then files, each alphabetically.
                entries = sorted(
                    it,
                    key=lambda e: (not e.is_dir(follow_symlinks=False), e.name),
                )
        except OSError:
            return False
        for entry in entries:
            if len(lines) >= max_entries:
                return True
            is_dir = entry.is_dir(follow_symlinks=False)
            if is_dir and entry.name in DISCOVERY_SKIP_DIRS:
                continue
            lines.append(f"{indent}{entry.name}{'/' if is_dir else ''}")
            if (
                is_dir
                and level + 1 < max_depth
                and self._walk(
                    entry.path, level + 1, max_depth,
                    indent + "  ", lines, max_entries,
                )
            ):
                return True
        return False

    def summarize(self, args: dict) -> str:
        return summarize_path("tree", args)
