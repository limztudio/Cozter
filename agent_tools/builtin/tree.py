"""tree: bounded directory-structure overview for codebase orientation."""

from __future__ import annotations

import asyncio
import heapq
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
            remaining = max_entries - len(lines)

            def entries():
                with os.scandir(path) as scan:
                    for entry in scan:
                        is_dir = entry.is_dir(follow_symlinks=False)
                        if is_dir and entry.name in DISCOVERY_SKIP_DIRS:
                            continue
                        yield is_dir, entry.name, entry.path

            # The global result cap means no more than ``remaining`` entries
            # from this directory can ever be shown. Keep just one extra so
            # we can retain the existing truncation indicator without sorting
            # every entry in a large generated directory.
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
            if (
                is_dir
                and level + 1 < max_depth
                and self._walk(
                    entry_path, level + 1, max_depth,
                    indent + "  ", lines, max_entries,
                )
            ):
                return True
        return truncated

    def summarize(self, args: dict) -> str:
        return summarize_path("tree", args)
