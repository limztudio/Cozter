"""Read several independent text ranges in one bounded tool round trip."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ..base import AgentTool, object_parameters, path_property, truncate_with_marker
from .read_file import ReadFileTool


_MAX_FILES = 16
_MAX_PARALLEL_READS = 4
_MAX_FILE_CHARS = 32 * 1024
_MAX_RESULT_CHARS = 64 * 1024
_TRUNCATION = (
    "truncated file preview; use read_file with offset/limit for the remaining"
    " lines before treating this file as fully inspected"
)


class ReadFilesTool(AgentTool):
    name = "read_files"
    description = (
        "Read up to 16 independent UTF-8 files or line ranges together, with"
        " bounded parallel reads and results in request order. Each file is"
        " a bounded preview (up to 32K characters; total 64K). Supply offset/limit"
        " for targeted ranges and page any marked remainder with read_file."
        " Uses the same path permissions and image metadata as read_file."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters({
        "files": {
            "type": "array", "minItems": 1, "maxItems": _MAX_FILES,
            "items": object_parameters({
                "path": path_property(),
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            }, ["path"]),
        },
    }, ["files"])

    async def run(self, workspace_path: str, args: dict) -> str:
        files = args.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= _MAX_FILES:
            return f"Error: 'files' must contain between 1 and {_MAX_FILES} file ranges"
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or len(item["path"]) > 4096
            for item in files
        ):
            return "Error: every file range must have a non-empty path of at most 4096 characters"

        # Reserve enough space for every label so large early files cannot
        # hide the result or refusal for a later requested path.
        labels = [
            f"=== {item['path'] if len(item['path']) <= 512 else item['path'][:509] + '...'} ===\n"
            for item in files
        ]
        available = _MAX_RESULT_CHARS - sum(map(len, labels)) - len(files) * 2
        per_file = min(_MAX_FILE_CHARS, max(128, available // len(files)))
        reader = ReadFileTool()

        async def read_one(index: int) -> str:
            result = await reader.run(workspace_path, files[index])
            return labels[index] + truncate_with_marker(
                result, per_file, _TRUNCATION,
            )

        results: list[str] = []
        for start in range(0, len(files), _MAX_PARALLEL_READS):
            results.extend(await asyncio.gather(*(
                read_one(index)
                for index in range(start, min(start + _MAX_PARALLEL_READS, len(files)))
            )))
        return truncate_with_marker(
            "\n\n".join(results), _MAX_RESULT_CHARS,
            "truncated batch; request smaller file ranges for the remainder",
        )

    def summarize(self, args: dict) -> str:
        files = args.get("files")
        return f"read_files: {len(files) if isinstance(files, list) else 0} files"
