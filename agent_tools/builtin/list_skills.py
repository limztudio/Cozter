"""Discover authorized workspace and native skills without loading their bodies."""

import asyncio
from typing import Any, ClassVar

from ..base import AgentTool, object_parameters


def _page_integer(args: dict, name: str, default: int, minimum: int, maximum: int | None = None) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer")
    if value < minimum or maximum is not None and value > maximum:
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"'{name}' must be {bound}")
    return value


class ListSkillsTool(AgentTool):
    name = "list_skills"
    description = (
        "List discovered workspace and authorized native skills with their exact names,"
        " descriptions and revisions. Load relevant instructions with read_skill;"
        " follow the returned offset to list remaining skills."
    )
    parallel_safe = True
    max_result_chars = 64 * 1024
    parameters: ClassVar[dict[str, Any]] = object_parameters({
        "offset": {"type": "integer", "minimum": 0},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
    }, [])

    async def run(self, workspace_path: str, args: dict) -> str:
        from ... import skills

        try:
            offset = _page_integer(args, "offset", 0, 0)
            count = _page_integer(args, "max_results", 50, 1, 200)
        except ValueError as exc:
            return f"Error: {exc}"
        return await asyncio.to_thread(skills.list_skills, workspace_path, offset=offset, max_results=count)

    def summarize(self, _args: dict) -> str:
        return "list_skills"


if __name__ == "__main__":
    ListSkillsTool.run_as_script()
