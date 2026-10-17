"""Progressively read one skill by its discovered, authorized catalog name."""

import asyncio
from typing import Any, ClassVar

from ..base import AgentTool, object_parameters
from .list_skills import _page_integer


class ReadSkillTool(AgentTool):
    name = "read_skill"
    description = (
        "Read a discovered skill by the exact name returned by list_skills."
        " Use offset/max_chars for progressive loading; follow any returned cursor."
        " Pass its revision when paging to detect changed instructions."
        " An optional relative resource reads a UTF-8 supporting reference inside that skill."
        " Arbitrary filesystem paths are not skill names."
    )
    parallel_safe = True
    max_result_chars = 64 * 1024
    parameters: ClassVar[dict[str, Any]] = object_parameters({
        "name": {"type": "string", "minLength": 1},
        "offset": {"type": "integer", "minimum": 0},
        "max_chars": {"type": "integer", "minimum": 1, "maximum": 48_000},
        "revision": {"type": "string", "minLength": 1},
        "resource": {"type": "string", "minLength": 1},
    }, ["name"])

    async def run(self, workspace_path: str, args: dict) -> str:
        from ... import skills

        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            return "Error: 'name' must be a non-empty discovered skill name"
        revision = args.get("revision")
        if revision is not None and (not isinstance(revision, str) or not revision):
            return "Error: 'revision' must be a non-empty string"
        resource = args.get("resource")
        if resource is not None and (not isinstance(resource, str) or not resource.strip()):
            return "Error: 'resource' must be a non-empty relative reference path"
        try:
            offset = _page_integer(args, "offset", 0, 0)
            count = _page_integer(args, "max_chars", 12_000, 1, 48_000)
        except ValueError as exc:
            return f"Error: {exc}"
        return await asyncio.to_thread(
            skills.read_skill, workspace_path, name, offset=offset, max_chars=count,
            revision=revision, resource=resource,
        )

    def summarize(self, _args: dict) -> str:
        return "read_skill"


if __name__ == "__main__":
    ReadSkillTool.run_as_script()
