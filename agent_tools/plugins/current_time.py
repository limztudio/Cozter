"""Plugin: return the current date/time, optionally in a given timezone."""

from __future__ import annotations

from datetime import datetime

from ..base import AgentTool


class CurrentTimeTool(AgentTool):
    name = "current_time"
    description = (
        "Return the current date and time as an ISO 8601 string."
        " Optionally accepts an IANA timezone name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "Optional IANA timezone (e.g. 'UTC',"
                    " 'America/New_York'). Defaults to system local."
                ),
            },
        },
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path  # not needed for this plugin
        tz_name = args.get("timezone")
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                return datetime.now(ZoneInfo(tz_name)).isoformat()
            except Exception as exc:
                return f"Invalid timezone {tz_name!r}: {exc}"
        return datetime.now().isoformat()


if __name__ == "__main__":
    CurrentTimeTool.run_as_script()
