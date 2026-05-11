"""Codex CLI backend."""

import asyncio
import logging

from .base import (
    AgentResult, Backend, ChatEvent, resolve_executable_prefix,
)

logger = logging.getLogger(__name__)


class CodexBackend(Backend):
    name = "codex"
    executable = "codex"
    available_models = (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2-codex",
        "gpt-5.2",
    )
    default_model = "gpt-5.5"
    default_summary_model = "gpt-5.3-codex"

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
    ) -> asyncio.subprocess.Process:
        prefix = resolve_executable_prefix("codex") or ["codex"]
        cmd = [*prefix, "exec", "--ephemeral", "--json", "-C", workspace_path]
        if model:
            cmd += ["-m", model]

        if compaction or approval == "full":
            # Compaction is a trusted internal LLM call with no tool use;
            # bypass avoids sandbox interference with model API access.
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        elif approval == "deny":
            cmd += ["--sandbox", "read-only"]
        elif approval == "confirm":
            cmd += ["--sandbox", "workspace-write"]
        else:
            cmd += ["--full-auto"]
        cmd.append("-")  # read prompt from stdin

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()
        return proc

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")
        item = event.get("item", {})
        item_type = item.get("type", "")

        if etype == "item.completed":
            if item_type == "agent_message":
                text = item.get("text", "")
                if text:
                    result.text = text
                    result.events.append(ChatEvent(kind="text", content=text))

            elif item_type == "command_execution":
                cmd = item.get("command", "?")
                exit_code = item.get("exit_code", "?")
                output = item.get("aggregated_output", "")
                summary = f"$ {cmd} (exit {exit_code})"
                if output:
                    if len(output) > 200:
                        output = output[:200] + "..."
                    summary += f"\n{output}"
                result.events.append(ChatEvent(kind="tool", content=summary))

            elif item_type == "file_change":
                for ch in item.get("changes", []):
                    path = ch.get("path", "?")
                    kind = ch.get("kind", "?")
                    result.events.append(ChatEvent(
                        kind="file",
                        content=f"📄 {kind}: {path}",
                    ))

        elif etype == "turn.failed":
            err_obj = event.get("error")
            if isinstance(err_obj, dict):
                err = err_obj.get("message") or "Unknown error"
            elif isinstance(err_obj, str):
                err = err_obj
            else:
                err = "Unknown error"
            result.text = f"Error: {err}"
            result.events.append(ChatEvent(kind="text", content=result.text))

        elif etype == "error":
            msg = event.get("message", "Unknown error")
            logger.warning("Codex stream error: %s", msg)

    def extract_agent_text(self, event: dict) -> str | None:
        if event.get("type") != "item.completed":
            return None
        item = event.get("item", {})
        if item.get("type") != "agent_message":
            return None
        return item.get("text") or None
