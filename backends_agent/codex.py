"""Codex CLI backend."""

import asyncio
import logging

from .base import (
    AgentResult, Backend, ChatEvent, create_prompt_subprocess,
    resolve_executable_prefix, set_error_result,
)

logger = logging.getLogger(__name__)


class CodexBackend(Backend):
    name = "codex"
    executable = "codex"
    # OpenAI's recommended Codex CLI/API models. gpt-5.3-codex is deprecated
    # for ChatGPT sign-in, while Spark remains a Pro research-preview option.
    available_models = (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex-spark",
    )
    default_model = "gpt-5.5"
    default_summary_model = "gpt-5.4-mini"
    # Codex CLI offers 5 levels including ``xhigh``.
    effort_levels = ("minimal", "low", "medium", "high", "xhigh")

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> asyncio.subprocess.Process:
        prefix = resolve_executable_prefix("codex") or ["codex"]
        cmd = [*prefix, "exec", "--ephemeral", "--json", "-C", workspace_path]
        if model:
            cmd += ["-m", model]
        native_effort = self.convert_effort(effort)
        if native_effort:
            # Codex CLI exposes reasoning effort via the generic config
            # override flag. Unknown levels are rejected by the CLI.
            cmd += ["-c", f"model_reasoning_effort={native_effort}"]

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

        return await create_prompt_subprocess(cmd, prompt)

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
            set_error_result(result, err)

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
