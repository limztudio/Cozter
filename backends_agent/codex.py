"""Codex CLI backend."""

import asyncio
import logging

from .base import (
    AgentResult, Backend, ChatEvent, append_text_result,
    create_prompt_subprocess, executable_command, set_error_result,
    truncate_status_text,
)

logger = logging.getLogger(__name__)


class CodexBackend(Backend):
    name = "codex"
    executable = "codex"
    # OpenAI's Codex model snapshot. Keep this to the documented Codex
    # picker models; API-only, private, and provider-routed models can still
    # be added through config.extra_models.
    available_models = (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex-spark",
    )
    default_model = "gpt-5.6-sol"
    default_summary_model = "gpt-5.6-luna"
    tier_models = {
        "low": "gpt-5.4-mini",
        "mid": "gpt-5.6-luna",
        "high": "gpt-5.6-sol",
    }
    common_effort_levels = ("low", "medium", "high", "xhigh")
    effort_levels = (*common_effort_levels, "max", "ultra")
    model_effort_levels = {
        "gpt-5.6-sol": effort_levels,
        "gpt-5.6-terra": effort_levels,
        "gpt-5.6-luna": (*common_effort_levels, "max"),
        "gpt-5.5": common_effort_levels,
        "gpt-5.4": common_effort_levels,
        "gpt-5.4-mini": common_effort_levels,
        "gpt-5.3-codex-spark": common_effort_levels,
    }

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return the effort vocabulary accepted by the selected model."""
        selected_model = model or self.default_model
        return self.model_effort_levels.get(
            selected_model,
            self.common_effort_levels,
        )

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
        prefix = executable_command(self.executable)
        cmd = [*prefix, "exec", "--ephemeral", "--json", "-C", workspace_path]
        self.append_model_effort_args(
            cmd,
            model,
            effort,
            model_flag="-m",
            effort_flag="-c",
            # Codex CLI exposes reasoning effort via the generic config
            # override flag. Unknown levels are rejected by the CLI.
            effort_template="model_reasoning_effort={effort}",
            effort_levels=self.effort_levels_for_model(model),
        )

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
                    append_text_result(result, text)

            elif item_type == "command_execution":
                cmd = item.get("command", "?")
                exit_code = item.get("exit_code", "?")
                output = item.get("aggregated_output", "")
                summary = f"$ {cmd} (exit {exit_code})"
                if output:
                    summary += f"\n{truncate_status_text(output)}"
                result.events.append(ChatEvent(kind="tool", content=summary))

            elif item_type == "file_change":
                for ch in item.get("changes", []):
                    path = ch.get("path", "?")
                    kind = ch.get("kind", "?")
                    result.events.append(ChatEvent(
                        kind="file",
                        content=f"📄 {kind}: {path}",
                    ))

        elif etype == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                result.usage = dict(usage)

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
