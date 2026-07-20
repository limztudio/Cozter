"""Claude Code CLI backend.

Claude Code's non-interactive flow (``claude --print --output-format
stream-json --verbose``) reads the prompt from stdin and streams a
JSONL event sequence:

  - ``{"type": "system", ...}`` - init / metadata
  - ``{"type": "assistant", "message": {...}}`` - assistant turn with
    ``content`` blocks (``text`` / ``tool_use``)
  - ``{"type": "user", "message": {...}}`` - tool-result deliveries
  - ``{"type": "result", "subtype": "success", "result": "...", ...}``
    - terminal event carrying the final aggregated assistant text

Workspace access uses the subprocess ``cwd`` (Claude Code has no -C flag).
Permission modes map to claude's ``--permission-mode`` choices, with the
trusted ``compaction=True`` path collapsing to ``--dangerously-skip-
permissions`` since summarization is a no-tool LLM call.
"""

import asyncio
import logging
import os

from .base import (
    AgentResult, Backend, ChatEvent, append_text_result,
    create_prompt_subprocess, executable_command, set_error_result,
    truncate_status_text,
)

logger = logging.getLogger(__name__)


class ClaudeCodeBackend(Backend):
    name = "claude_code"
    executable = "claude"
    # Claude Code has no safe non-interactive catalog command.  In
    # particular, a managed Bedrock/Vertex/Foundry login cannot be enumerated
    # through Anthropic's public API, and probing candidate IDs can make a
    # billable request.  Keep this curated fallback plus config.extra_models
    # until the CLI exposes an account-aware model-list interface.
    # Mirrors the model registry embedded in the Claude Code CLI. Aliases
    # resolve to the current default for each tier; ``default`` clears a pin
    # and lets Claude Code choose the account-tier default. Full IDs pin a
    # specific version. Mythos stays out of the picker: it ships only to
    # Project Glasswing participants. Users can still add gateway or local IDs
    # through config.extra_models.
    #
    # Three rules the CLI enforces, each of which this tuple has gotten wrong
    # before - check them before adding an entry:
    #   - A dated snapshot exists only where the API publishes one (Opus 4.5,
    #     Sonnet 4.5, Haiku 4.5). From Opus/Sonnet 4.6 on, the ID is undated
    #     and inventing a date suffix 404s.
    #   - ``[1m]`` is only valid on models whose registry entry sets
    #     supports_1m_suffix. Opus 4.8, Sonnet 5, Fable 5, and Mythos 5 are
    #     natively 1M, so they take no suffix (the CLI strips one if given);
    #     note Opus 4.6/4.7 are also native-1M but *do* set the flag, so their
    #     ``[1m]`` variant is real and stays in the picker.
    #   - Fast mode is a session toggle (``/fast``) on Opus 4.8/4.7, not a
    #     model ID. The ``claude-opus-4-*-fast`` strings are retired API IDs:
    #     4.6-fast silently degrades to standard Opus 4.6, and 4.7-fast errors
    #     once removed.
    available_models = (
        "default",
        "sonnet",
        "opus",
        "fable",
        "haiku",
        "best",
        "opusplan",
        "sonnet[1m]",
        "opus[1m]",
        "fable[1m]",
        "opusplan[1m]",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7[1m]",
        "claude-opus-4-6[1m]",
        "claude-sonnet-4-6[1m]",
        "claude-sonnet-4-5-20250929[1m]",
    )
    default_model = "default"
    default_summary_model = "haiku"
    tier_models = {"low": "haiku", "mid": "sonnet", "high": "opus"}
    effort_levels = ("low", "medium", "high", "xhigh", "max")

    # File-editing tools whose tool_use blocks we surface as kind="file"
    # ChatEvents (the rest of the tool name is kept as the action label).
    _FILE_TOOLS = frozenset({
        "Write", "Edit", "MultiEdit", "NotebookEdit",
    })

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
        cmd: list[str] = [
            *prefix,
            "--print",
            "--output-format", "stream-json",
            "--verbose",  # required by claude when stream-json is set
            "--no-session-persistence",  # we manage sessions ourselves
        ]
        self.append_model_effort_args(cmd, model, effort)

        if compaction or approval == "full":
            # Compaction is a trusted internal call; same for the user's
            # full-access mode.
            cmd.append("--dangerously-skip-permissions")
        elif approval == "deny":
            # Plan mode produces a response without performing edits or
            # shell commands - closest analogue to "no tools".
            cmd += ["--permission-mode", "plan"]
        else:
            # Telegram/Slack flows can't surface interactive permission
            # prompts; bypass so the run doesn't block. Log when the
            # user's intent was stricter than what we can enforce.
            if approval == "confirm":
                logger.info(
                    "Claude Code can't prompt for confirmation in"
                    " non-interactive mode; falling back to"
                    " bypassPermissions",
                )
            cmd += ["--permission-mode", "bypassPermissions"]

        return await create_prompt_subprocess(cmd, prompt, cwd=workspace_path)

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")

        if etype == "assistant":
            msg = event.get("message", {}) or {}
            for block in msg.get("content") or []:
                self._handle_assistant_block(block, result)
            return

        if etype == "result":
            # The terminal event. If the assistant streamed text blocks
            # above, we already captured them; otherwise fall back to
            # the cumulative 'result' field.
            usage = event.get("usage")
            if isinstance(usage, dict):
                result.usage = dict(usage)
                cost = event.get("total_cost_usd")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    result.usage["total_cost_usd"] = cost
            if event.get("is_error"):
                err = (
                    event.get("error")
                    or event.get("result")
                    or "Unknown error"
                )
                set_error_result(result, err)
                return
            text = event.get("result", "")
            if (
                isinstance(text, str)
                and text
                and not any(e.kind == "text" for e in result.events)
            ):
                append_text_result(result, text)
            return

        # 'user' (tool results) and 'system' (init/meta) events are noisy
        # for the status display and don't contribute new info; skip.
        if etype in ("user", "system"):
            return

        if etype == "error":
            msg = event.get("message") or "Unknown error"
            set_error_result(result, msg)
            return

        logger.debug(
            "Claude Code: unhandled event type=%r keys=%r",
            etype, list(event.keys()),
        )

    def extract_agent_text(self, event: dict) -> str | None:
        # Compaction prefers the terminal result.result field since it's
        # the aggregated, fully-rendered final reply. Streaming assistant
        # text blocks are partials and may not include the full answer.
        etype = event.get("type", "")
        if etype == "result" and not event.get("is_error"):
            text = event.get("result", "")
            return text if isinstance(text, str) and text else None
        if etype == "assistant":
            msg = event.get("message", {}) or {}
            for block in msg.get("content") or []:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        return text
        return None

    # -- helpers ----------------------------------------------------------

    def _handle_assistant_block(
        self, block: object, result: AgentResult,
    ) -> None:
        # Anthropic allows ``content`` to be either a list of typed blocks
        # or, for plain-text messages, a bare string. Iterating yields
        # dicts normally, but a non-dict entry must not raise here and
        # crash the turn.
        if not isinstance(block, dict):
            return
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                append_text_result(result, text)
            return
        if btype == "tool_use":
            self._emit_tool_event(block, result)

    @staticmethod
    def _emit_tool_event(block: dict, result: AgentResult) -> None:
        tool = block.get("name", "?")
        inp = block.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}

        # File-editing tools: emit kind="file" with the touched path so
        # the bot's "Thinking..." status renders them under the file UX.
        if tool in ClaudeCodeBackend._FILE_TOOLS:
            path = inp.get("file_path") or inp.get("notebook_path") or "?"
            action = "write" if tool == "Write" else "edit"
            result.events.append(ChatEvent(
                kind="file",
                content=f"📄 {action}: {os.path.basename(path)}",
            ))
            return

        # Bash gets the command itself; other tools get just their name.
        if tool == "Bash":
            cmd = inp.get("command") or "?"
            result.events.append(ChatEvent(
                kind="tool",
                content=f"$ {truncate_status_text(cmd)}",
            ))
            return

        # Best-effort summary for generic tools.
        result.events.append(ChatEvent(kind="tool", content=tool))
