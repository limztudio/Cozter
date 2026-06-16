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
    AgentResult, Backend, ChatEvent, resolve_executable_prefix,
)

logger = logging.getLogger(__name__)


class ClaudeCodeBackend(Backend):
    name = "claude_code"
    executable = "claude"
    # Aliases ('sonnet', 'opus', 'haiku') resolve to the latest of each
    # tier; full IDs pin a specific version. Users on newer CLIs can edit
    # the list locally.
    available_models = (
        "sonnet",
        "opus",
        "haiku",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    )
    default_model = "sonnet"
    default_summary_model = "haiku"

    # File-editing tools whose tool_use blocks we surface as kind="file"
    # ChatEvents (the rest of the tool name is kept as the action label).
    _FILE_TOOLS = frozenset({
        "Write", "Edit", "MultiEdit", "NotebookEdit",
    })

    def convert_effort(self, percent: int) -> str | None:
        # Claude Code's only public "effort"-like control is the
        # extended-thinking toggle, whose highest setting is called
        # "max". With no finer gradation, treat the percentage as a
        # binary toggle: above the midpoint, ask for max thinking;
        # otherwise leave it off.
        if percent >= 50:
            return "max"
        return None

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
        native_effort = self.convert_effort(effort)
        if native_effort:
            # Claude Code CLI has no public reasoning-effort flag - the
            # closest control is the model tier (sonnet < opus). Log so
            # the user knows the workspace setting is being dropped,
            # but show the *native* level name from convert_effort so
            # the mapping is visibly applied.
            logger.info(
                "Claude Code has no reasoning_effort flag; ignoring"
                " workspace setting %d%% (native level %r); use a"
                " higher-tier model instead",
                effort, native_effort,
            )
        prefix = resolve_executable_prefix("claude") or ["claude"]
        cmd: list[str] = [
            *prefix,
            "--print",
            "--output-format", "stream-json",
            "--verbose",  # required by claude when stream-json is set
            "--no-session-persistence",  # we manage sessions ourselves
        ]
        if model:
            cmd += ["--model", model]

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

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_path,
        )
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()
        return proc

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
            if event.get("is_error"):
                err = (
                    event.get("error")
                    or event.get("result")
                    or "Unknown error"
                )
                result.error = err
                result.text = f"Error: {err}"
                result.events.append(
                    ChatEvent(kind="text", content=result.text),
                )
                return
            text = event.get("result", "")
            if (
                isinstance(text, str)
                and text
                and not any(e.kind == "text" for e in result.events)
            ):
                result.text = text
                result.events.append(ChatEvent(kind="text", content=text))
            return

        # 'user' (tool results) and 'system' (init/meta) events are noisy
        # for the status display and don't contribute new info; skip.
        if etype in ("user", "system"):
            return

        if etype == "error":
            msg = event.get("message") or "Unknown error"
            result.error = msg
            result.text = f"Error: {msg}"
            result.events.append(ChatEvent(kind="text", content=result.text))
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
        self, block: dict, result: AgentResult,
    ) -> None:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                result.text = text
                result.events.append(ChatEvent(kind="text", content=text))
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
            if len(cmd) > 200:
                cmd = cmd[:200] + "..."
            result.events.append(ChatEvent(kind="tool", content=f"$ {cmd}"))
            return

        # Best-effort summary for generic tools.
        result.events.append(ChatEvent(kind="tool", content=tool))
