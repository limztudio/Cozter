"""GitHub Copilot CLI backend.

Copilot's non-interactive mode (`copilot -p <prompt> --output-format json`)
streams JSONL events to stdout. Unlike codex, copilot:
  - takes the prompt via argv (`-p`), not stdin - prompts are capped to
    stay under Windows' ~32K CreateProcess argv limit
  - has no `-C <dir>` flag - we use the subprocess `cwd` parameter
  - has coarser permission semantics: --allow-all-tools, --yolo, or default
    (which would prompt, unusable in non-interactive mode)

The JSONL event schema is not formally documented. ``parse_event`` and
``extract_agent_text`` use best-effort key probing (``text``/``content``
for assistant messages, ``tool_use``/``tool_call`` for tool invocations)
and log unknown event types so the schema can be refined.
"""

import asyncio
import logging

from .base import (
    AgentResult, Backend, ChatEvent, resolve_executable_prefix,
)

logger = logging.getLogger(__name__)

# Conservative cap: CreateProcess on Windows takes a 32767-char command
# line for the whole argv. Other flags + the executable path need room,
# so keep the prompt itself under ~28K.
_MAX_PROMPT_CHARS = 28_000


class CopilotBackend(Backend):
    name = "copilot"
    executable = "copilot"
    # Copilot's model list is not exposed via --help and evolves over time.
    # These are current defaults; users on newer CLIs can edit the list.
    available_models = (
        "claude-sonnet-4.6",
        "claude-opus-4.7",
        "claude-opus-4.6",
        "gpt-5.4",
        "gpt-5.2",
        "gpt-4o",
        "o3",
    )
    default_model = "claude-opus-4.6"
    default_summary_model = "gpt-5.2"

    # Inherits the base ``convert_effort`` (returns None), since copilot
    # has no public reasoning-effort control.

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
        if effort > 0:
            # Copilot CLI has no reasoning_effort flag; log so the user
            # knows the workspace setting is being dropped.
            logger.info(
                "Copilot has no reasoning_effort flag; ignoring"
                " workspace setting %d%%",
                effort,
            )
        if len(prompt) > _MAX_PROMPT_CHARS:
            # Keep the tail: the user's current message is at the end of the
            # composed prompt, and the head (CAPABILITY_HINT + oldest context)
            # is the least costly to drop.
            logger.warning(
                "Copilot prompt %d chars exceeds %d-char cap; "
                "dropping oldest %d chars of context",
                len(prompt), _MAX_PROMPT_CHARS,
                len(prompt) - _MAX_PROMPT_CHARS,
            )
            prompt = prompt[-_MAX_PROMPT_CHARS:]

        prefix = resolve_executable_prefix("copilot") or ["copilot"]
        cmd: list[str] = [
            *prefix, "--output-format", "json", "--no-color",
        ]
        if model:
            cmd += ["--model", model]

        if compaction or approval == "full":
            cmd.append("--yolo")
        else:
            # "auto" / "confirm" / "deny" all collapse to --allow-all-tools
            # here: copilot cannot interactively prompt a Telegram user, so
            # non-interactive runs must be non-blocking. Log when the user's
            # intent is stricter than what copilot can enforce.
            if approval in ("confirm", "deny"):
                logger.info(
                    "Copilot backend does not support approval=%s; "
                    "falling back to --allow-all-tools",
                    approval,
                )
            cmd.append("--allow-all-tools")

        cmd += ["-p", prompt]

        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_path,
        )

    _TOOL_USE_TYPES = ("tool_use", "tool_call", "tool_start", "tool")
    _TOOL_RESULT_TYPES = ("tool_result", "tool_output", "tool_end")
    _FILE_TYPES = ("file_change", "edit", "file")
    _ERROR_TYPES = ("error", "turn.failed", "failed")
    _ASSISTANT_TYPES = (
        "assistant_message", "agent_message", "message",
        "completion", "response", "text",
    )
    _NON_AGENT_TYPES = frozenset(
        _TOOL_USE_TYPES + _TOOL_RESULT_TYPES + _FILE_TYPES + _ERROR_TYPES
    )

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type") or event.get("event") or ""

        # Typed branches first so a tool event with an "output"/"content"
        # field doesn't get misrouted as assistant text by _extract_text.
        if etype in self._TOOL_USE_TYPES:
            tool = (
                event.get("name")
                or event.get("tool")
                or event.get("tool_name")
                or "?"
            )
            inp = event.get("input") or event.get("args") or {}
            result.events.append(ChatEvent(
                kind="tool", content=self._summarize_tool(tool, inp),
            ))
            return

        if etype in self._TOOL_RESULT_TYPES:
            tool = event.get("name") or event.get("tool") or "tool"
            output = event.get("output") or event.get("result") or ""
            if isinstance(output, (dict, list)):
                output = str(output)
            if len(output) > 200:
                output = output[:200] + "..."
            content = f"{tool} done"
            if output:
                content += f"\n{output}"
            result.events.append(ChatEvent(kind="tool", content=content))
            return

        if etype in self._FILE_TYPES:
            path = (
                event.get("path")
                or event.get("file")
                or event.get("filename")
                or "?"
            )
            kind = event.get("action") or event.get("kind") or "change"
            result.events.append(ChatEvent(
                kind="file", content=f"📄 {kind}: {path}",
            ))
            return

        if etype in self._ERROR_TYPES:
            msg = event.get("message")
            if not msg:
                err = event.get("error")
                if isinstance(err, dict):
                    msg = err.get("message")
                elif isinstance(err, str):
                    msg = err
            if not msg:
                msg = "Unknown error"
            result.error = msg
            result.text = f"Error: {msg}"
            result.events.append(ChatEvent(kind="text", content=result.text))
            return

        # Fall through: treat as assistant text if it looks like one.
        text = self._assistant_text(event, etype)
        if text:
            result.text = text
            result.events.append(ChatEvent(kind="text", content=text))
            return

        logger.debug("Copilot: unhandled event type=%r keys=%r",
                     etype, list(event.keys()))

    def extract_agent_text(self, event: dict) -> str | None:
        etype = event.get("type") or event.get("event") or ""
        # Typed tool/file/error events never carry the agent's final reply.
        if etype in self._NON_AGENT_TYPES:
            return None
        return self._assistant_text(event, etype)

    # -- helpers ----------------------------------------------------------

    def _assistant_text(self, event: dict, etype: str) -> str | None:
        """Return the event's text content if it looks like assistant output."""
        role = event.get("role")
        if role and role not in ("assistant", "model", "agent"):
            return None
        is_assistant_type = (
            etype in self._ASSISTANT_TYPES
            or etype == ""
            or etype == "item.completed"
        )
        if not is_assistant_type:
            return None
        return self._extract_text(event)

    @staticmethod
    def _extract_text(event: dict) -> str | None:
        """Pull text content from an event using best-effort key probing."""
        for key in ("text", "content", "message"):
            val = event.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict):
                inner = val.get("text") or val.get("content")
                if isinstance(inner, str) and inner.strip():
                    return inner
        return None

    @staticmethod
    def _summarize_tool(tool: str, inp: dict) -> str:
        if not isinstance(inp, dict):
            return tool
        # Common patterns: {command, cmd} for shells; {path, file} for edits
        cmd = inp.get("command") or inp.get("cmd")
        if cmd:
            return f"$ {cmd}"
        path = inp.get("path") or inp.get("file") or inp.get("filename")
        if path:
            return f"{tool}: {path}"
        return tool
