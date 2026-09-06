"""Grok Build CLI backend.

Grok Build's headless ``streaming-messages-json`` output follows the
Anthropic Messages-style JSONL shape.  It carries whole assistant messages,
tool uses, and a terminal ``result`` event, which makes it a better fit for
Cozter than the delta-oriented ``streaming-json`` format:

  - ``assistant.message.content`` contains ``text`` and ``tool_use`` blocks
  - ``user.message.content`` contains tool results (not shown as chat status)
  - ``result`` carries the final aggregate reply plus usage and cost metadata

The CLI receives its prompt in argv through ``-p``.  Its documented headless
permission modes map Cozter's four levels to Grok's always-approve, auto, and
strict deny-by-default modes.  Restricted calls additionally use Grok's
read-only OS sandbox and allowlist only non-mutating workspace tools.
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import threading
import time

from .base import (
    MODEL_CATALOG_TTL_SEC,
    AgentResult,
    Backend,
    ChatEvent,
    append_text_result,
    create_captured_subprocess,
    executable_command,
    normalize_error_message,
    set_error_result,
)

logger = logging.getLogger(__name__)

# ``grok models`` is account-aware, but the picker must remain useful when
# Grok is absent, unauthenticated, or its model-list command is unavailable.
# Keep only currently documented CLI models in the conservative fallback.
_FALLBACK_MODELS = ("grok-4.6", "grok-4.5")
_MODEL_DISCOVERY_TIMEOUT_SEC = 15
_MODEL_LINE_RE = re.compile(r"^\s*[*-]\s+(\S+)\s*(?:\([^)]*\))?\s*$")


def _parse_models_output(output: str | bytes) -> tuple[str, ...]:
    """Extract model IDs from the human-readable ``grok models`` listing.

    Current Grok Build releases intentionally expose this command as plain
    text rather than JSON.  Only accept entries after its ``Available models``
    heading so login banners and diagnostics cannot become selectable models.
    """
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not isinstance(output, str):
        return ()

    models: list[str] = []
    seen: set[str] = set()
    reading_models = False
    for line in output.splitlines():
        if line.strip().casefold() == "available models:":
            reading_models = True
            continue
        if not reading_models:
            continue
        match = _MODEL_LINE_RE.match(line)
        if match is None:
            # The list is a contiguous block. A blank separator or another
            # diagnostic heading ends it without interpreting later output.
            if not line.strip() or not line.startswith((" ", "\t")):
                break
            continue
        model = match.group(1).strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return tuple(models)


class GrokBackend(Backend):
    name = "grok"
    executable = "grok"
    default_model = "grok-4.6"
    default_summary_model = "grok-4.6"
    # Grok Build's per-model fallback menu exposes these four levels. The
    # parser also accepts power-user values such as ``minimal`` and ``max``,
    # but models that do not publish their own effort menu need not support
    # them. Keep Cozter's percentage picker on the safe shared subset.
    # effort=0 still means do not override the model.
    effort_levels = ("low", "medium", "high", "xhigh")
    # Grok's model catalog is account-dependent, so do not route flexible's
    # low tier to a potentially unavailable pinned model before its picker
    # refreshes. Every unset tier uses the policy-safe default_model.
    tier_models: dict[str, str] = {}
    permission_arg_sets = {
        "full": ("--always-approve",),
        "auto": ("--permission-mode", "auto", "--sandbox", "workspace"),
        "restricted": (
            "--permission-mode",
            "dontAsk",
            "--sandbox",
            "read-only",
            "--tools",
            "read_file,grep,list_dir",
        ),
    }

    def __init__(self) -> None:
        self._cached_models: tuple[str, ...] | None = None
        self._catalog_expires_at = 0.0
        self._model_catalog_lock = threading.Lock()

    # ---- model discovery -----------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Return Grok models available to the currently authenticated CLI."""
        return self._model_catalog()

    def _model_catalog(self) -> tuple[str, ...]:
        now = time.monotonic()
        if self._cached_models is not None and now < self._catalog_expires_at:
            return self._cached_models

        with self._model_catalog_lock:
            now = time.monotonic()
            if self._cached_models is None or now >= self._catalog_expires_at:
                self._cached_models = self._discover_models()
                self._catalog_expires_at = time.monotonic() + MODEL_CATALOG_TTL_SEC
        return self._cached_models

    def _discover_models(self) -> tuple[str, ...]:
        if shutil.which(self.executable) is None:
            logger.debug("grok not on PATH; using fallback model list")
            return _FALLBACK_MODELS
        try:
            proc = subprocess.run(
                [*executable_command(self.executable), "models"],
                capture_output=True,
                timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("grok models probe failed (%s); using fallback", exc)
            return _FALLBACK_MODELS
        if proc.returncode != 0:
            logger.debug("grok models exited %s; using fallback", proc.returncode)
            return _FALLBACK_MODELS
        models = _parse_models_output(proc.stdout)
        if not models:
            logger.debug("grok models output was unrecognised; using fallback")
            return _FALLBACK_MODELS
        return models

    # ---- launch ---------------------------------------------------------

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
        cmd: list[str] = [
            *executable_command(self.executable),
            "--cwd",
            workspace_path,
            "--output-format",
            "streaming-messages-json",
        ]
        self.append_launch_options(cmd, model, effort, approval)
        # Grok's documented non-interactive entry point. Keep it last so
        # model/permission flags cannot be parsed as prompt text.
        cmd += ["-p", prompt]
        return await create_captured_subprocess(
            cmd,
            cwd=workspace_path,
            # Keep launched shell commands in an owned POSIX process group;
            # /stop and /inject can then stop the complete agent tree.
            start_new_session=os.name != "nt",
        )

    # ---- streaming event parsing ---------------------------------------

    _FILE_TOOL_NAMES = frozenset(
        {
            "search_replace",
            "write_file",
            "create_file",
            "edit_file",
        }
    )

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")

        if etype == "assistant":
            self._handle_assistant_message(event, result)
            return

        if etype == "result":
            self._handle_result(event, result)
            return

        if etype == "error":
            message = self._error_message(event)
            if result.text:
                # A late provider error must be recorded, but must not erase
                # a useful model reply that was already streamed.
                result.error = normalize_error_message(message)
            else:
                set_error_result(result, message)
            return

        # ``system``, ``user`` (tool result), reasoning, and metadata events
        # are useful to Grok's own transcript but do not improve Cozter's
        # compact live status display.
        if etype not in {
            "system",
            "user",
            "usage",
            "available_commands",
            "plan",
            "auto_compact_start",
            "auto_compact_end",
            "thought",
        }:
            logger.debug(
                "Grok: unhandled event type=%r keys=%r",
                etype,
                list(event.keys()),
            )

    def extract_agent_text(self, event: dict) -> str | None:
        """Return Grok's terminal reply for internal summary calls."""
        etype = event.get("type", "")
        if etype == "result" and not event.get("is_error"):
            text = event.get("result")
            return text if isinstance(text, str) and text else None
        if etype != "assistant":
            return None
        message = event.get("message") or {}
        if not isinstance(message, dict):
            return None
        return self._message_text(message)

    def _handle_assistant_message(
        self,
        event: dict,
        result: AgentResult,
    ) -> None:
        message = event.get("message") or {}
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if isinstance(content, str):
            if content:
                append_text_result(result, content)
            return
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    append_text_result(result, text)
            elif block_type == "tool_use":
                self._append_tool_event(block, result)

    def _handle_result(self, event: dict, result: AgentResult) -> None:
        usage = event.get("usage")
        if isinstance(usage, dict):
            result.usage = dict(usage)
            cost = event.get("total_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                result.usage["total_cost_usd"] = cost

        if event.get("is_error"):
            message = self._error_message(event)
            if result.text:
                result.error = normalize_error_message(message)
            else:
                set_error_result(result, message)
            return

        # In normal streams the last assistant message has already supplied
        # this reply. Keep the terminal field as a compatibility fallback for
        # a truncated/older stream that emits only ``result``.
        text = event.get("result")
        if (
            isinstance(text, str)
            and text
            and not any(item.kind == "text" for item in result.events)
        ):
            append_text_result(result, text)

    @staticmethod
    def _error_message(event: dict) -> str:
        """Return Grok's human-readable error across its two envelopes.

        Stream-level errors use ``message``.  Terminal headless failures,
        including model-selection failures, instead expose one or more strings
        under ``errors`` and leave both ``error`` and ``result`` absent.  Do
        not stringify provider objects: an absent usable message must retain
        the common ``Unknown error`` fallback instead of leaking a Python
        representation into a chat reply.
        """
        for key in ("message", "error", "result"):
            message = event.get(key)
            if isinstance(message, str) and message.strip():
                return message
        errors = event.get("errors")
        if isinstance(errors, list):
            for message in errors:
                if isinstance(message, str) and message.strip():
                    return message
        return "Unknown error"

    def _append_tool_event(self, block: dict, result: AgentResult) -> None:
        name = block.get("name") or "tool"
        if not isinstance(name, str):
            name = "tool"
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        command = tool_input.get("command") or tool_input.get("cmd")
        path = (
            tool_input.get("path")
            or tool_input.get("file_path")
            or tool_input.get("file")
        )
        if isinstance(command, str) and command:
            summary = f"$ {command}"
        elif isinstance(path, str) and path:
            summary = f"{name}: {path}"
        else:
            summary = name
        result.events.append(ChatEvent(kind="tool", content=summary))

        if name in self._FILE_TOOL_NAMES and isinstance(path, str) and path:
            result.events.append(
                ChatEvent(
                    kind="file",
                    content=f"📄 changed: {path}",
                )
            )

    @staticmethod
    def _message_text(message: dict) -> str | None:
        """Flatten a complete Messages-style assistant text response."""
        content = message.get("content")
        if isinstance(content, str):
            return content or None
        if not isinstance(content, list):
            return None
        texts = [
            block["text"]
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"]
            )
        ]
        return "\n".join(texts) or None
