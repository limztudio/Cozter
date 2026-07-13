"""GitHub Copilot CLI backend.

Copilot's non-interactive mode (`copilot -p <prompt> --output-format json`)
streams JSONL events to stdout. Unlike codex, copilot:
  - takes the prompt via argv (`-p`), not stdin - prompts are capped to
    the platform's exec limit (Windows' ~32K CreateProcess command line;
    POSIX ARG_MAX, commonly ~2 MB). copilot has no prompt-via-stdin or
    --prompt-file path yet, so argv is the only delivery mechanism.
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
import os
import shutil
import subprocess
import sys

from .base import (
    AgentResult, Backend, ChatEvent, append_text_result, executable_command,
    set_error_result, truncate_status_text,
)

logger = logging.getLogger(__name__)

# Floor applied on every platform: the Windows CreateProcess command line
# caps at 32767 chars for the whole argv, so keep the prompt well under it.
_WINDOWS_PROMPT_CHARS = 28_000


def _max_prompt_chars() -> int:
    """Largest prompt (chars) we can safely pass to copilot via ``-p``.

    copilot delivers the prompt as a single argv value - it has no
    prompt-via-stdin or ``--prompt-file`` path yet - so the OS exec limit
    applies. Windows' CreateProcess caps the whole command line at 32767
    chars; POSIX bounds argv + env combined by ARG_MAX (commonly ~2 MB).
    Use a conservative fraction of ARG_MAX to leave room for env vars, the
    executable path, and the other flags. The old fixed 28K cap truncated
    POSIX prompts far below both the OS limit and agent.py's own 50K
    context budget.
    """
    if sys.platform == "win32":
        return _WINDOWS_PROMPT_CHARS
    try:
        arg_max = os.sysconf("SC_ARG_MAX")
    except (ValueError, OSError, AttributeError):
        arg_max = 0
    if arg_max <= 0:
        return 128_000
    return max(_WINDOWS_PROMPT_CHARS, min(arg_max // 4, 1_000_000))


class CopilotBackend(Backend):
    name = "copilot"
    executable = "copilot"
    # Cozter's sentinel: `auto` is accepted by `--model` but never appears in
    # the CLI's concrete-model list, so it is prepended to whatever the
    # installed binary reports rather than discovered from it.
    default_model = "auto"
    default_summary_model = "claude-haiku-4.5"
    # The high tier stays on the Sonnet line rather than reaching for Opus or
    # Fable: it is meant for genuinely hard work, not for the most expensive
    # model available. Sonnet 5 supersedes the 4.6 that used to sit here.
    tier_models = {
        "low": "claude-haiku-4.5",
        "mid": "gpt-5.4",
        "high": "claude-sonnet-5",
    }
    # No override is represented by Cozter's effort=0 (omit the flag).
    effort_levels = ("low", "medium", "high", "xhigh", "max")

    def __init__(self) -> None:
        # Discovered once per process: the backend is a process-wide
        # singleton (see backends_agent.__init__._BACKENDS), so caching on
        # the instance avoids re-shelling out to `copilot help config` on
        # every picker open. ``None`` means "not yet probed"; a probe that
        # finds nothing falls back to _FALLBACK_MODELS.
        self._cached_models: tuple[str, ...] | None = None

    # ---- model discovery -----------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Models the installed Copilot CLI accepts.

        Discovered at first access by parsing ``copilot help config``, which
        lists the binary's concrete model slugs. This tracks the installed
        CLI version/build rather than a hand-maintained tuple, so the picker
        only ever offers models the CLI on this host actually recognizes.
        ``auto`` is always first. Falls back to :data:`_FALLBACK_MODELS`
        when the CLI is absent or the catalog can't be parsed.
        """
        if self._cached_models is None:
            self._cached_models = self._discover_models()
        return self._cached_models

    def _discover_models(self) -> tuple[str, ...]:
        binary = shutil.which(self.executable)
        if binary is None:
            logger.debug(
                "copilot not on PATH; using fallback model list"
            )
            return _FALLBACK_MODELS
        prefix = executable_command(self.executable)
        try:
            proc = subprocess.run(
                prefix + ["help", "config"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug(
                "copilot help config probe failed (%s); using fallback",
                exc,
            )
            return _FALLBACK_MODELS
        if proc.returncode != 0:
            logger.debug(
                "copilot help config exited %d (%s); using fallback",
                proc.returncode, proc.stderr.strip()[:200],
            )
            return _FALLBACK_MODELS
        models = _parse_help_config_models(proc.stdout)
        if not models:
            logger.debug(
                "copilot help config yielded no model catalog; "
                "using fallback"
            )
            return _FALLBACK_MODELS
        return ("auto", *models)

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
        max_prompt_chars = _max_prompt_chars()
        if len(prompt) > max_prompt_chars:
            # Keep the tail: the user's current message is at the end of the
            # composed prompt, and the head (capability-hint preamble + old context)
            # is the least costly to drop.
            logger.warning(
                "Copilot prompt %d chars exceeds %d-char cap; "
                "dropping oldest %d chars of context",
                len(prompt), max_prompt_chars,
                len(prompt) - max_prompt_chars,
            )
            prompt = prompt[-max_prompt_chars:]

        prefix = executable_command(self.executable)
        cmd: list[str] = [
            *prefix, "--output-format", "json", "--no-color",
        ]
        self.append_model_effort_args(cmd, model, effort)

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
            # Own process group so a /stop or inject-restart kills the whole
            # tree, not just the copilot parent (see
            # utils.terminate_process_group). POSIX only; no-op on Windows.
            start_new_session=os.name != "nt",
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
            content = f"{tool} done"
            if output:
                content += f"\n{truncate_status_text(output)}"
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
            set_error_result(result, msg)
            return

        # Fall through: treat as assistant text if it looks like one.
        text = self._assistant_text(event, etype)
        if text:
            append_text_result(result, text)
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


def _parse_help_config_models(text: str) -> tuple[str, ...]:
    """Extract concrete model slugs from ``copilot help config`` output.

    The ``model`` setting section renders each choice as an indented
    ``- "slug"`` line. Returns slugs in document order, de-duplicated.
    """
    models: list[str] = []
    seen: set[str] = set()
    in_model_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("`model`:"):
            in_model_section = True
            continue
        if not in_model_section:
            continue
        if stripped.startswith('- "') and stripped.endswith('"'):
            slug = stripped[3:-1]
            if slug and slug not in seen:
                seen.add(slug)
                models.append(slug)
            continue
        # The first non-empty, non-choice line after choices ends the section.
        if models and stripped:
            break
    return tuple(models)


# Safety net used when the Copilot CLI is absent or its catalog can't be
# parsed. Kept current with the 1.0.70 binary so the picker is never empty.
_FALLBACK_MODELS = (
    "auto",
    "claude-sonnet-5",
    "claude-sonnet-4.6",
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "claude-fable-5",
    "claude-opus-4.8",
    "claude-opus-4.8-fast",
    "claude-opus-4.7",
    "claude-opus-4.6",
    "claude-opus-4.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "kimi-k2.7-code",
)
