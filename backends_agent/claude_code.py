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
import glob
import json
import logging
import os
import re

from .base import (
    AgentResult, Backend, ChatEvent, DetachedTaskStatus,
    append_detached_task, append_text_result, create_prompt_subprocess,
    executable_command, set_error_result, truncate_status_text,
)

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_BACKGROUND_ID_RE = re.compile(
    r"(?im)^[ \t]*backgrounded[ \t]*(?:·|\*)[ \t]*"
    r"(?P<id>[A-Za-z0-9][A-Za-z0-9_-]{0,127})[ \t]*$",
)
_BACKGROUND_BASH_RE = re.compile(
    r"(?:^|[\s;&|])claude\b[^\n]*--(?:bg|background)\b",
    re.IGNORECASE,
)
_SAFE_BACKGROUND_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_DETACHED_COMMAND_TIMEOUT_SEC = 30


def _decode_cli_output(value: bytes | None) -> str:
    """Decode one short Claude CLI command stream defensively."""
    return (value or b"").decode("utf-8", errors="replace").strip()


def _background_task_ids(text: str) -> list[str]:
    """Extract only Claude's dedicated background-launch output lines."""
    normalized = _ANSI_ESCAPE_RE.sub("", text).replace("\r\n", "\n")
    ids: list[str] = []
    for match in _BACKGROUND_ID_RE.finditer(normalized):
        task_id = match.group("id")
        if task_id not in ids:
            ids.append(task_id)
    return ids


def _claude_home() -> str:
    """Return Claude Code's user-level persistence root."""
    return os.path.join(os.path.expanduser("~"), ".claude")


def _workspace_contains(workspace_path: str, candidate: object) -> bool:
    """Return whether a provider-reported directory belongs to this workspace."""
    if not isinstance(candidate, str):
        return False
    expected = os.path.realpath(workspace_path)
    actual = os.path.realpath(candidate)
    return actual == expected or actual.startswith(expected + os.sep)


def _local_background_state(
    workspace_path: str, task_id: str,
) -> dict | None:
    """Read a Claude background job's durable local state, if it is trusted."""
    if not _SAFE_BACKGROUND_ID_RE.fullmatch(task_id):
        return None
    path = os.path.join(_claude_home(), "jobs", task_id, "state.json")
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or not _workspace_contains(
        workspace_path, state.get("cwd"),
    ):
        return None
    return state


def _local_background_status(
    workspace_path: str, task_id: str,
) -> DetachedTaskStatus | None:
    state = _local_background_state(workspace_path, task_id)
    if state is None:
        return None
    value = state.get("state")
    if not isinstance(value, str) or not value:
        return None
    return DetachedTaskStatus(state=value)


def _transcript_text_from_content(content: object) -> str:
    """Flatten one persisted Claude assistant message's visible text blocks."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def _local_background_output(workspace_path: str, task_id: str) -> str:
    """Read the visible result from Claude's durable background transcript.

    ``claude logs`` renders an interactive terminal transcript on current CLI
    versions, which is unsuitable for a chat callback. Claude's persisted
    JSONL transcript is the same durable source the supervisor resumes from
    and preserves just the assistant's actual response text.
    """
    state = _local_background_state(workspace_path, task_id)
    if state is None:
        return ""
    session_id = state.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        return ""
    if os.path.basename(session_id) != session_id:
        return ""
    project_root = os.path.realpath(os.path.join(_claude_home(), "projects"))
    paths: list[str] = []
    linked_path = state.get("linkScanPath")
    if isinstance(linked_path, str):
        real_linked_path = os.path.realpath(linked_path)
        if (
            real_linked_path.startswith(project_root + os.sep)
            and os.path.basename(real_linked_path) == f"{session_id}.jsonl"
        ):
            paths.append(real_linked_path)
    pattern = os.path.join(_claude_home(), "projects", "*", f"{session_id}.jsonl")
    paths.extend(path for path in glob.glob(pattern) if path not in paths)
    for path in paths:
        texts: list[str] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    message = item.get("message")
                    if not isinstance(message, dict) or message.get("role") != "assistant":
                        continue
                    text = _transcript_text_from_content(message.get("content"))
                    if text:
                        texts.append(text)
        except OSError:
            continue
        if texts:
            return "\n\n".join(texts)

    output = state.get("output")
    if isinstance(output, dict):
        summary = output.get("result")
        if isinstance(summary, str):
            return summary.strip()
    return ""


async def _run_claude_command(
    cmd: list[str], *, cwd: str,
) -> tuple[int, str, str]:
    """Run a short Claude control command without owning its worker tree."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_DETACHED_COMMAND_TIMEOUT_SEC,
        )
    except TimeoutError as exc:
        # Do not kill the command's process group here. ``claude --bg``
        # hands a worker to Claude's supervisor; timing out while the
        # launcher is slow must not terminate the detached session itself.
        proc.kill()
        await proc.wait()
        raise RuntimeError("Claude Code detached-task command timed out") from exc
    return proc.returncode or 0, _decode_cli_output(stdout), _decode_cli_output(stderr)


class ClaudeCodeBackend(Backend):
    name = "claude_code"
    executable = "claude"
    supports_detached_tasks = True
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

    async def launch_detached(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        effort: int = 0,
    ) -> str:
        """Start a whole Claude Code session through ``claude --bg``.

        Background sessions are intentionally *not* print/stream-json runs:
        Claude Code rejects that combination because its supervisor needs a
        persistent interactive-session transcript to host the detached worker.
        """
        cmd: list[str] = [*executable_command(self.executable), "--bg"]
        self.append_model_effort_args(cmd, model, effort)

        if approval == "full":
            cmd.append("--dangerously-skip-permissions")
        elif approval == "deny":
            cmd += ["--permission-mode", "plan"]
        else:
            if approval == "confirm":
                logger.info(
                    "Claude Code background sessions cannot surface a "
                    "chat approval dialog; using bypassPermissions",
                )
            cmd += ["--permission-mode", "bypassPermissions"]
        # ``--bg`` takes a positional prompt, not ``--print``/stdin.
        cmd.append(prompt)

        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        combined = "\n".join(part for part in (stdout, stderr) if part)
        if returncode != 0:
            raise RuntimeError(
                "Claude Code could not start a background session"
                + (f": {combined}" if combined else ""),
            )
        task_ids = _background_task_ids(combined)
        if len(task_ids) != 1:
            raise RuntimeError(
                "Claude Code started a background session but did not report "
                "one unambiguous task id",
            )
        return task_ids[0]

    async def get_detached_task_status(
        self,
        workspace_path: str,
        task_id: str,
    ) -> DetachedTaskStatus | None:
        """Inspect one Claude supervisor job through its JSON task list."""
        cmd = [
            *executable_command(self.executable),
            "agents", "--json", "--all", "--cwd", workspace_path,
        ]
        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        if returncode != 0:
            logger.warning(
                "Claude Code background task listing failed for %s: %s",
                task_id, stderr or stdout,
            )
            # A transient supervisor failure must not be mistaken for a task
            # disappearing. A completed task may have already retired from
            # the daemon, though, so consult its durable local state first.
            local = _local_background_status(workspace_path, task_id)
            if local is not None:
                return local
            return DetachedTaskStatus("unknown")
        try:
            sessions = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning(
                "Claude Code background task listing was not JSON: %s",
                stdout[:200],
            )
            return DetachedTaskStatus("unknown")
        if not isinstance(sessions, list):
            return DetachedTaskStatus("unknown")
        for item in sessions:
            if (
                not isinstance(item, dict)
                or item.get("kind") != "background"
                or item.get("id") != task_id
            ):
                continue
            if not _workspace_contains(workspace_path, item.get("cwd")):
                logger.warning(
                    "Claude Code task %s belongs to a different workspace",
                    task_id,
                )
                return None
            state = item.get("state")
            if not isinstance(state, str) or not state:
                state = "unknown"
            waiting_for = item.get("waitingFor")
            return DetachedTaskStatus(
                state=state,
                waiting_for=waiting_for if isinstance(waiting_for, str) else "",
            )
        # Claude's daemon can retire a completed worker before the next poll.
        # The persisted state keeps the callback restart-safe across that gap.
        return _local_background_status(workspace_path, task_id)

    async def get_detached_task_output(
        self,
        workspace_path: str,
        task_id: str,
    ) -> str:
        """Retrieve a detached task's visible result without terminal ANSI."""
        local = _local_background_output(workspace_path, task_id)
        if local:
            return local

        # Compatibility fallback for older Claude Code versions that do not
        # persist the current job/transcript layout. Newer versions render a
        # full terminal screen here, hence the durable transcript above.
        cmd = [*executable_command(self.executable), "logs", task_id]
        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        if returncode != 0:
            raise RuntimeError(
                "Claude Code could not read background task output"
                + (f": {stderr or stdout}" if stderr or stdout else ""),
            )
        return stdout

    async def stop_detached_task(
        self,
        workspace_path: str,
        task_id: str,
    ) -> bool:
        """Ask Claude's supervisor to stop a detached session."""
        cmd = [*executable_command(self.executable), "stop", task_id]
        returncode, stdout, stderr = await _run_claude_command(
            cmd, cwd=workspace_path,
        )
        if returncode != 0:
            logger.warning(
                "Claude Code could not stop background task %s: %s",
                task_id, stderr or stdout,
            )
            return False
        return True

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

        if etype == "user":
            # Tool results normally stay out of the status display. The one
            # exception is a paired Bash result from ``claude --bg``: it
            # contains the short supervisor task id that Cozter can later
            # validate and monitor after this foreground stream exits.
            self._handle_user_tool_results(event, result)
            return

        # System/init events are noisy for the status display and don't
        # contribute new info; skip.
        if etype == "system":
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

    def _handle_user_tool_results(
        self, event: dict, result: AgentResult,
    ) -> None:
        """Pick up a background-id only from its matching Bash result."""
        message = event.get("message") or {}
        if not isinstance(message, dict):
            return
        blocks = message.get("content") or []
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if (
                not isinstance(tool_use_id, str)
                or tool_use_id not in result.detached_task_tool_use_ids
            ):
                continue
            for task_id in _background_task_ids(
                self._tool_result_text(block.get("content")),
            ):
                append_detached_task(result, self.name, task_id)

    @staticmethod
    def _tool_result_text(content: object) -> str:
        """Flatten Anthropic's string/list shaped Bash tool-result content."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

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
            tool_use_id = block.get("id")
            if (
                isinstance(cmd, str)
                and isinstance(tool_use_id, str)
                and _BACKGROUND_BASH_RE.search(cmd)
            ):
                result.detached_task_tool_use_ids.add(tool_use_id)
            result.events.append(ChatEvent(
                kind="tool",
                content=f"$ {truncate_status_text(cmd)}",
            ))
            return

        # Best-effort summary for generic tools.
        result.events.append(ChatEvent(kind="tool", content=tool))
