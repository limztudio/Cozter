"""Codex CLI wrapper — runs codex exec and parses JSON event output."""

import asyncio
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatEvent:
    """An event produced during a codex exec turn."""
    kind: str  # "tool", "file", "text"
    content: str


@dataclass
class CodexResult:
    """Collected result from a single codex exec run."""
    events: list[ChatEvent] = field(default_factory=list)
    text: str = "(no response)"


# Track whether a user has an active Codex session per workspace.
# Key: (user_id, workspace_path) → True if at least one message was sent.
_active_sessions: dict[tuple[int, str], bool] = {}


def has_session(user_id: int, workspace_path: str) -> bool:
    return _active_sessions.get((user_id, workspace_path), False)


def mark_session(user_id: int, workspace_path: str) -> None:
    _active_sessions[(user_id, workspace_path)] = True


def clear_session(user_id: int, workspace_path: str) -> None:
    _active_sessions.pop((user_id, workspace_path), None)


async def run(
    prompt: str,
    workspace_path: str,
    user_id: int,
    model: str | None = None,
    approval: str = "auto",
) -> CodexResult:
    """Run ``codex exec`` and return parsed events.

    On first message per workspace, starts a new session.
    On follow-up messages, resumes the last session with ``resume --last``.

    approval maps to sandbox/approval flags:
      - "auto"    → --full-auto
      - "confirm" → --sandbox workspace-write
      - "deny"    → --sandbox read-only
    """
    resume = has_session(user_id, workspace_path)

    if resume:
        cmd = ["codex", "exec", "--json", "-C", workspace_path, "resume", "--last"]
    else:
        cmd = ["codex", "exec", "--json", "-C", workspace_path]

    if model:
        cmd += ["-m", model]

    if approval == "deny":
        cmd += ["--sandbox", "read-only"]
    elif approval == "confirm":
        cmd += ["--sandbox", "workspace-write"]
    else:
        cmd += ["--full-auto"]

    cmd.append(prompt)

    logger.info("Running: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    result = CodexResult()

    while True:
        line = await proc.stdout.readline()
        if not line:
            break

        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON line: %s", line)
            continue

        _process_event(event, result)

    await proc.wait()

    stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
    if stderr:
        logger.debug("codex stderr: %s", stderr)

    if proc.returncode != 0 and not result.events:
        # If resume failed (e.g. no previous session), retry without resume
        if resume:
            logger.warning("Resume failed (exit %d), retrying as new session", proc.returncode)
            clear_session(user_id, workspace_path)
            return await run(prompt, workspace_path, user_id, model, approval)

        result.text = f"Codex exited with code {proc.returncode}"
        if stderr:
            result.text += f"\n{stderr}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    # Mark session as active after successful run
    if proc.returncode == 0:
        mark_session(user_id, workspace_path)

    # Ensure there's at least one text event
    if not any(e.kind == "text" for e in result.events):
        result.events.append(ChatEvent(kind="text", content=result.text))

    return result


def _process_event(event: dict, result: CodexResult) -> None:
    """Parse a single JSON event from codex exec output."""
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
            changes = item.get("changes", [])
            for ch in changes:
                path = ch.get("path", "?")
                kind = ch.get("kind", "?")
                result.events.append(ChatEvent(
                    kind="file",
                    content=f"📄 {kind}: {path}",
                ))

    elif etype == "turn.failed":
        err = event.get("error", {}).get("message", "Unknown error")
        result.text = f"Error: {err}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    elif etype == "error":
        msg = event.get("message", "Unknown error")
        logger.warning("Codex stream error: %s", msg)
