"""Codex CLI wrapper — runs codex exec and parses JSON event output."""

import asyncio
import json
import logging
from dataclasses import dataclass, field

from . import session

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 50_000
KEEP_RECENT_AFTER_COMPACT = 10

SUMMARY_PROMPT = (
    "Summarize the following conversation history into a concise context block. "
    "Preserve all key decisions, file changes, file paths, tool results, and "
    "the current state of work. This summary will replace the full history "
    "to save space, so include everything needed to continue seamlessly."
)


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


# ------------------------------------------------------------------
# Contextual prompt building
# ------------------------------------------------------------------

def _build_contextual_prompt(
    prompt: str, workspace_path: str, user_id: int,
) -> str:
    """Prepend session history to the prompt so Codex has full context."""
    sid = session.ensure_session(workspace_path, user_id)
    summary = session.get_summary(workspace_path, sid)
    messages = session.get_messages(workspace_path, sid)

    if not summary and not messages:
        return prompt

    parts: list[str] = []

    if summary:
        parts.append("[Session Summary]")
        parts.append(summary)
        parts.append("[End of Session Summary]\n")

    if messages:
        parts.append("[Recent Messages]")
        for msg in messages:
            role = msg.get("role", "?").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("[End of Recent Messages]\n")

    parts.append(
        "Continue the conversation. The user's new message follows.\n"
    )
    parts.append(prompt)

    full = "\n".join(parts)

    # Truncate if too long — drop oldest messages, keep summary + recent
    if len(full) > MAX_HISTORY_CHARS:
        # Rebuild with fewer messages
        budget = MAX_HISTORY_CHARS - len(prompt) - 500  # margin
        history_parts: list[str] = []
        if summary:
            history_parts.append(f"[Session Summary]\n{summary}\n[End of Session Summary]\n")
        # Add messages from newest to oldest until budget runs out
        msg_lines: list[str] = []
        for msg in reversed(messages):
            role = msg.get("role", "?").capitalize()
            content = msg.get("content", "")
            line = f"{role}: {content}"
            if sum(len(l) for l in msg_lines) + len(line) > budget:
                break
            msg_lines.insert(0, line)
        if msg_lines:
            history_parts.append("[Recent Messages]")
            history_parts.extend(msg_lines)
            history_parts.append("[End of Recent Messages]\n")
        history_parts.append(
            "Continue the conversation. The user's new message follows.\n"
        )
        history_parts.append(prompt)
        full = "\n".join(history_parts)

    return full


# ------------------------------------------------------------------
# Main run function
# ------------------------------------------------------------------

async def run(
    prompt: str,
    workspace_path: str,
    user_id: int,
    model: str | None = None,
    approval: str = "auto",
) -> CodexResult:
    """Run ``codex exec --json`` with session history prepended.

    approval maps to sandbox/approval flags:
      - "auto"    → --full-auto
      - "confirm" → --sandbox workspace-write
      - "deny"    → --sandbox read-only
    """
    contextual_prompt = _build_contextual_prompt(prompt, workspace_path, user_id)

    cmd = ["codex", "exec", "--json", "-C", workspace_path]

    if model:
        cmd += ["-m", model]

    if approval == "deny":
        cmd += ["--sandbox", "read-only"]
    elif approval == "confirm":
        cmd += ["--sandbox", "workspace-write"]
    else:
        cmd += ["--full-auto"]

    # Pass prompt via stdin to avoid command-line length limits
    cmd.append("-")

    logger.info("Running codex exec (prompt %d chars, context %d chars)",
                len(prompt), len(contextual_prompt))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1024 * 1024,  # 1 MB stream buffer
    )
    proc.stdin.write(contextual_prompt.encode("utf-8"))
    proc.stdin.close()

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
        result.text = f"Codex exited with code {proc.returncode}"
        if stderr:
            result.text += f"\n{stderr}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    # Log the original prompt (not the contextual one) to session
    _log_to_session(workspace_path, user_id, prompt, result)

    # Auto-compact if threshold reached
    await _maybe_compact(workspace_path, user_id, model)

    # Ensure there's at least one text event
    if not any(e.kind == "text" for e in result.events):
        result.events.append(ChatEvent(kind="text", content=result.text))

    return result


# ------------------------------------------------------------------
# Session logging
# ------------------------------------------------------------------

def _log_to_session(
    workspace_path: str, user_id: int, prompt: str, result: CodexResult,
) -> None:
    """Append the user prompt and AI response to the local session log."""
    try:
        sid = session.ensure_session(workspace_path, user_id)
        session.append_message(workspace_path, sid, {
            "role": "user", "content": prompt,
        })
        session.append_message(workspace_path, sid, {
            "role": "assistant", "content": result.text,
        })
    except Exception:
        logger.debug("Failed to log session", exc_info=True)


# ------------------------------------------------------------------
# Auto-compaction
# ------------------------------------------------------------------

async def _maybe_compact(
    workspace_path: str, user_id: int, model: str | None = None,
) -> None:
    """Summarize session history if the message count crosses the compact interval."""
    try:
        sid = session.ensure_session(workspace_path, user_id)
        total = session.get_total_message_count(workspace_path, sid)
        interval = session.get_compact_interval(workspace_path, sid)

        if interval <= 0 or total < interval:
            return

        # Check if we've already compacted at this threshold
        data = session.load_session(workspace_path, sid)
        if data is None:
            return
        msgs = data.get("messages", [])
        if len(msgs) < interval:
            return  # not enough un-compacted messages

        logger.info("Auto-compact triggered (total=%d, interval=%d)", total, interval)
        await _compact_session(workspace_path, sid, model)
    except Exception:
        logger.debug("Compaction check failed", exc_info=True)


async def _compact_session(
    workspace_path: str, session_id: str, model: str | None = None,
) -> None:
    """Run Codex to summarize the session, then trim messages."""
    summary = session.get_summary(workspace_path, session_id)
    messages = session.get_messages(workspace_path, session_id)

    if not messages:
        return

    # Build the content to summarize
    parts: list[str] = []
    if summary:
        parts.append(f"Previous summary:\n{summary}\n")
    parts.append("Conversation to summarize:")
    for msg in messages:
        role = msg.get("role", "?").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")

    summarize_input = "\n".join(parts)
    full_prompt = f"{SUMMARY_PROMPT}\n\n{summarize_input}"

    cmd = ["codex", "exec", "--ephemeral", "--full-auto", "-C", workspace_path]
    if model:
        cmd += ["-m", model]
    cmd.append("-")

    logger.info("Running compaction for session %s", session_id)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1024 * 1024,  # 1 MB stream buffer
    )
    proc.stdin.write(full_prompt.encode("utf-8"))
    proc.stdin.close()

    # Collect the agent_message text from JSON output
    new_summary = ""
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
            # Non-JSON — might be the plain text output
            if not new_summary:
                new_summary = line
            continue
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    new_summary = text

    await proc.wait()

    if not new_summary:
        # Fallback: read stderr or use old summary
        stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
        logger.warning("Compaction produced no summary (exit %d): %s", proc.returncode, stderr)
        return

    session.set_summary(
        workspace_path, session_id, new_summary,
        keep_recent=KEEP_RECENT_AFTER_COMPACT,
    )
    logger.info("Session %s compacted, summary %d chars", session_id, len(new_summary))


# ------------------------------------------------------------------
# Event parsing
# ------------------------------------------------------------------

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
