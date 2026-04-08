"""Codex CLI wrapper — runs codex exec and parses JSON event output."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from . import session

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 50_000

# Per-workspace lock to prevent concurrent session file corruption
_workspace_locks: dict[str, asyncio.Lock] = {}


def _get_workspace_lock(workspace: str) -> asyncio.Lock:
    if workspace not in _workspace_locks:
        _workspace_locks[workspace] = asyncio.Lock()
    return _workspace_locks[workspace]


KEEP_RECENT_AFTER_COMPACT = 10

SUMMARY_PROMPT = (
    "Summarize the following conversation history into a concise context block. "
    "Preserve all key decisions, file changes, file paths, tool results, "
    "scheduled actions, open commitments, and the current state of work. "
    "This summary will replace the full history to save space, so include "
    "everything needed to continue seamlessly."
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
    prompt: str, workspace_path: str, session_id: str,
) -> str:
    """Prepend session history to the prompt so Codex has full context."""
    # Single load — avoids two separate file reads for summary + messages
    data = session.load_session(workspace_path, session_id)
    if data is None:
        return prompt
    summary: str | None = data.get("summary")
    messages: list[dict] = data.get("messages", [])

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

    parts.append("Continue the conversation. The user's new message follows.\n")
    parts.append(prompt)

    full = "\n".join(parts)

    # Truncate if too long — drop oldest messages, keep summary + recent
    if len(full) > MAX_HISTORY_CHARS:
        # Reserve space for the prompt, footer, and summary block
        summary_block = (
            f"[Session Summary]\n{summary}\n[End of Session Summary]\n" if summary else ""
        )
        overhead = len(prompt) + len(summary_block) + 500  # 500-char margin
        msg_budget = MAX_HISTORY_CHARS - overhead

        history_parts: list[str] = []
        if summary_block:
            history_parts.append(summary_block)

        # Add messages newest-to-oldest until the budget is exhausted
        msg_lines: list[str] = []
        used = 0
        for msg in reversed(messages):
            role = msg.get("role", "?").capitalize()
            content = msg.get("content", "")
            line = f"{role}: {content}"
            if used + len(line) > msg_budget:
                break
            msg_lines.insert(0, line)
            used += len(line) + 1  # +1 for the joining newline

        if msg_lines:
            history_parts.append("[Recent Messages]")
            history_parts.extend(msg_lines)
            history_parts.append("[End of Recent Messages]\n")

        history_parts.append("Continue the conversation. The user's new message follows.\n")
        history_parts.append(prompt)
        full = "\n".join(history_parts)

    return full


async def _iter_stream_lines(
    stream: asyncio.StreamReader, chunk_size: int = 64 * 1024,
) -> AsyncIterator[str]:
    """Yield decoded stdout lines without StreamReader.readline() size limits."""
    buffer = bytearray()

    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
            return

        buffer.extend(chunk)
        parts = buffer.split(b"\n")
        buffer = bytearray(parts.pop())

        for part in parts:
            yield part.decode("utf-8", errors="replace")


# ------------------------------------------------------------------
# Main run function
# ------------------------------------------------------------------

async def run(
    prompt: str,
    workspace_path: str,
    user_id: int,
    model: str | None = None,
    summary_model: str | None = None,
    approval: str = "auto",
) -> CodexResult:
    """Run ``codex exec --json`` with session history prepended.

    approval maps to sandbox/approval flags:
      - "auto"    → --full-auto
      - "confirm" → --sandbox workspace-write
      - "deny"    → --sandbox read-only
    """
    # Resolve session ID once here; pass it through to avoid redundant file reads
    session_id = session.ensure_session(workspace_path, user_id)
    contextual_prompt = _build_contextual_prompt(prompt, workspace_path, session_id)

    cmd = ["codex", "exec", "--ephemeral", "--json", "-C", workspace_path]

    if model:
        cmd += ["-m", model]

    if approval == "full":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    elif approval == "deny":
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
    )
    # Use drain() to avoid deadlock when the prompt exceeds the OS pipe buffer
    proc.stdin.write(contextual_prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()
    await proc.stdin.wait_closed()

    result = CodexResult()

    try:
        async for line in _iter_stream_lines(proc.stdout):
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line: %s", line)
                continue

            _process_event(event, result)

        await proc.wait()
    except asyncio.CancelledError:
        logger.info("Codex run cancelled, killing subprocess %d", proc.pid)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        raise

    stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
    if stderr:
        logger.debug("codex stderr: %s", stderr)

    if proc.returncode != 0 and not result.events:
        result.text = f"Codex exited with code {proc.returncode}"
        if stderr:
            result.text += f"\n{stderr}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    # Log the original prompt (not the contextual one) to session.
    # The lock guards the file write only — compaction subprocess runs outside it.
    async with _get_workspace_lock(workspace_path):
        _log_to_session(workspace_path, session_id, prompt, result)

    await _maybe_compact(workspace_path, session_id, summary_model)

    # Ensure there's at least one text event
    if not any(e.kind == "text" for e in result.events):
        result.events.append(ChatEvent(kind="text", content=result.text))

    return result


# ------------------------------------------------------------------
# Session logging
# ------------------------------------------------------------------

def _log_to_session(
    workspace_path: str, session_id: str, prompt: str, result: CodexResult,
) -> None:
    """Append the user prompt and AI response in a single read+write."""
    try:
        session.append_messages(workspace_path, session_id, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _format_session_response(result)},
        ])
    except Exception:
        logger.debug("Failed to log session", exc_info=True)


def _format_session_response(result: CodexResult) -> str:
    """Serialize the full assistant-side turn so later prompts can recover state."""
    parts: list[str] = []
    for event in result.events:
        if event.kind == "text":
            parts.append(event.content)
        elif event.kind == "tool":
            parts.append(f"[Tool Result]\n{event.content}")
        elif event.kind == "file":
            parts.append(f"[File Change]\n{event.content}")

    if not parts and result.text:
        parts.append(result.text)

    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Auto-compaction
# ------------------------------------------------------------------

async def _maybe_compact(
    workspace_path: str, session_id: str, summary_model: str | None = None,
) -> None:
    """Summarize session history if uncompacted messages reach the compact interval.

    The trigger is len(messages) >= interval, checked with a single session load.
    Compaction runs outside the workspace lock so other requests aren't stalled.
    """
    try:
        data = session.load_session(workspace_path, session_id)
        if data is None:
            return
        msgs = data.get("messages", [])
        interval = data.get("compact_interval", 20)
        if interval <= 0 or len(msgs) < interval:
            return

        logger.info("Auto-compact triggered (msgs=%d, interval=%d)", len(msgs), interval)
        existing_summary = data.get("summary") or ""
        new_summary = await _compact_session(workspace_path, session_id, summary_model)
        if not new_summary:
            logger.error("Compaction produced empty summary for session %s", session_id)
            return
        # Reject summaries that are suspiciously short compared to the existing
        # one — a sign of a truncated or failed codex response.
        min_len = max(100, len(existing_summary) // 2)
        if len(new_summary) < min_len:
            logger.error(
                "Compaction summary too short (%d chars, min %d) for session %s — keeping existing",
                len(new_summary), min_len, session_id,
            )
            return
        async with _get_workspace_lock(workspace_path):
            session.set_summary(
                workspace_path, session_id, new_summary,
                keep_recent=KEEP_RECENT_AFTER_COMPACT,
            )
        logger.info("Session %s compacted, summary %d chars", session_id, len(new_summary))
    except Exception:
        logger.error("Compaction check failed", exc_info=True)


async def _compact_session(
    workspace_path: str, session_id: str, summary_model: str | None = None,
) -> str:
    """Run Codex to summarize the session. Returns the summary string (or "" on failure).

    Does NOT write to disk — caller takes the workspace lock and calls set_summary.
    """
    data = session.load_session(workspace_path, session_id)
    if data is None:
        return ""
    messages = data.get("messages", [])
    existing_summary = data.get("summary")

    if not messages:
        return ""

    # Build the content to summarize, staying within a token budget.
    # Large prompts cause the summary model to return truncated/empty output.
    MAX_SUMMARY_CHARS = 80_000  # ~20K tokens — safe for most models

    parts: list[str] = []
    if existing_summary:
        parts.append(f"Previous summary:\n{existing_summary}\n")
    parts.append("Conversation to summarize:")

    # Add messages newest-first until we hit the budget, then reverse
    overhead = len(SUMMARY_PROMPT) + sum(len(p) for p in parts) + 200
    budget = MAX_SUMMARY_CHARS - overhead
    msg_lines: list[str] = []
    used = 0
    for msg in reversed(messages):
        role = msg.get("role", "?").capitalize()
        content = msg.get("content", "")
        line = f"{role}: {content}"
        if used + len(line) > budget:
            break
        msg_lines.insert(0, line)
        used += len(line) + 1
    parts.extend(msg_lines)

    if not msg_lines:
        logger.warning("Session %s messages too large even for a single entry", session_id)
        return ""

    full_prompt = f"{SUMMARY_PROMPT}\n\n" + "\n".join(parts)

    cmd = ["codex", "exec", "--ephemeral", "--full-auto", "-C", workspace_path]
    if summary_model:
        cmd += ["-m", summary_model]
    cmd.append("-")

    logger.info("Running compaction for session %s", session_id)

    COMPACT_TIMEOUT = 120  # seconds

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    proc.stdin.write(full_prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()
    await proc.stdin.wait_closed()

    # Collect agent_message text from JSON event stream; keep the last one
    new_summary = ""
    try:
        async with asyncio.timeout(COMPACT_TIMEOUT):
            async for line in _iter_stream_lines(proc.stdout):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if not new_summary:
                        new_summary = line  # bare-text fallback
                    continue
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            new_summary = text  # keep updating — last one wins

            await proc.wait()
    except TimeoutError:
        logger.error("Compaction timed out after %ds for session %s", COMPACT_TIMEOUT, session_id)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return ""

    if not new_summary:
        stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
        logger.warning("Compaction produced no summary (exit %d): %s", proc.returncode, stderr)
        return ""

    return new_summary


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
