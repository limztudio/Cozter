"""Codex CLI wrapper - runs codex exec and parses JSON event output."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
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
    "You are compacting a conversation into two memory layers: a SCRATCH "
    "summary (rewritten each compaction) and LONG-TERM memory (persistent "
    "facts/decisions that survive future compactions).\n\n"
    "=== SCRATCH SUMMARY ===\n"
    "Produce a detailed abstract of the conversation below. This abstract "
    "REPLACES the raw history, so it must be thorough enough to continue "
    "work seamlessly.\n"
    "Capture: user goals and intent (the 'why'), decisions and their "
    "reasoning/tradeoffs, file paths touched and the nature of each change, "
    "concrete tool results and errors, current state (done/in-progress/"
    "blocked), open commitments, and non-obvious constraints or preferences.\n"
    "Drop: greetings, filler, repeated restatements, exploratory turns the "
    "user redirected (keep only the final direction + why), superseded "
    "tool output.\n"
    "Prose paragraphs grouped by topic. Aim for 150–500 words. "
    "Prefer specificity (names, paths, values) over generic phrasing.\n\n"
    "=== LONG-TERM MEMORY ===\n"
    "Long-term items are durable facts the user would want remembered across "
    "many compactions: explicit preferences, architectural decisions, "
    "invariants, stable project facts, hard constraints. They are NOT a "
    "second summary - each item is a single self-contained sentence.\n"
    "ADD an item only when the conversation establishes something durable "
    "that is not already present in the existing long-term list shown below.\n"
    "REMOVE an item ONLY when the user has explicitly changed direction and "
    "the prior item is now wrong. Match removal strings to existing items "
    "EXACTLY (copy verbatim). Do not remove items just because they are "
    "not discussed in this window.\n\n"
    "=== OUTPUT FORMAT ===\n"
    "Emit exactly these three sections, each wrapped in its markers. "
    "Omit a section's body if empty, but always include the markers.\n\n"
    "[SUMMARY]\n"
    "<scratch summary prose here>\n"
    "[/SUMMARY]\n\n"
    "[LONG_TERM_ADD]\n"
    "- <new durable item 1>\n"
    "- <new durable item 2>\n"
    "[/LONG_TERM_ADD]\n\n"
    "[LONG_TERM_REMOVE]\n"
    "- <exact existing item text to remove>\n"
    "[/LONG_TERM_REMOVE]"
)


def _parse_compaction_output(text: str) -> tuple[str, list[str], list[str]]:
    """Extract (summary, long_term_add, long_term_remove) from model output.

    Falls back to treating the entire text as the summary if the [SUMMARY]
    markers are absent, so older / misbehaving models still produce a
    usable scratch summary.
    """
    def _extract(tag: str) -> str | None:
        open_tag = f"[{tag}]"
        close_tag = f"[/{tag}]"
        i = text.find(open_tag)
        if i == -1:
            return None
        j = text.find(close_tag, i + len(open_tag))
        if j == -1:
            return None
        return text[i + len(open_tag):j].strip()

    def _bullets(block: str | None) -> list[str]:
        if not block:
            return []
        items: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            if line:
                items.append(line)
        return items

    summary = _extract("SUMMARY")
    adds = _bullets(_extract("LONG_TERM_ADD"))
    removes = _bullets(_extract("LONG_TERM_REMOVE"))
    if summary is None:
        # Fallback: treat full text as summary, but strip any long-term
        # blocks so they don't appear verbatim inside the summary.
        fallback = text
        for tag in ("LONG_TERM_ADD", "LONG_TERM_REMOVE"):
            open_tag, close_tag = f"[{tag}]", f"[/{tag}]"
            i = fallback.find(open_tag)
            if i == -1:
                continue
            j = fallback.find(close_tag, i + len(open_tag))
            if j == -1:
                continue
            fallback = fallback[:i] + fallback[j + len(close_tag):]
        summary = fallback.strip()
    return summary, adds, removes


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
    # Single load - avoids two separate file reads for summary + messages
    data = session.load_session(workspace_path, session_id)
    if data is None:
        return prompt
    summary: str | None = data.get("summary")
    long_term: list[str] = data.get("long_term") or []
    messages: list[dict] = data.get("messages", [])

    if not summary and not messages and not long_term:
        return prompt

    parts: list[str] = []

    if long_term:
        parts.append("[Long-term Memory]")
        for item in long_term:
            parts.append(f"- {item}")
        parts.append("[End of Long-term Memory]\n")

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

    # Truncate if too long - drop oldest messages, keep summary + recent
    if len(full) > MAX_HISTORY_CHARS:
        # Reserve space for the prompt, footer, long-term + summary blocks.
        # Cap long-term at ~40% of budget so scratch summary + some recent
        # messages always have room, even if the item list is huge.
        LONG_TERM_MAX_SHARE = int(MAX_HISTORY_CHARS * 0.4)
        long_term_block = ""
        if long_term:
            header, footer = "[Long-term Memory]\n", "\n[End of Long-term Memory]\n"
            wrap_overhead = len(header) + len(footer)
            # Keep newest items (end of list) that fit within the share
            kept: list[str] = []
            used_chars = 0
            for item in reversed(long_term):
                line = f"- {item}"
                cost = len(line) + 1  # +1 for newline
                if used_chars + cost + wrap_overhead > LONG_TERM_MAX_SHARE:
                    break
                kept.append(line)
                used_chars += cost
            kept.reverse()
            if len(kept) < len(long_term):
                logger.warning(
                    "History truncation: long-term trimmed to %d/%d newest items",
                    len(kept), len(long_term),
                )
            if kept:
                long_term_block = header + "\n".join(kept) + footer
        summary_block = (
            f"[Session Summary]\n{summary}\n[End of Session Summary]\n" if summary else ""
        )
        overhead = len(prompt) + len(long_term_block) + len(summary_block) + 500
        msg_budget = max(0, MAX_HISTORY_CHARS - overhead)
        if msg_budget == 0 and messages:
            logger.warning(
                "History truncation: long-term/summary fill budget; "
                "dropping all %d recent messages", len(messages),
            )

        history_parts: list[str] = []
        if long_term_block:
            history_parts.append(long_term_block)
        if summary_block:
            history_parts.append(summary_block)

        # Add messages newest-to-oldest until the budget is exhausted
        msg_lines: list[str] = []
        used = 0
        if msg_budget > 0:
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
    on_event: Callable[[ChatEvent], Awaitable[None]] | None = None,
    inject_queue: asyncio.Queue[str] | None = None,
) -> CodexResult:
    """Run ``codex exec --json`` with session history prepended.

    on_event  - called for each parsed event as it arrives (streaming).
    inject_queue - when a message is put, the running subprocess is killed
                   and restarted with the injected context appended.

    approval maps to sandbox/approval flags:
      - "full"    -> --dangerously-bypass-approvals-and-sandbox
      - "auto"    -> --full-auto
      - "confirm" -> --sandbox workspace-write
      - "deny"    -> --sandbox read-only
    """
    session_id = session.ensure_session(workspace_path, user_id)

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
    cmd.append("-")

    injected: list[str] = []

    while True:  # restart loop for inject
        effective_prompt = prompt
        if injected:
            effective_prompt += (
                "\n\n[Additional context from user while you were thinking]:\n"
                + "\n".join(injected)
            )

        contextual_prompt = _build_contextual_prompt(
            effective_prompt, workspace_path, session_id,
        )
        logger.info("Running codex exec (prompt %d chars, context %d chars)",
                     len(prompt), len(contextual_prompt))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            result = CodexResult()
            result.text = "Error: codex CLI not found on PATH."
            result.events.append(ChatEvent(kind="text", content=result.text))
            return result

        proc.stdin.write(contextual_prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()

        result = CodexResult()
        restarting = False

        # Watch inject_queue - kill subprocess when a message arrives
        async def _watch_inject() -> None:
            nonlocal restarting
            msg = await inject_queue.get()
            injected.append(msg)
            restarting = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass

        inject_task: asyncio.Task | None = None
        if inject_queue is not None:
            inject_task = asyncio.create_task(_watch_inject())

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

                prev_count = len(result.events)
                _process_event(event, result)

                if on_event:
                    for ev in result.events[prev_count:]:
                        await on_event(ev)

            await proc.wait()
        except asyncio.CancelledError:
            logger.info("Codex run cancelled, killing subprocess %d", proc.pid)
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise
        finally:
            if inject_task and not inject_task.done():
                inject_task.cancel()
                try:
                    await inject_task
                except asyncio.CancelledError:
                    pass

        # If we're restarting due to inject, drain pipes and any extra
        # injects that arrived while we were shutting down.
        if restarting:
            try:
                await proc.stderr.read()
            except Exception:
                pass
            if inject_queue is not None:
                while not inject_queue.empty():
                    try:
                        injected.append(inject_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            logger.info("Restarting codex with %d injected message(s)", len(injected))
            if on_event:
                await on_event(ChatEvent(
                    kind="tool",
                    content="Restarting with injected context...",
                ))
            continue  # restart loop

        break  # normal completion

    # Discard any inject messages that arrived after the final answer.
    if inject_queue is not None:
        while not inject_queue.empty():
            try:
                inject_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
    if stderr:
        logger.debug("codex stderr: %s", stderr)

    if proc.returncode != 0 and not result.events:
        result.text = f"Codex exited with code {proc.returncode}"
        if stderr:
            result.text += f"\n{stderr}"
        result.events.append(ChatEvent(kind="text", content=result.text))

    # Log the original prompt (including injected context) to session.
    async with _get_workspace_lock(workspace_path):
        log_prompt = effective_prompt if injected else prompt
        _log_to_session(workspace_path, session_id, log_prompt, result)

    await _maybe_compact(workspace_path, session_id, summary_model)

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
        logger.error("Failed to log session", exc_info=True)


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
        new_summary, lt_add, lt_remove = await _compact_session(
            workspace_path, session_id, summary_model)
        if not new_summary:
            logger.error("Compaction produced empty summary for session %s", session_id)
            return
        # Reject summaries that are suspiciously short compared to the existing
        # one - a sign of a truncated or failed codex response.
        min_len = max(100, len(existing_summary) // 2)
        if len(new_summary) < min_len:
            logger.error(
                "Compaction summary too short (%d chars, min %d) for session %s - keeping existing",
                len(new_summary), min_len, session_id,
            )
            return
        async with _get_workspace_lock(workspace_path):
            session.set_summary(
                workspace_path, session_id, new_summary,
                keep_recent=KEEP_RECENT_AFTER_COMPACT,
                long_term_add=lt_add,
                long_term_remove=lt_remove,
            )
        logger.info(
            "Session %s compacted, summary %d chars, long_term +%d/-%d",
            session_id, len(new_summary), len(lt_add), len(lt_remove),
        )
    except Exception:
        logger.error("Compaction check failed", exc_info=True)


async def _compact_session(
    workspace_path: str, session_id: str, summary_model: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Run Codex to compact a session.

    Returns (summary, long_term_add, long_term_remove). On failure returns
    ("", [], []). Does NOT write to disk - caller takes the workspace lock
    and calls set_summary with the deltas.
    """
    data = session.load_session(workspace_path, session_id)
    if data is None:
        return ("", [], [])
    messages = data.get("messages", [])
    existing_summary = data.get("summary")
    existing_long_term: list[str] = data.get("long_term") or []

    if not messages:
        return ("", [], [])

    # Build the content to summarize, staying within a token budget.
    # Large prompts cause the summary model to return truncated/empty output.
    MAX_SUMMARY_CHARS = 80_000  # ~20K tokens - safe for most models

    parts: list[str] = []
    if existing_long_term:
        # Cap the existing-list dump at ~25% of the summary budget so the
        # conversation still has room. Dedup on write catches re-adds of
        # any trimmed items.
        lt_max = int(MAX_SUMMARY_CHARS * 0.25)
        lt_lines: list[str] = []
        lt_used = 0
        for item in reversed(existing_long_term):
            line = f"- {item}"
            if lt_used + len(line) + 1 > lt_max:
                break
            lt_lines.append(line)
            lt_used += len(line) + 1
        lt_lines.reverse()
        if lt_lines:
            parts.append("Existing long-term memory (do not re-add these items; remove only if explicitly invalidated):")
            parts.extend(lt_lines)
            if len(lt_lines) < len(existing_long_term):
                parts.append(
                    f"(showing {len(lt_lines)} of {len(existing_long_term)} items; older items omitted)"
                )
            parts.append("")
    if existing_summary:
        parts.append(f"Previous summary:\n{existing_summary}\n")
    parts.append("Conversation to summarize:")

    # Add messages newest-first until we hit the budget, then reverse
    overhead = len(SUMMARY_PROMPT) + sum(len(p) for p in parts) + 200
    budget = max(0, MAX_SUMMARY_CHARS - overhead)
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
        return ("", [], [])

    full_prompt = f"{SUMMARY_PROMPT}\n\n" + "\n".join(parts)

    # --json so we can parse agent_message events reliably; bypass mode
    # because compaction is a trusted internal LLM call with no tool use,
    # and --full-auto's sandbox can interfere with model API access in some
    # environments.
    cmd = [
        "codex", "exec", "--ephemeral", "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", workspace_path,
    ]
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
                            new_summary = text  # keep updating - last one wins

            await proc.wait()
    except TimeoutError:
        logger.error("Compaction timed out after %ds for session %s", COMPACT_TIMEOUT, session_id)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return ("", [], [])

    if not new_summary:
        stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
        logger.warning("Compaction produced no summary (exit %d): %s", proc.returncode, stderr)
        return ("", [], [])

    summary, lt_add, lt_remove = _parse_compaction_output(new_summary)
    return (summary, lt_add, lt_remove)


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
