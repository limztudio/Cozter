"""Per-session compaction.

Rolls a session's message history into a SCRATCH summary plus a
LONG-TERM list, replacing the raw messages so the conversation can
continue without exhausting the context window. When every model that will
receive a conversation has a known input capacity, compaction is triggered
from a conservative estimate of the stored context; otherwise it falls back
to ``len(messages) >= compact_interval`` after a turn. The interval is set
via ``/compact <number>``; there is no one-shot manual compaction.

A successful compaction also bumps the workspace-wide colony counter
and may fire a colony consolidation pass via ``colony.maybe_trigger``.
"""

import logging

from . import backends_agent, colony, config, session, titling
from . import workspace as workspace_mod
from .utils import (
    extract_marker_block, parse_bullets, run_internal_backend,
    strip_marker_block, take_recent_lines,
)

logger = logging.getLogger(__name__)

# A reply can finish while a previous post-turn compaction is still running.
# Do not let both snapshots race to summarize and rewrite the same session.
_in_flight: set[tuple[str, str]] = set()


KEEP_RECENT_AFTER_COMPACT = 5
MAX_SUMMARY_CHARS = 80_000  # ~20K tokens - safe for most models
COMPACT_TIMEOUT = 240  # seconds; large sessions with rich long-term lists need headroom

# Model windows include fixed system instructions, tools, a future user
# message, and the model's reply. Trigger before stored conversation material
# consumes more than this portion so the estimate does not crowd those out.
_MODEL_CONTEXT_COMPACT_FRACTION = 0.60
# ``/context`` remains a per-workspace character ceiling for saved context.
# Summarize before its truncation logic has to discard much raw history.
_HISTORY_BUDGET_COMPACT_FRACTION = 0.75

# A previously persisted summary is model output, so it is not guaranteed to
# respect the requested 80-200 word target.  Reserve most of the compaction
# prompt for the raw messages that actually need compacting; otherwise one
# oversized old summary can leave no room for even a single message and block
# every later compaction attempt.
_PREVIOUS_SUMMARY_FRACTION = 4
_PREVIOUS_SUMMARY_TRUNCATION_MARKER = "\n… [previous summary truncated]\n"

SUMMARY_PROMPT = (
    "You are compacting a conversation into two memory layers: a SCRATCH "
    "summary (rewritten each compaction) and LONG-TERM memory (persistent "
    "facts that survive future compactions). You will also produce a short "
    "TITLE that names the session.\n\n"
    "IMPORTANT: This is a pure text-summarization task. Do NOT call any "
    "tools, shell commands, file reads, or web fetches. The conversation "
    "below is everything you need — work from it directly and emit the "
    "output as plain text.\n\n"
    "=== SCRATCH SUMMARY ===\n"
    "Produce a concise abstract of the conversation below. This abstract "
    "REPLACES the raw history, so it must be thorough enough to continue "
    "work seamlessly.\n"
    "The reader will always see the [Long-term Memory] list alongside this "
    "summary, so do NOT repeat facts already captured there. Focus only on "
    "ephemeral context: recent work, current state, open commitments, "
    "in-progress decisions, and errors encountered.\n"
    "Capture: what was just done and why, file paths touched and the nature "
    "of each change, concrete tool results and errors, current "
    "done/in-progress/blocked state, and open commitments not yet in "
    "long-term memory.\n"
    "Drop: greetings, filler, repeated restatements, exploratory turns the "
    "user redirected, superseded tool output, and anything already in the "
    "long-term list.\n"
    "Prose paragraphs grouped by topic. Aim for 80-200 words. "
    "Prefer specificity (names, paths, values) over generic phrasing.\n\n"
    "=== LONG-TERM MEMORY ===\n"
    "Long-term items are durable facts: explicit preferences, architectural "
    "decisions, invariants, stable project facts, hard constraints. "
    "Each item is ONE self-contained sentence.\n"
    "Output the COMPLETE new list in [LONG_TERM] markers - this REPLACES "
    "the existing list entirely. Rules:\n"
    "- Carry forward items that are still true\n"
    "- Merge similar items into one sentence\n"
    "- Remove items that are now wrong or redundant with the summary\n"
    "- Add new durable facts from this conversation\n"
    "Keep the list SHORT: aim for 5-15 items, hard max 30. "
    "Fewer precise items beat many vague ones.\n\n"
    "=== TITLE ===\n"
    "A short descriptive name for this session: 3-7 words, Title Case, "
    "no trailing punctuation, no quotes. Pick the dominant topic of the "
    "conversation as a whole. Example: 'Refactor Schedule Storage'.\n\n"
    "=== OUTPUT FORMAT ===\n"
    "Emit all three sections. Omit a body if empty, but always include "
    "the markers.\n\n"
    "[SUMMARY]\n"
    "<scratch summary prose here>\n"
    "[/SUMMARY]\n\n"
    "[LONG_TERM]\n"
    "- <item 1>\n"
    "- <item 2>\n"
    "[/LONG_TERM]\n\n"
    "[TITLE]\n"
    "<short title>\n"
    "[/TITLE]"
)


def _parse_output(
    text: str,
) -> tuple[str, list[str] | None, str | None]:
    """Extract (summary, long_term, title) from the compaction output.

    long_term is None if [LONG_TERM] markers are absent (no rewrite).
    title is None if [TITLE] markers are absent or empty.
    Falls back to treating the entire text as the summary if [SUMMARY]
    markers are absent, so misbehaving models still produce a usable result.
    """
    summary = extract_marker_block(text, "SUMMARY")
    lt_block = extract_marker_block(text, "LONG_TERM")
    long_term: list[str] | None = (
        parse_bullets(lt_block) if lt_block is not None else None
    )
    title_block = extract_marker_block(text, "TITLE")
    title: str | None = (
        titling.clean_title(title_block) if title_block else None
    )
    if summary is None:
        # Fallback: treat full text as summary, stripping the other blocks.
        fallback = text
        for tag in ("LONG_TERM", "TITLE"):
            fallback = strip_marker_block(fallback, tag)
        summary = fallback.strip()
    return summary, long_term, title


def _previous_summary_budget() -> int:
    """Return the portion of the compaction input reserved for old summary."""
    return max(1, MAX_SUMMARY_CHARS // _PREVIOUS_SUMMARY_FRACTION)


def _bounded_previous_summary(summary: str) -> str:
    """Clip pathological stored summaries while preserving both ends.

    Keeping the beginning retains the original task framing; keeping the end
    retains the most recently recorded state and open work.
    """
    budget = _previous_summary_budget()
    if len(summary) <= budget:
        return summary
    marker = _PREVIOUS_SUMMARY_TRUNCATION_MARKER
    if budget <= len(marker):
        return summary[:budget]
    keep = (budget - len(marker)) // 2
    return summary[:keep] + marker + summary[-(budget - len(marker) - keep):]


def _compaction_prompt_parts(
    existing_summary: str | None,
    existing_long_term: list[str],
) -> list[str]:
    """Build the fixed prefix of a compaction prompt before raw messages."""
    parts: list[str] = []
    if existing_long_term:
        # Show up to 15% of the budget for the existing list so the model
        # knows what to rewrite. With a target of <=30 items this is plenty.
        lt_max = int(MAX_SUMMARY_CHARS * 0.15)
        lt_lines = take_recent_lines(
            existing_long_term, lt_max, lambda x: f"- {x}",
        )
        if lt_lines:
            parts.append(
                "Existing long-term items (rewrite this list per the "
                "instructions above):"
            )
            parts.extend(lt_lines)
            parts.append("")
    if existing_summary:
        parts.append(
            "Previous summary:\n"
            f"{_bounded_previous_summary(existing_summary)}\n"
        )
    parts.append("Conversation to summarize:")
    return parts


def _compaction_message_budget(parts: list[str]) -> int:
    """Return the remaining raw-message budget for a prompt prefix."""
    # Reserve a little extra for join separators and the backend's response
    # envelope. This matches the existing conservative prompt accounting.
    overhead = len(SUMMARY_PROMPT) + sum(len(part) for part in parts) + 200
    return max(0, MAX_SUMMARY_CHARS - overhead)


def _oversized_first_message_prefix(
    data: dict,
) -> tuple[str, int] | None:
    """Return the raw prefix length for a safely-progressible first message.

    ``_take_oldest_message_lines`` represents an oversized first line with a
    bounded prefix plus an ellipsis.  Its unseen suffix must remain durable,
    so this returns the exact number of original content characters exposed
    before that marker.  The caller can replace the first message with the
    suffix only after a successful summary is safely stored.
    """
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    if not isinstance(content, str):
        return None
    summary = data.get("summary")
    existing_summary = summary if isinstance(summary, str) else None
    long_term = data.get("long_term")
    existing_long_term = long_term if isinstance(long_term, list) else []
    parts = _compaction_prompt_parts(
        existing_summary, existing_long_term,
    )
    budget = _compaction_message_budget(parts)
    line = session.format_msg_line(first, cap=None)
    if len(line) <= budget:
        return None
    prefix = session.format_msg_line(
        {"role": first.get("role"), "content": ""}, cap=None,
    )
    # The partial line consumes one character for the ellipsis inserted by
    # _take_oldest_message_lines, so leave that slot out of the raw prefix.
    content_chars = budget - len(prefix) - 1
    if content_chars <= 0:
        return None
    return content, content_chars


def _session_context_text(data: dict, colony_items: list[str]) -> str:
    """Render the persisted context blocks that precede a normal turn.

    The current user message and backend preamble are not included; the
    model-window trigger reserves substantial headroom for both.
    """
    return "\n".join(session.format_context_blocks(data, colony_items))


def _estimate_context_tokens(text: str) -> int:
    """Return a provider-neutral upper-bound estimate for arbitrary text.

    Cozter supports several unrelated tokenizers. Every tokenizer ultimately
    consumes UTF-8 bytes, so counting bytes is deliberately conservative: a
    byte cannot span more than one token, while ordinary text usually packs
    several bytes per token. This biases compaction early rather than risking
    an input-window overflow for non-ASCII text, code, or unknown models.
    """
    return len(text.encode("utf-8"))


def _context_window_tokens(
    context_targets: tuple[tuple[str, str | None], ...] | None,
) -> int | None:
    """Return the smallest known capacity among conversation recipients.

    Flexible turns can send the same saved context to its planner, merger,
    and any tier worker. One unknown recipient means there is no safe numeric
    threshold, so the caller must keep using the configured message interval.
    """
    if not context_targets:
        return None
    limits: list[int] = []
    for backend_name, model in context_targets:
        try:
            backend = backends_agent.get_backend(backend_name)
        except ValueError:
            return None
        selected_model = model
        if selected_model is None:
            default_model = getattr(backend, "default_model", None)
            if isinstance(default_model, str) and default_model:
                selected_model = default_model
        configured = config.get_model_context_window(backend_name, selected_model)
        if configured is not None:
            limits.append(configured)
            continue
        getter = getattr(backend, "context_window_tokens", None)
        window = getter(selected_model) if callable(getter) else None
        if (
            not isinstance(window, int)
            or isinstance(window, bool)
            or window < 1
        ):
            return None
        limits.append(window)
    return min(limits) if limits else None


def _token_trigger(
    data: dict,
    workspace_path: str,
    context_targets: tuple[tuple[str, str | None], ...] | None,
) -> tuple[str, int, int, int] | None:
    """Return token/character trigger details, or ``None`` if not ready.

    The result is ``(reason, measured, threshold, model_window)``. A known
    model window produces a token trigger; the workspace's explicit history
    cap remains a second guard against silently dropping raw messages before
    they have been summarized.
    """
    model_window = _context_window_tokens(context_targets)
    if model_window is None:
        return None

    context_text = _session_context_text(data, colony.get_items(workspace_path))
    estimated_tokens = _estimate_context_tokens(context_text)
    token_threshold = max(
        1, int(model_window * _MODEL_CONTEXT_COMPACT_FRACTION),
    )
    if estimated_tokens >= token_threshold:
        return "tokens", estimated_tokens, token_threshold, model_window

    history_budget = workspace_mod.get_history_budget(workspace_path)
    history_threshold = max(
        1, int(history_budget * _HISTORY_BUDGET_COMPACT_FRACTION),
    )
    if len(context_text) >= history_threshold:
        return "history_chars", len(context_text), history_threshold, model_window
    return None


async def maybe_compact(
    workspace_path: str, session_id: str, summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    context_targets: tuple[tuple[str, str | None], ...] | None = None,
) -> None:
    """Compact a session when its token or fallback trigger is reached.

    ``context_targets`` identifies every backend/model that can receive the
    conversation. When their capacities are all known, token use drives the
    trigger; an unknown target retains the configured message interval.
    Compaction runs outside the workspace lock so other requests aren't
    stalled.
    """
    key = (
        workspace_mod.canonicalize_workspace_path(workspace_path), session_id,
    )
    if key in _in_flight:
        logger.debug("Compaction already in progress for session %s", session_id)
        return

    _in_flight.add(key)
    try:
        async with workspace_mod.get_memory_maintenance_lock(workspace_path):
            await _maybe_compact_under_maintenance_lock(
                workspace_path, session_id, summary_model,
                backend_name=backend_name,
                context_targets=context_targets,
            )
    except Exception:
        logger.error("Compaction check failed", exc_info=True)
    finally:
        _in_flight.discard(key)


async def _maybe_compact_under_maintenance_lock(
    workspace_path: str,
    session_id: str,
    summary_model: str | None,
    *,
    backend_name: str | None,
    context_targets: tuple[tuple[str, str | None], ...] | None,
) -> None:
    """Run one compaction while no colony rewrite can race its snapshot."""
    data = session.load_session(workspace_path, session_id)
    if data is None:
        return
    msgs = data.get("messages", [])
    trigger = _token_trigger(data, workspace_path, context_targets)
    if trigger is not None:
        reason, measured, threshold, model_window = trigger
        if reason == "tokens":
            logger.info(
                "Auto-compact triggered (context≈%d tokens, threshold=%d, "
                "model_window=%d)",
                measured, threshold, model_window,
            )
        else:
            logger.info(
                "Auto-compact triggered (context=%d chars, history "
                "threshold=%d, model_window=%d)",
                measured, threshold, model_window,
            )
    else:
        interval = workspace_mod.get_compact_interval(workspace_path)
        if len(msgs) < interval:
            return
        logger.info(
            "Auto-compact triggered (msgs=%d, interval=%d; no known "
            "context window)",
            len(msgs), interval,
        )
    oversized_first = _oversized_first_message_prefix(data)
    existing_summary = data.get("summary") or ""
    new_summary, new_long_term, new_title, covered_count = await compact_session(
        workspace_path, session_id, summary_model,
        backend_name=backend_name,
        _preloaded_data=data,
    )
    if not new_summary:
        logger.error(
            "Compaction did not cover enough messages for session %s "
            "(covered=%d, keep_recent=%d)",
            session_id, covered_count, KEEP_RECENT_AFTER_COMPACT,
        )
        return
    if oversized_first is not None:
        # The prompt includes only a persisted prefix of the original first
        # message. Save its summary, then retain the exact unseen suffix in
        # that same message slot. Any other count would make a later raw
        # message look covered even though it was not sent to the backend.
        if covered_count != 1:
            logger.error(
                "Oversized-message compaction covered an unsafe number of "
                "messages for session %s (covered=%d)",
                session_id, covered_count,
            )
            return
    elif covered_count <= KEEP_RECENT_AFTER_COMPACT:
        logger.error(
            "Compaction did not cover enough messages for session %s "
            "(covered=%d, keep_recent=%d)",
            session_id, covered_count, KEEP_RECENT_AFTER_COMPACT,
        )
        return
    # Reject summaries that are suspiciously short compared to the existing
    # one - a sign of a truncated or failed backend response.
    # An oversized stored summary is explicitly truncated before the next
    # model call. It may therefore be replaced by a normally sized summary;
    # comparing the result with the pathological original would reject every
    # recovery attempt even after the prompt itself has room again.
    min_len = (
        100
        if len(existing_summary) > _previous_summary_budget()
        else max(100, len(existing_summary) // 2)
    )
    if len(new_summary) < min_len:
        logger.error(
            "Compaction summary too short (%d chars, min %d) "
            "for session %s - keeping existing",
            len(new_summary), min_len, session_id,
        )
        return
    async with workspace_mod.get_lock(workspace_path):
        # The summarizer runs outside this lock. A user can rename the
        # session while it is in flight, so do not let a title derived from
        # the older snapshot overwrite that newer choice. The summary itself
        # is still safe to apply: set_summary trims only the prefix captured
        # in that snapshot and keeps any appended messages.
        latest = session.load_session(workspace_path, session_id)
        if latest is None:
            return
        title_to_save = new_title
        if latest.get("name") != data.get("name"):
            title_to_save = None
            logger.debug(
                "Skipping stale compaction title for session %s", session_id,
            )
        if oversized_first is None:
            session.set_summary(
                workspace_path, session_id, new_summary,
                keep_recent=KEEP_RECENT_AFTER_COMPACT,
                long_term_rewrite=new_long_term,
                title=title_to_save,
                # Only trim the contiguous prefix actually sent to the summary
                # backend. A foreground turn can append more while the summary
                # runs, and oversized histories intentionally leave their later
                # messages raw for the next pass.
                summarized_count=covered_count,
            )
        else:
            original_content, consumed_chars = oversized_first
            messages = latest.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(messages[0], dict)
                or messages[0].get("content") != original_content
            ):
                # A non-append writer changed the oldest message while the
                # summary was in flight. Keeping the raw message is safer
                # than applying a suffix to unrelated session state.
                logger.warning(
                    "Skipping stale oversized-message compaction for "
                    "session %s", session_id,
                )
                return
            # Store the summary and unseen tail in one durable write. No raw
            # entry is trimmed: the same oldest message becomes its suffix.
            session.set_summary(
                workspace_path, session_id, new_summary,
                keep_recent=KEEP_RECENT_AFTER_COMPACT,
                long_term_rewrite=new_long_term,
                title=title_to_save,
                summarized_count=0,
                first_message_tail=original_content[consumed_chars:],
            )
        colony_count = colony.bump_compact_count(workspace_path)
    lt_count = len(new_long_term) if new_long_term is not None else "?"
    logger.info(
        "Session %s compacted, summary %d chars, long_term %s items",
        session_id, len(new_summary), lt_count,
    )
    colony.maybe_trigger(
        workspace_path, colony_count, summary_model,
        backend_name=backend_name,
    )


def _take_oldest_message_lines(messages: list[dict], budget: int) -> list[str]:
    """Return the largest budget-fitting contiguous prefix of *messages*.

    ``session.set_summary`` can only safely discard a prefix of the raw
    history. Selecting a newest suffix made it possible to mark older,
    omitted messages as summarized and then delete them. Keep the prefix
    contiguous and let a later pass compact the remaining suffix once the
    raw overlap has advanced. If the very first message alone exceeds the
    budget, return a marked prefix of it so compaction can still make forward
    progress instead of retrying the same uncompactable history forever.
    """
    lines: list[str] = []
    used = 0
    for message in messages:
        line = session.format_msg_line(message, cap=None)
        if used + len(line) > budget:
            if not lines and budget > 0:
                # An exceptionally large first message must not block all
                # future compactions: only a contiguous oldest prefix may be
                # removed, so skipping it means every later pass chooses the
                # same zero-message prefix.  Give the summary model a marked
                # prefix and let successful compaction advance the history.
                if budget == 1:
                    lines.append("…")
                else:
                    lines.append(line[:budget - 1] + "…")
            break
        lines.append(line)
        # Match utils.take_recent_lines' conservative accounting for the
        # newline that joins adjacent lines.
        used += len(line) + 1
    return lines


async def compact_session(
    workspace_path: str,
    session_id: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    _preloaded_data: dict | None = None,
) -> tuple[str, list[str] | None, str | None, int]:
    """Run the selected backend to compact a session.

    Returns ``(summary, long_term, title, covered_count)``. ``covered_count``
    is the number of consecutive oldest raw messages included in the prompt;
    callers must only trim against that count. ``long_term`` is the new
    complete list, or None if the model did not emit a [LONG_TERM] block
    (existing list kept). ``title`` is None if the model did not emit a
    [TITLE] block. On failure returns ``("", None, None, 0)``. Does NOT write
    to disk - caller takes the workspace lock and calls ``set_summary``.

    _preloaded_data: pass already-loaded session dict to skip a disk read
    (used by maybe_compact which loads the data to check the interval).
    """
    backend = backends_agent.get_backend(backend_name)
    data = _preloaded_data or session.load_session(workspace_path, session_id)
    if data is None:
        return ("", None, None, 0)
    messages = data.get("messages", [])
    existing_summary = data.get("summary")
    existing_long_term: list[str] = data.get("long_term") or []

    if not messages:
        return ("", None, None, 0)

    # Build the content to summarize, staying within a token budget.
    # Large prompts cause the summary model to return truncated/empty output.
    parts = _compaction_prompt_parts(existing_summary, existing_long_term)

    # Add a contiguous oldest prefix until we hit the budget. cap=None so
    # the model sees full message content (compaction's budget is generous
    # enough to afford it). The caller only removes this exact prefix, which
    # prevents unsent messages from being mistaken for summarized history.
    budget = _compaction_message_budget(parts)
    msg_lines = _take_oldest_message_lines(messages, budget)
    parts.extend(msg_lines)

    if not msg_lines:
        logger.warning(
            "Session %s messages too large even for a single entry",
            session_id,
        )
        return ("", None, None, 0)

    full_prompt = f"{SUMMARY_PROMPT}\n\n" + "\n".join(parts)

    logger.info(
        "Running %s compaction for session %s", backend.name, session_id,
    )

    new_summary = await run_internal_backend(
        backend,
        workspace_path,
        full_prompt,
        summary_model,
        timeout=COMPACT_TIMEOUT,
        label=f"Compaction (session {session_id})",
        log=logger,
        missing_executable_message=(
            "%s CLI not found on PATH - cannot compact session"
        ),
    )
    if not new_summary:
        return ("", None, None, 0)

    summary, long_term, title = _parse_output(new_summary)
    return (summary, long_term, title, len(msg_lines))
