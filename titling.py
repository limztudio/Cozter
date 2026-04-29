"""Session auto-titling.

After each user turn, ``maybe_auto_title`` checks whether the session
still has the auto-generated default name and — if so — generates a
short topical title via a lightweight backend call. Compaction also
emits a ``[TITLE]`` block as part of the summary prompt; both paths
share ``clean_title`` for trimming the model's output.
"""

import logging

from . import backends_agent, session
from . import workspace as workspace_mod
from .utils import drain_llm_subprocess

logger = logging.getLogger(__name__)


TITLE_PROMPT = (
    "You are titling a chat session for a list view. Read the recent "
    "messages and produce a short descriptive name: 3-7 words, "
    "Title Case, no trailing punctuation, no quotes, no commentary. "
    "Pick the dominant topic, not the most recent line. Output only "
    "the title."
)
TITLE_TIMEOUT = 30  # seconds
TITLE_MAX_CHARS = 60
TITLE_CONTEXT_CHARS = 8_000

# Per-session guard so two concurrent run() calls (in theory) on the
# same session don't both spawn a title pass. The bot's per-user lock
# already serializes turns, but we don't rely on that here.
_in_flight: set[tuple[str, str]] = set()


def clean_title(raw: str) -> str | None:
    """Trim a model-emitted title to a single short line."""
    stripped = raw.strip()
    if not stripped:
        return None
    line = stripped.splitlines()[0].strip(" \t.\"'`*_")
    if not line:
        return None
    if len(line) > TITLE_MAX_CHARS:
        line = line[:TITLE_MAX_CHARS].rstrip()
    return line or None


async def maybe_auto_title(
    workspace_path: str, session_id: str, summary_model: str | None,
    *, backend_name: str | None,
) -> None:
    """Generate a title once a session has its first assistant reply.

    Skipped when the session already has a custom name (anything other
    than the auto-generated ``Session YYYY-MM-DD``) so we don't
    overwrite a meaningful title with a freshly-generated one. The
    compaction path refreshes the title separately.
    """
    key = (workspace_path, session_id)
    if key in _in_flight:
        return
    try:
        data = session.load_session(workspace_path, session_id)
        if data is None:
            return
        if not session.is_default_name(data.get("name")):
            return
        # Need at least one assistant reply before titling makes sense.
        msgs = data.get("messages", [])
        if not any(m.get("role") == "assistant" for m in msgs):
            return
        _in_flight.add(key)
        title = await generate(
            workspace_path, session_id, summary_model,
            backend_name=backend_name, _preloaded_data=data,
        )
        if not title:
            return
        async with workspace_mod.get_lock(workspace_path):
            session.set_session_name(workspace_path, session_id, title)
        logger.info("Session %s auto-titled: %s", session_id, title)
    except Exception:
        logger.warning("Auto-title failed", exc_info=True)
    finally:
        _in_flight.discard(key)


async def generate(
    workspace_path: str,
    session_id: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
    _preloaded_data: dict | None = None,
) -> str | None:
    """Run a small backend call to title a session. Returns None on failure."""
    backend = backends_agent.get_backend(backend_name)
    data = _preloaded_data or session.load_session(workspace_path, session_id)
    if data is None:
        return None

    parts: list[str] = []
    summary = data.get("summary")
    if summary:
        parts.append(f"Previous summary:\n{summary}\n")
    parts.append("Recent messages:")

    # Newest-first under a tight char budget; the title only needs the
    # gist, so we don't pull the whole history.
    msg_lines = session.take_recent_messages(
        data.get("messages", []), TITLE_CONTEXT_CHARS,
    )
    if not msg_lines:
        return None
    parts.extend(msg_lines)

    full_prompt = f"{TITLE_PROMPT}\n\n" + "\n".join(parts)

    try:
        proc = await backend.launch(
            workspace_path, full_prompt, summary_model, approval="full",
            compaction=True,
        )
    except FileNotFoundError:
        logger.warning(
            "%s CLI not found on PATH - cannot title session",
            backend.executable,
        )
        return None

    raw = await drain_llm_subprocess(
        proc, backend, TITLE_TIMEOUT, f"Title (session {session_id})",
    )
    return clean_title(raw) if raw else None
