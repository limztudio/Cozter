"""Session router — picks the best-matching existing session for a new
user message, or creates a new one when no session is a good fit.
"""

import logging

from . import backends_agent, session
from .utils import run_internal_backend

logger = logging.getLogger(__name__)


ROUTER_PROMPT = (
    "You are a session router for a multi-session chat assistant.\n"
    "The user is about to send a new message. Pick the existing "
    "session whose ongoing topic best fits the message — or output "
    "NEW if the user has switched to a topic none of the sessions "
    "match.\n\n"
    "Rules:\n"
    "- Prefer to continue an existing session when there is a clear "
    "topical match.\n"
    "- Choose NEW for genuinely new topics, not minor digressions.\n"
    "- Output exactly one line: either the bare session id "
    "(no quotes, no commentary), or the literal word NEW.\n"
    "- Do NOT call any tools or read any files; decide from the "
    "input below.\n"
)
ROUTER_TIMEOUT = 60  # seconds; on timeout the router falls back to NEW
ROUTER_MAX_SESSIONS = 20  # cap input size; sessions are listed newest-first
ROUTER_PER_SESSION_CHARS = 600
ROUTER_PROMPT_PREVIEW_CHARS = 1_000


def _truncate_router_text(text: str, limit: int) -> str:
    """Keep one router field/block within *limit* characters."""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"[:limit]
    return text[:limit - 1] + "…"


def _build_session_block(data: dict) -> str:
    """Return one bounded session description for the routing prompt.

    Session summaries and long-term memory are model-produced persisted text,
    so neither may be trusted to remain small.  Keeping the whole block under
    the advertised per-session cap protects the router even when old state is
    malformed or unusually verbose.  The id remains first, allowing a useful
    choice whenever it fits in normal session-id bounds.
    """
    sid = data["id"]
    raw_name = data.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else sid[:8]
    block = [
        f"id: {sid}",
        f"name: {_truncate_router_text(name, ROUTER_PER_SESSION_CHARS)}",
    ]
    summary = data.get("summary")
    if isinstance(summary, str) and summary:
        block.append(
            "summary: " + _truncate_router_text(
                summary, ROUTER_PER_SESSION_CHARS,
            ),
        )
    long_term = data.get("long_term")
    if isinstance(long_term, list):
        items = [
            _truncate_router_text(item, ROUTER_PER_SESSION_CHARS)
            for item in long_term[:5]
            if isinstance(item, str) and item
        ]
        if items:
            block.append("long-term: " + "; ".join(items))
    return _truncate_router_text("\n".join(block), ROUTER_PER_SESSION_CHARS)


def _build_router_prompt(prompt: str, sessions_data: list[dict]) -> str:
    """Assemble the router prompt body. Caller prepends ROUTER_PROMPT."""
    parts: list[str] = ["User message:"]
    preview = prompt.strip()
    if len(preview) > ROUTER_PROMPT_PREVIEW_CHARS:
        preview = preview[:ROUTER_PROMPT_PREVIEW_CHARS] + "…"
    parts.append(preview)
    parts.append("")
    parts.append(f"Existing sessions ({len(sessions_data)}, newest first):")
    parts.append("")
    for s in sessions_data:
        parts.append(_build_session_block(s))
        parts.append("")
    parts.append(
        "Output exactly one line: a session id from the list above, or NEW."
    )
    return "\n".join(parts)


def _parse_router_output(raw: str, valid_ids: set[str]) -> str | None:
    """Return a session id, "NEW", or None if the output is unparseable.

    The model is instructed to output exactly one line, but in practice
    it sometimes adds a trailing period, code-fences, or explanatory
    text. We scan lines and take the first that's either "NEW" or a
    known session id.
    """
    for line in raw.splitlines():
        token = line.strip().strip("`'\"., ")
        if not token:
            continue
        if token.upper() == "NEW":
            return "NEW"
        # Session state deliberately permits safe IDs with punctuation such
        # as ``-`` and ``_``.  Membership is the relevant safety check here;
        # a narrower grammar silently made those valid restored sessions
        # unrouteable.
        if token in valid_ids:
            return token
    return None


async def select_or_create_session(
    prompt: str,
    workspace_path: str,
    summary_model: str | None = None,
    *,
    backend_name: str | None = None,
) -> tuple[str, dict]:
    """Pick the session whose topic best matches *prompt*, else create one.

    Shortcuts an LLM call for the trivial cases (zero sessions →
    create new). Returns (session_id, loaded session data).
    """
    backend = backends_agent.get_backend(backend_name)

    sessions_data = session.list_sessions_with_data(workspace_path)
    sessions_data = sessions_data[:ROUTER_MAX_SESSIONS]

    if not sessions_data:
        data = session.create_session(workspace_path)
        logger.info(
            "Router: no existing sessions, created %s", data["id"],
        )
        return (data["id"], data)

    body = _build_router_prompt(prompt, sessions_data)
    full_prompt = f"{ROUTER_PROMPT}\n\n{body}"

    raw = await run_internal_backend(
        backend,
        workspace_path,
        full_prompt,
        summary_model,
        timeout=ROUTER_TIMEOUT,
        label="Router",
        log=logger,
        missing_executable_message="%s CLI not found - router falling back to NEW",
        missing_level=logging.WARNING,
    )
    if raw is None:
        data = session.create_session(workspace_path)
        return (data["id"], data)

    valid_ids = {s["id"] for s in sessions_data}
    decision = _parse_router_output(raw, valid_ids) if raw else None
    if decision and decision != "NEW":
        loaded = session.load_session(workspace_path, decision)
        if loaded is not None:
            logger.info("Router: continuing session %s", decision)
            return (decision, loaded)

    if decision is None:
        logger.warning("Router output unparseable; defaulting to NEW: %r", raw)
    fresh = session.create_session(workspace_path)
    logger.info("Router: created new session %s", fresh["id"])
    return (fresh["id"], fresh)
