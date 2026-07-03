import asyncio
import logging
import os

from . import backends_agent
from . import config
from .utils import CONFIG_DIR, COZTER_DIR
from .utils import load_json_object
from .utils import save_json_object

logger = logging.getLogger(__name__)

WORKSPACE_STATE_PATH = os.path.join(CONFIG_DIR, "workspaces.json")
MAX_RECENT = 50  # cap on stored recent-workspaces list


def _load_all() -> dict:
    """Load workspace state.

    Shape: {user_id_str: {current: {bot_id: path}, recent: [path, ...]}}.
    """
    return load_json_object(
        WORKSPACE_STATE_PATH, "workspace state file", logger,
    )


def _save_all(data: dict) -> None:
    save_json_object(WORKSPACE_STATE_PATH, data)


def _get_user(user_id: int | str) -> dict:
    user_state = _load_all().get(str(user_id))
    return user_state if isinstance(user_state, dict) else {
        "current": {},
        "recent": [],
    }


def get_current(
    user_id: int | str, bot_id: int | str = "_default",
) -> str | None:
    current = _get_user(user_id).get("current", {})
    if not isinstance(current, dict):
        return None
    path = current.get(str(bot_id))
    return path if isinstance(path, str) and path else None


def get_recent(user_id: int | str, limit: int = 10) -> list[str]:
    recent = _get_user(user_id).get("recent", [])
    if not isinstance(recent, list):
        return []
    return [p for p in recent if isinstance(p, str) and p][:limit]


def iter_current_workspaces(
    bot_id: int | str,
) -> list[tuple[str, str]]:
    """Return [(user_id, workspace_path)] for users with a current ws on *bot_id*.

    Used by the scheduler to enumerate active (user, workspace) pairs
    independent of whether the platform addresses users directly
    (Telegram) or via channels (Slack).
    """
    pairs: list[tuple[str, str]] = []
    for uid, state in _load_all().items():
        if not isinstance(state, dict):
            continue
        current = state.get("current")
        if not isinstance(current, dict):
            continue
        path = current.get(str(bot_id))
        if isinstance(path, str) and path:
            pairs.append((uid, path))
    return pairs


def select_workspace(
    user_id: int | str, path: str, bot_id: int | str = "_default",
) -> None:
    """Set path as current workspace for a bot and push it to recent."""
    all_state = _load_all()
    uid = str(user_id)
    user_state = all_state.get(uid, {"current": {}, "recent": []})
    if not isinstance(user_state, dict):
        user_state = {"current": {}, "recent": []}
    if not isinstance(user_state.get("current"), dict):
        user_state["current"] = {}
    user_state["current"][str(bot_id)] = path

    recent = user_state.get("recent", [])
    if not isinstance(recent, list):
        recent = []
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    user_state["recent"] = recent[:MAX_RECENT]

    all_state[uid] = user_state
    _save_all(all_state)


def migrate_current_workspace(
    source_user_id: int | str,
    target_user_id: int | str,
    target_bot_id: int | str,
    *,
    source_bot_ids: tuple[int | str, ...] = (),
    source_bot_prefixes: tuple[str, ...] = (),
) -> bool:
    """Copy a current workspace to a new state key if the target is empty."""
    all_state = _load_all()
    source_uid = str(source_user_id)
    target_uid = str(target_user_id)
    target_bot = str(target_bot_id)

    source_state = all_state.get(source_uid)
    if not isinstance(source_state, dict):
        return False
    source_current = source_state.get("current")
    if not isinstance(source_current, dict):
        return False

    target_state = all_state.get(target_uid)
    if not isinstance(target_state, dict):
        target_state = {"current": {}, "recent": []}
    target_current = target_state.get("current")
    if not isinstance(target_current, dict):
        target_current = {}
        target_state["current"] = target_current
    if target_current.get(target_bot):
        return False

    path = source_current.get(target_bot)
    for bot_id in source_bot_ids:
        if path:
            break
        path = source_current.get(str(bot_id))
    if not path and source_bot_prefixes:
        for bot_id, candidate in source_current.items():
            if any(str(bot_id).startswith(p) for p in source_bot_prefixes):
                path = candidate
                break
    if not isinstance(path, str) or not path:
        return False

    target_current[target_bot] = path
    target_state["recent"] = _merge_recent(
        path,
        target_state.get("recent", []),
        source_state.get("recent", []),
    )
    all_state[target_uid] = target_state
    _save_all(all_state)
    return True


def migrate_current_workspace_platform_keys(
    source_bot_prefix: str,
    target_bot_id: int | str,
) -> int:
    """Copy legacy platform current keys to *target_bot_id* for all users."""
    all_state = _load_all()
    target_bot = str(target_bot_id)
    changed = 0
    for uid, state in all_state.items():
        if not isinstance(state, dict):
            continue
        current = state.get("current")
        if not isinstance(current, dict) or current.get(target_bot):
            continue
        path = None
        for bot_id, candidate in current.items():
            if str(bot_id).startswith(source_bot_prefix):
                path = candidate
                break
        if not isinstance(path, str) or not path:
            continue
        current[target_bot] = path
        state["recent"] = _merge_recent(path, state.get("recent", []))
        all_state[uid] = state
        changed += 1
    if changed:
        _save_all(all_state)
    return changed


def _merge_recent(primary: str, *recent_lists: object) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in (primary,):
        if value and value not in seen:
            merged.append(value)
            seen.add(value)
    for recent in recent_lists:
        if not isinstance(recent, list):
            continue
        for value in recent:
            if not isinstance(value, str) or not value or value in seen:
                continue
            merged.append(value)
            seen.add(value)
            if len(merged) >= MAX_RECENT:
                return merged
    return merged[:MAX_RECENT]


def ensure_cozter_dir(path: str) -> None:
    """Create .cozter folder inside the workspace if it doesn't exist."""
    cozter_path = os.path.join(path, COZTER_DIR)
    os.makedirs(cozter_path, exist_ok=True)


# ---------------------------------------------------------------------------
# Workspace settings (stored in .cozter/settings.json)
# ---------------------------------------------------------------------------

AVAILABLE_BACKENDS = backends_agent.AVAILABLE_BACKENDS
DEFAULT_BACKEND = backends_agent.DEFAULT_BACKEND
# Compaction + auto-titling default to codex regardless of which agent
# runs the chat turns - codex is the cheapest summarizer for most users.
# Both the agent and the model under it are independently configurable
# via /summaryagent and /summarymodel.
DEFAULT_SUMMARY_BACKEND = "codex"

AVAILABLE_PERMISSIONS = ["full", "auto", "confirm", "deny"]
DEFAULT_PERMISSION = "auto"
PERMISSION_DESCRIPTIONS = {
    "full": "Full access - bypass all approvals and sandbox",
    "auto": "Run tool calls automatically (sandboxed where supported)",
    "confirm": (
        "Cautious - chat can't prompt per tool call, so llama exposes"
        " read-only tools only and CLI backends fall back to auto."
        " For ask-before-acting, use /style collaborative"
    ),
    "deny": "Block all tool calls (text-only responses)",
}

# Interaction style: how collaborative the agent is on interactive chat
# turns. It selects which policy the shared prompt preamble in
# agent.py carries, so it steers every backend the same way (codex,
# copilot, claude_code, llama). Scheduled/ephemeral turns cannot pause on
# [[await]], so they always run autonomously regardless of this setting.
AVAILABLE_STYLES = ["collaborative", "autonomous"]
DEFAULT_STYLE = "collaborative"
STYLE_DESCRIPTIONS = {
    "collaborative": (
        "Ask before big or ambiguous actions and pause for your reply"
        " (Claude-Code-like)"
    ),
    "autonomous": "Decide and proceed without asking (full-auto)",
}

# Reasoning effort: a single 0-100 percentage. Each agent backend maps
# the percentage to its own native scale (codex has 5 levels including
# "xhigh", llama has 4, claude_code and copilot both have 5).
# This sidesteps the per-backend vocabulary problem - the user picks one
# number and every backend reacts in its own way.
#
# 0 is the "off" value: no effort signal is sent and each backend's
# server-side default is used. 1-100 are explicit overrides.
DEFAULT_REASONING_EFFORT = 0


def _settings_path(workspace_path: str) -> str:
    return os.path.join(workspace_path, COZTER_DIR, "settings.json")


def _load_settings(workspace_path: str) -> dict:
    return load_json_object(
        _settings_path(workspace_path), "workspace settings", logger,
    )


def _save_settings(workspace_path: str, settings: dict) -> None:
    save_json_object(_settings_path(workspace_path), settings)


def _set_setting(workspace_path: str, key: str, value: object) -> None:
    settings = _load_settings(workspace_path)
    settings[key] = value
    _save_settings(workspace_path, settings)


def _coerce_backend_name(name: object, default: str = DEFAULT_BACKEND) -> str:
    if isinstance(name, str) and name in AVAILABLE_BACKENDS:
        return name
    return default


def _coerce_permission(permission: object) -> str:
    return (
        permission
        if isinstance(permission, str) and permission in AVAILABLE_PERMISSIONS
        else DEFAULT_PERMISSION
    )


def get_backend_name(workspace_path: str) -> str:
    return _coerce_backend_name(_load_settings(workspace_path).get("backend"))


def set_backend_name(workspace_path: str, name: str) -> None:
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {AVAILABLE_BACKENDS}"
        )
    _set_setting(workspace_path, "backend", name)


def get_summary_backend_name(workspace_path: str) -> str:
    """Backend that runs compaction / auto-titling for this workspace."""
    return _coerce_backend_name(
        _load_settings(workspace_path).get("summary_backend"),
        DEFAULT_SUMMARY_BACKEND,
    )


def set_summary_backend_name(workspace_path: str, name: str) -> None:
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {AVAILABLE_BACKENDS}"
        )
    _set_setting(workspace_path, "summary_backend", name)


def _with_extra_models(backend_name: str, base) -> list[str]:
    """Append user-configured extra models to a backend's built-in list.

    Built-in lists are a curated snapshot; ``config.extra_models`` lets
    users add newer/private models without editing source. Built-ins come
    first, extras keep their order, and duplicates are dropped.
    """
    models = list(base)
    seen = set(models)
    for model in config.get_extra_models(backend_name):
        if model not in seen:
            seen.add(model)
            models.append(model)
    return models


def get_available_models(workspace_path: str) -> list[str]:
    """List models for the workspace's currently selected backend."""
    backend_name = get_backend_name(workspace_path)
    return _with_extra_models(
        backend_name,
        backends_agent.get_backend(backend_name).available_models,
    )


def get_available_summary_models(workspace_path: str) -> list[str]:
    """List models for the workspace's summary backend (may differ from chat)."""
    backend_name = get_summary_backend_name(workspace_path)
    return _with_extra_models(
        backend_name,
        backends_agent.get_backend(backend_name).available_models,
    )


def _model_keys(backend_name: str) -> tuple[str, str]:
    """Return (model_key, summary_key) for the given backend."""
    return f"{backend_name}_model", f"{backend_name}_summary_model"


def _resolve_model(
    settings: dict, backend_name: str, summary: bool,
) -> str:
    backend_name = _coerce_backend_name(backend_name)
    backend = backends_agent.get_backend(backend_name)
    model_key, summary_key = _model_keys(backend_name)
    key = summary_key if summary else model_key
    default = (
        backend.default_summary_model if summary else backend.default_model
    )
    configured = settings.get(key)
    return configured if isinstance(configured, str) and configured else default


def get_run_config(workspace_path: str) -> tuple[str, str, str, str, str]:
    """Return (chat_backend, model, summary_model, permission, summary_backend).

    The summary backend (and the summary model resolved against it) can
    differ from the chat backend - users may want, say, codex to run
    chat turns while a cheap local llama handles compaction.
    """
    s = _load_settings(workspace_path)
    backend_name = _coerce_backend_name(s.get("backend"))
    summary_backend = _coerce_backend_name(
        s.get("summary_backend"), DEFAULT_SUMMARY_BACKEND,
    )
    return (
        backend_name,
        _resolve_model(s, backend_name, summary=False),
        _resolve_model(s, summary_backend, summary=True),
        _coerce_permission(s.get("permission")),
        summary_backend,
    )


def get_model(workspace_path: str) -> str:
    s = _load_settings(workspace_path)
    return _resolve_model(
        s, _coerce_backend_name(s.get("backend")), summary=False,
    )


def set_model(workspace_path: str, model: str) -> None:
    backend_name = get_backend_name(workspace_path)
    model_key, _ = _model_keys(backend_name)
    _set_setting(workspace_path, model_key, model)


def get_summary_model(workspace_path: str) -> str:
    """Summary model scoped to the summary backend (not the chat backend)."""
    s = _load_settings(workspace_path)
    summary_backend = _coerce_backend_name(
        s.get("summary_backend"), DEFAULT_SUMMARY_BACKEND,
    )
    return _resolve_model(s, summary_backend, summary=True)


def set_summary_model(workspace_path: str, model: str) -> None:
    """Store the summary model under the summary backend's key."""
    summary_backend = get_summary_backend_name(workspace_path)
    _, summary_key = _model_keys(summary_backend)
    _set_setting(workspace_path, summary_key, model)


def get_permission(workspace_path: str) -> str:
    return _coerce_permission(_load_settings(workspace_path).get("permission"))


def set_permission(workspace_path: str, permission: str) -> None:
    if permission not in AVAILABLE_PERMISSIONS:
        raise ValueError(
            f"Unknown permission: {permission}. "
            f"Available: {AVAILABLE_PERMISSIONS}"
        )
    _set_setting(workspace_path, "permission", permission)


def _coerce_style(style: object) -> str:
    return (
        style
        if isinstance(style, str) and style in AVAILABLE_STYLES
        else DEFAULT_STYLE
    )


def get_interaction_style(workspace_path: str) -> str:
    return _coerce_style(_load_settings(workspace_path).get("style"))


def set_interaction_style(workspace_path: str, style: str) -> None:
    if style not in AVAILABLE_STYLES:
        raise ValueError(
            f"Unknown interaction style: {style}. "
            f"Available: {AVAILABLE_STYLES}"
        )
    _set_setting(workspace_path, "style", style)


def get_reasoning_effort(workspace_path: str) -> int:
    """Workspace reasoning effort as 0-100. Falls back to default on bad data."""
    val = _load_settings(workspace_path).get(
        "reasoning_effort", DEFAULT_REASONING_EFFORT,
    )
    if not isinstance(val, int) or isinstance(val, bool):
        return DEFAULT_REASONING_EFFORT
    return max(0, min(val, 100))


def set_reasoning_effort(workspace_path: str, effort: int) -> None:
    """Clamp to 0-100 and persist."""
    clamped = max(0, min(int(effort), 100))
    _set_setting(workspace_path, "reasoning_effort", clamped)


def _positive_int_setting(
    workspace_path: str, key: str, default: int,
) -> int:
    val = _load_settings(workspace_path).get(key, default)
    if not isinstance(val, int) or isinstance(val, bool) or val < 1:
        return default
    return val


# Defaults for the two per-workspace turn-counter knobs. Owned here
# because :mod:`colony` and :mod:`compaction` need workspace settings
# at module import time; defining them up there too creates a cycle
# only late imports could break. Settled here, both consumer modules
# can import workspace.DEFAULT_* directly.
DEFAULT_COLONY_INTERVAL = 3
DEFAULT_COMPACT_INTERVAL = 10


def get_colony_interval(workspace_path: str) -> int:
    """Compactions per workspace-wide colony consolidation pass."""
    return _positive_int_setting(
        workspace_path, "colony_interval", DEFAULT_COLONY_INTERVAL,
    )


def set_colony_interval(workspace_path: str, interval: int) -> None:
    if interval < 1:
        raise ValueError("colony_interval must be >= 1")
    _set_setting(workspace_path, "colony_interval", interval)


def get_compact_interval(workspace_path: str) -> int:
    """Messages between auto-compactions for sessions in this workspace."""
    return _positive_int_setting(
        workspace_path, "compact_interval", DEFAULT_COMPACT_INTERVAL,
    )


def set_compact_interval(workspace_path: str, interval: int) -> None:
    if interval < 1:
        raise ValueError("compact_interval must be >= 1")
    _set_setting(workspace_path, "compact_interval", interval)


# Character budget for the context block (colony + long-term memory +
# session summary + recent messages) that agent.py prepends to each turn's
# prompt. Measured in characters as a provider-agnostic proxy for tokens -
# there is no single tokenizer across the codex/claude/copilot/llama
# backends - so raise it for large-context models and lower it for small
# local ones. agent.py drops the oldest recent messages first to fit.
DEFAULT_HISTORY_BUDGET = 50_000
MIN_HISTORY_BUDGET = 2_000


def get_history_budget(workspace_path: str) -> int:
    """Max characters of prepended context per turn (see agent.py)."""
    return _positive_int_setting(
        workspace_path, "history_budget", DEFAULT_HISTORY_BUDGET,
    )


def set_history_budget(workspace_path: str, budget: int) -> None:
    if budget < MIN_HISTORY_BUDGET:
        raise ValueError(f"history_budget must be >= {MIN_HISTORY_BUDGET}")
    _set_setting(workspace_path, "history_budget", budget)


# ---------------------------------------------------------------------------
# Per-workspace lock — every concurrent reader/writer of a workspace's files
# (sessions, colony, schedules, settings) takes this lock to serialize.
# ---------------------------------------------------------------------------

_locks: dict[str, asyncio.Lock] = {}


def get_lock(workspace_path: str) -> asyncio.Lock:
    """Return the per-workspace asyncio lock; creates it on first use."""
    lock = _locks.get(workspace_path)
    if lock is None:
        lock = asyncio.Lock()
        _locks[workspace_path] = lock
    return lock
