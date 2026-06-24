import asyncio
import json
import logging
import os

from . import backends_agent
from .utils import CONFIG_DIR, COZTER_DIR
from .utils import atomic_write as _atomic_write

logger = logging.getLogger(__name__)

WORKSPACE_STATE_PATH = os.path.join(CONFIG_DIR, "workspaces.json")
MAX_RECENT = 50  # cap on stored recent-workspaces list


def _load_json(path: str, label: str) -> dict:
    """Load a JSON file, returning {} on missing/corrupt."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt or unreadable %s (%s): %s", label, path, e)
    return {}


def _load_all() -> dict:
    """Load workspace state.

    Shape: {user_id_str: {current: {bot_id: path}, recent: [path, ...]}}.
    """
    return _load_json(WORKSPACE_STATE_PATH, "workspace state file")


def _save_all(data: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    _atomic_write(WORKSPACE_STATE_PATH, data, CONFIG_DIR)


def _get_user(user_id: int) -> dict:
    return _load_all().get(str(user_id), {"current": {}, "recent": []})


def get_current(user_id: int, bot_id: int | str = "_default") -> str | None:
    return _get_user(user_id).get("current", {}).get(str(bot_id))


def get_recent(user_id: int, limit: int = 10) -> list[str]:
    return _get_user(user_id).get("recent", [])[:limit]


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
        path = state.get("current", {}).get(str(bot_id))
        if path:
            pairs.append((uid, path))
    return pairs


def select_workspace(
    user_id: int, path: str, bot_id: int | str = "_default",
) -> None:
    """Set path as current workspace for a bot and push it to recent."""
    all_state = _load_all()
    uid = str(user_id)
    user_state = all_state.get(uid, {"current": {}, "recent": []})
    user_state["current"][str(bot_id)] = path

    recent = user_state.get("recent", [])
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
    target_current = target_state.setdefault("current", {})
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
    "auto": "Execute all tool calls automatically (sandboxed)",
    "confirm": "Ask before each tool call",
    "deny": "Block all tool calls (text-only responses)",
}

# Reasoning effort: a single 0-100 percentage. Each agent backend maps
# the percentage to its own native scale (codex has 5 levels including
# "xhigh", llama has 4, claude_code is binary, copilot ignores entirely).
# This sidesteps the per-backend vocabulary problem - the user picks one
# number and every backend reacts in its own way.
#
# 0 is the "off" value: no effort signal is sent and each backend's
# server-side default is used. 1-100 are explicit overrides.
DEFAULT_REASONING_EFFORT = 0


def _settings_path(workspace_path: str) -> str:
    return os.path.join(workspace_path, COZTER_DIR, "settings.json")


def _load_settings(workspace_path: str) -> dict:
    return _load_json(_settings_path(workspace_path), "workspace settings")


def _save_settings(workspace_path: str, settings: dict) -> None:
    ensure_cozter_dir(workspace_path)
    tmp_dir = os.path.join(workspace_path, COZTER_DIR)
    _atomic_write(_settings_path(workspace_path), settings, tmp_dir)


def _set_setting(workspace_path: str, key: str, value: str) -> None:
    settings = _load_settings(workspace_path)
    settings[key] = value
    _save_settings(workspace_path, settings)


def get_backend_name(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("backend", DEFAULT_BACKEND)


def set_backend_name(workspace_path: str, name: str) -> None:
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {AVAILABLE_BACKENDS}"
        )
    _set_setting(workspace_path, "backend", name)


def get_summary_backend_name(workspace_path: str) -> str:
    """Backend that runs compaction / auto-titling for this workspace."""
    return _load_settings(workspace_path).get(
        "summary_backend", DEFAULT_SUMMARY_BACKEND,
    )


def set_summary_backend_name(workspace_path: str, name: str) -> None:
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {AVAILABLE_BACKENDS}"
        )
    _set_setting(workspace_path, "summary_backend", name)


def get_available_models(workspace_path: str) -> list[str]:
    """List models for the workspace's currently selected backend."""
    backend_name = get_backend_name(workspace_path)
    return list(backends_agent.get_backend(backend_name).available_models)


def get_available_summary_models(workspace_path: str) -> list[str]:
    """List models for the workspace's summary backend (may differ from chat)."""
    backend_name = get_summary_backend_name(workspace_path)
    return list(backends_agent.get_backend(backend_name).available_models)


def _model_keys(backend_name: str) -> tuple[str, str]:
    """Return (model_key, summary_key) for the given backend."""
    return f"{backend_name}_model", f"{backend_name}_summary_model"


def _resolve_model(
    settings: dict, backend_name: str, summary: bool,
) -> str:
    backend = backends_agent.get_backend(backend_name)
    model_key, summary_key = _model_keys(backend_name)
    key = summary_key if summary else model_key
    default = (
        backend.default_summary_model if summary else backend.default_model
    )
    return settings.get(key, default)


def get_run_config(workspace_path: str) -> tuple[str, str, str, str, str]:
    """Return (chat_backend, model, summary_model, permission, summary_backend).

    The summary backend (and the summary model resolved against it) can
    differ from the chat backend - users may want, say, codex to run
    chat turns while a cheap local llama handles compaction.
    """
    s = _load_settings(workspace_path)
    backend_name = s.get("backend", DEFAULT_BACKEND)
    summary_backend = s.get("summary_backend", DEFAULT_SUMMARY_BACKEND)
    return (
        backend_name,
        _resolve_model(s, backend_name, summary=False),
        _resolve_model(s, summary_backend, summary=True),
        s.get("permission", DEFAULT_PERMISSION),
        summary_backend,
    )


def get_model(workspace_path: str) -> str:
    s = _load_settings(workspace_path)
    return _resolve_model(s, s.get("backend", DEFAULT_BACKEND), summary=False)


def set_model(workspace_path: str, model: str) -> None:
    backend_name = get_backend_name(workspace_path)
    model_key, _ = _model_keys(backend_name)
    _set_setting(workspace_path, model_key, model)


def get_summary_model(workspace_path: str) -> str:
    """Summary model scoped to the summary backend (not the chat backend)."""
    s = _load_settings(workspace_path)
    summary_backend = s.get("summary_backend", DEFAULT_SUMMARY_BACKEND)
    return _resolve_model(s, summary_backend, summary=True)


def set_summary_model(workspace_path: str, model: str) -> None:
    """Store the summary model under the summary backend's key."""
    summary_backend = get_summary_backend_name(workspace_path)
    _, summary_key = _model_keys(summary_backend)
    _set_setting(workspace_path, summary_key, model)


def get_permission(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("permission", DEFAULT_PERMISSION)


def set_permission(workspace_path: str, permission: str) -> None:
    _set_setting(workspace_path, "permission", permission)


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
