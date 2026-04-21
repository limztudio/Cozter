import json
import logging
import os

from . import backends_agent
from .utils import atomic_write as _atomic_write

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
WORKSPACE_STATE_PATH = os.path.join(CONFIG_DIR, "workspaces.json")
COZTER_DIR_NAME = ".cozter"
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


def ensure_cozter_dir(path: str) -> None:
    """Create .cozter folder inside the workspace if it doesn't exist."""
    cozter_path = os.path.join(path, COZTER_DIR_NAME)
    os.makedirs(cozter_path, exist_ok=True)


# ---------------------------------------------------------------------------
# Workspace settings (stored in .cozter/settings.json)
# ---------------------------------------------------------------------------

AVAILABLE_BACKENDS = backends_agent.AVAILABLE_BACKENDS
DEFAULT_BACKEND = backends_agent.DEFAULT_BACKEND

AVAILABLE_PERMISSIONS = ["full", "auto", "confirm", "deny"]
DEFAULT_PERMISSION = "auto"
PERMISSION_DESCRIPTIONS = {
    "full": "Full access - bypass all approvals and sandbox",
    "auto": "Execute all tool calls automatically (sandboxed)",
    "confirm": "Ask before each tool call",
    "deny": "Block all tool calls (text-only responses)",
}


def _settings_path(workspace_path: str) -> str:
    return os.path.join(workspace_path, COZTER_DIR_NAME, "settings.json")


def _load_settings(workspace_path: str) -> dict:
    return _load_json(_settings_path(workspace_path), "workspace settings")


def _save_settings(workspace_path: str, settings: dict) -> None:
    ensure_cozter_dir(workspace_path)
    tmp_dir = os.path.join(workspace_path, COZTER_DIR_NAME)
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


def get_available_models(workspace_path: str) -> list[str]:
    """List models for the workspace's currently selected backend."""
    backend_name = get_backend_name(workspace_path)
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
    # Legacy fallback: settings prior to the backend split stored bare
    # "model" / "summary_model" keys scoped implicitly to the codex backend.
    if backend_name == "codex":
        legacy = "summary_model" if summary else "model"
        return settings.get(key, settings.get(legacy, default))
    return settings.get(key, default)


def get_run_config(workspace_path: str) -> tuple[str, str, str, str]:
    """Return (backend, model, summary_model, permission) from one read."""
    s = _load_settings(workspace_path)
    backend_name = s.get("backend", DEFAULT_BACKEND)
    return (
        backend_name,
        _resolve_model(s, backend_name, summary=False),
        _resolve_model(s, backend_name, summary=True),
        s.get("permission", DEFAULT_PERMISSION),
    )


def get_model(workspace_path: str) -> str:
    s = _load_settings(workspace_path)
    return _resolve_model(s, s.get("backend", DEFAULT_BACKEND), summary=False)


def set_model(workspace_path: str, model: str) -> None:
    backend_name = get_backend_name(workspace_path)
    model_key, _ = _model_keys(backend_name)
    _set_setting(workspace_path, model_key, model)


def get_summary_model(workspace_path: str) -> str:
    s = _load_settings(workspace_path)
    return _resolve_model(s, s.get("backend", DEFAULT_BACKEND), summary=True)


def set_summary_model(workspace_path: str, model: str) -> None:
    backend_name = get_backend_name(workspace_path)
    _, summary_key = _model_keys(backend_name)
    _set_setting(workspace_path, summary_key, model)


def get_permission(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("permission", DEFAULT_PERMISSION)


def set_permission(workspace_path: str, permission: str) -> None:
    _set_setting(workspace_path, "permission", permission)
