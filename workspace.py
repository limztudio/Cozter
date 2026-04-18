import json
import logging
import os

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

AVAILABLE_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.2",
]
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_SUMMARY_MODEL = "gpt-5.3-codex"

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


def get_run_config(workspace_path: str) -> tuple[str, str, str]:
    """Return (model, summary_model, permission) from a single settings read."""
    s = _load_settings(workspace_path)
    return (
        s.get("model", DEFAULT_MODEL),
        s.get("summary_model", DEFAULT_SUMMARY_MODEL),
        s.get("permission", DEFAULT_PERMISSION),
    )


def get_model(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("model", DEFAULT_MODEL)


def set_model(workspace_path: str, model: str) -> None:
    _set_setting(workspace_path, "model", model)


def get_summary_model(workspace_path: str) -> str:
    return _load_settings(workspace_path).get(
        "summary_model", DEFAULT_SUMMARY_MODEL,
    )


def set_summary_model(workspace_path: str, model: str) -> None:
    _set_setting(workspace_path, "summary_model", model)


def get_permission(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("permission", DEFAULT_PERMISSION)


def set_permission(workspace_path: str, permission: str) -> None:
    _set_setting(workspace_path, "permission", permission)
