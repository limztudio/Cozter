import json
import os
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
WORKSPACE_STATE_PATH = os.path.join(CONFIG_DIR, "workspaces.json")
COZTER_DIR_NAME = ".cozter"


def _load_all() -> dict:
    """Load the full state: {user_id_str: {current, recent}, ...}"""
    if os.path.exists(WORKSPACE_STATE_PATH):
        with open(WORKSPACE_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_all(data: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(WORKSPACE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_user(user_id: int) -> dict:
    return _load_all().get(str(user_id), {"current": None, "recent": []})


def get_current(user_id: int) -> str | None:
    return _get_user(user_id).get("current")


def get_recent(user_id: int, limit: int = 10) -> list[str]:
    return _get_user(user_id).get("recent", [])[:limit]


def select_workspace(user_id: int, path: str) -> None:
    """Set path as current workspace and push it to the top of recent list."""
    all_state = _load_all()
    uid = str(user_id)
    user_state = all_state.get(uid, {"current": None, "recent": []})
    user_state["current"] = path
    recent = user_state.get("recent", [])
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    user_state["recent"] = recent
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


def _settings_path(workspace_path: str) -> str:
    return os.path.join(workspace_path, COZTER_DIR_NAME, "settings.json")


def _load_settings(workspace_path: str) -> dict:
    path = _settings_path(workspace_path)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_settings(workspace_path: str, settings: dict) -> None:
    ensure_cozter_dir(workspace_path)
    with open(_settings_path(workspace_path), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def get_model(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("model", DEFAULT_MODEL)


def set_model(workspace_path: str, model: str) -> None:
    settings = _load_settings(workspace_path)
    settings["model"] = model
    _save_settings(workspace_path, settings)


DEFAULT_COMPACT_INTERVAL = 20

AVAILABLE_PERMISSIONS = ["auto", "confirm", "deny"]
DEFAULT_PERMISSION = "auto"
PERMISSION_DESCRIPTIONS = {
    "auto": "Execute all tool calls automatically",
    "confirm": "Ask before each tool call",
    "deny": "Block all tool calls (text-only responses)",
}


def get_permission(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("permission", DEFAULT_PERMISSION)


def set_permission(workspace_path: str, permission: str) -> None:
    settings = _load_settings(workspace_path)
    settings["permission"] = permission
    _save_settings(workspace_path, settings)
