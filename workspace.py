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

AVAILABLE_MODELS = ["o3", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]
DEFAULT_MODEL = "gpt-4o"

AVAILABLE_EFFORTS = ["low", "medium", "high"]
DEFAULT_EFFORT = "medium"


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


def get_effort(workspace_path: str) -> str:
    return _load_settings(workspace_path).get("effort", DEFAULT_EFFORT)


def set_effort(workspace_path: str, effort: str) -> None:
    settings = _load_settings(workspace_path)
    settings["effort"] = effort
    _save_settings(workspace_path, settings)


DEFAULT_COMPACT_INTERVAL = 15
DEFAULT_REREAD_INTERVAL = 30


def get_compact_interval(workspace_path: str) -> int:
    return _load_settings(workspace_path).get("compact_interval", DEFAULT_COMPACT_INTERVAL)


def get_reread_interval(workspace_path: str) -> int:
    return _load_settings(workspace_path).get("reread_interval", DEFAULT_REREAD_INTERVAL)
