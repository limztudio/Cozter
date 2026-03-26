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
