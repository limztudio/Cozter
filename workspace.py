import json
import os
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
WORKSPACE_STATE_PATH = os.path.join(CONFIG_DIR, "workspaces.json")
COZTER_DIR_NAME = ".cozter"


def _load_state() -> dict:
    if os.path.exists(WORKSPACE_STATE_PATH):
        with open(WORKSPACE_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"current": None, "recent": []}


def _save_state(state: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(WORKSPACE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_current() -> str | None:
    return _load_state().get("current")


def get_recent(limit: int = 10) -> list[str]:
    return _load_state().get("recent", [])[:limit]


def select_workspace(path: str) -> None:
    """Set path as current workspace and push it to the top of recent list."""
    state = _load_state()
    state["current"] = path
    recent = state.get("recent", [])
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    state["recent"] = recent
    _save_state(state)


def ensure_cozter_dir(path: str) -> None:
    """Create .cozter folder inside the workspace if it doesn't exist."""
    cozter_path = os.path.join(path, COZTER_DIR_NAME)
    os.makedirs(cozter_path, exist_ok=True)
