import json
import os
import sys

from .utils import CONFIG_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

_DEFAULT_CONFIG = {
    "telegram_bot_tokens": [],
    "user_ids": [],
    "slack_bot_token": "",
    "slack_app_token": "",
    "slack_channel_ids": [],
    "llama_server_url": "http://127.0.0.1:8080",
    "llama_max_agent_turns": 60,
    "llama_tool_repeat_limit": 3,
    "update_check_interval": 10,
    "recent_workspace_limit": 10,
    "message_queue_size": 50,
}


def get_llama_server_url() -> str:
    """Return the configured llama-server URL without bot-token validation.

    CLI mode never calls load_config (it would create a daemon-only
    config.json), so the llama backend has to read the file itself
    if one exists - and fall back to the default otherwise.
    """
    if not os.path.exists(CONFIG_PATH):
        return _DEFAULT_CONFIG["llama_server_url"]
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _DEFAULT_CONFIG["llama_server_url"]
    url = cfg.get("llama_server_url") or _DEFAULT_CONFIG["llama_server_url"]
    if isinstance(url, str):
        return url.strip() or _DEFAULT_CONFIG["llama_server_url"]
    return _DEFAULT_CONFIG["llama_server_url"]


def get_llama_max_agent_turns() -> int:
    """Return the per-turn cap on llama agent-loop iterations."""
    if not os.path.exists(CONFIG_PATH):
        return _DEFAULT_CONFIG["llama_max_agent_turns"]
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _DEFAULT_CONFIG["llama_max_agent_turns"]
    val = cfg.get("llama_max_agent_turns")
    if isinstance(val, int) and val > 0:
        return val
    return _DEFAULT_CONFIG["llama_max_agent_turns"]


def get_llama_tool_repeat_limit() -> int:
    """Return the cap on identical repeated tool calls within a turn."""
    if not os.path.exists(CONFIG_PATH):
        return _DEFAULT_CONFIG["llama_tool_repeat_limit"]
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _DEFAULT_CONFIG["llama_tool_repeat_limit"]
    val = cfg.get("llama_tool_repeat_limit")
    if isinstance(val, int) and val > 0:
        return val
    return _DEFAULT_CONFIG["llama_tool_repeat_limit"]


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
        print(f"Config file created at: {CONFIG_PATH}")
        print(
            "Fill in either 'telegram_bot_tokens' + 'user_ids'"
            " or 'slack_bot_token' + 'slack_app_token' +"
            " 'slack_channel_ids', then restart."
        )
        sys.exit(0)

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: config.json is corrupted or unreadable: {e}")
        print(f"Fix or delete {CONFIG_PATH}, then restart.")
        sys.exit(1)

    cfg = {**_DEFAULT_CONFIG, **cfg}

    # Filter whitespace-only / empty tokens so users who leave placeholders
    # in the file get a "not configured" error rather than a runtime
    # auth-failure later.
    tg_tokens = [
        t for t in (cfg.get("telegram_bot_tokens") or [])
        if isinstance(t, str) and t.strip()
    ]
    cfg["telegram_bot_tokens"] = tg_tokens
    slack_bot_raw = cfg.get("slack_bot_token") or ""
    cfg["slack_bot_token"] = (
        slack_bot_raw.strip() if isinstance(slack_bot_raw, str) else ""
    )
    slack_app_raw = cfg.get("slack_app_token") or ""
    cfg["slack_app_token"] = (
        slack_app_raw.strip() if isinstance(slack_app_raw, str) else ""
    )

    has_telegram = bool(cfg["telegram_bot_tokens"])
    has_slack = bool(cfg["slack_bot_token"])

    if has_telegram and has_slack:
        print(
            f"ERROR: {CONFIG_PATH} has both 'telegram_bot_tokens' and"
            " 'slack_bot_token' set."
        )
        print(
            "Pick one - sessions and workspace state aren't shared"
            " across platforms."
        )
        sys.exit(1)
    if not has_telegram and not has_slack:
        print(
            f"ERROR: {CONFIG_PATH} must set either 'telegram_bot_tokens'"
            " or 'slack_bot_token'."
        )
        sys.exit(1)

    if has_telegram and not cfg.get("user_ids"):
        print(f"ERROR: 'user_ids' is empty in {CONFIG_PATH}")
        print("Add at least one Telegram user ID and restart.")
        sys.exit(1)
    if has_slack:
        if not cfg.get("slack_app_token"):
            print(
                f"ERROR: 'slack_app_token' (xapp-...) is required for"
                f" Socket Mode in {CONFIG_PATH}."
            )
            sys.exit(1)
        # Normalize: drop non-string / whitespace-only entries so a stray
        # placeholder doesn't pass the populated-list check.
        slack_channels = [
            c for c in (cfg.get("slack_channel_ids") or [])
            if isinstance(c, str) and c.strip()
        ]
        cfg["slack_channel_ids"] = slack_channels
        if not slack_channels:
            print(f"ERROR: 'slack_channel_ids' is empty in {CONFIG_PATH}")
            print(
                "Add at least one Slack channel ID (C..., G..., D...,"
                " or MP...) and restart."
            )
            sys.exit(1)

    return cfg
