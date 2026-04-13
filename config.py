import json
import os
import sys

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

_DEFAULT_CONFIG = {
    "telegram_bot_tokens": [],
    "user_ids": [],
    "update_check_interval": 10,
    "recent_workspace_limit": 10,
    "message_queue_size": 50,
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
        print(f"Config file created at: {CONFIG_PATH}")
        print(
            "Please fill in 'telegram_bot_tokens' and 'user_ids',"
            " then restart."
        )
        sys.exit(0)

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: config.json is corrupted or unreadable: {e}")
        print(f"Fix or delete {CONFIG_PATH}, then restart.")
        sys.exit(1)

    tokens = cfg.get("telegram_bot_tokens")
    if not tokens:
        print(f"ERROR: 'telegram_bot_tokens' is empty in {CONFIG_PATH}")
        print("Add at least one bot token and restart.")
        sys.exit(1)

    if not cfg.get("user_ids"):
        print(f"ERROR: 'user_ids' is empty in {CONFIG_PATH}")
        print("Add at least one Telegram user ID and restart.")
        sys.exit(1)

    return {**_DEFAULT_CONFIG, **cfg}
