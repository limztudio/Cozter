import json
import os
import sys

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

_DEFAULT_CONFIG = {
    "telegram_bot_token": "",
    "user_ids": [],
    "update_check_interval": 10,
    "recent_workspace_limit": 10,
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
        print(f"Config file created at: {CONFIG_PATH}")
        print("Please fill in 'telegram_bot_token' and 'user_ids', then restart.")
        sys.exit(0)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    if not cfg.get("telegram_bot_token"):
        print(f"ERROR: 'telegram_bot_token' is empty in {CONFIG_PATH}")
        print("Fill it in and restart.")
        sys.exit(1)

    if not cfg.get("user_ids"):
        print(f"ERROR: 'user_ids' is empty in {CONFIG_PATH}")
        print("Add at least one Telegram user ID and restart.")
        sys.exit(1)

    return {**_DEFAULT_CONFIG, **cfg}
