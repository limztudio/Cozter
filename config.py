import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

_DEFAULT_CONFIG = {
    "telegram_bot_token": "",
    "user_ids": [],
    "update_check_interval": 10,
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
        raise FileNotFoundError(
            f"Config created at {CONFIG_PATH} — fill in telegram_bot_token and user_ids, then restart."
        )

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    if not cfg.get("telegram_bot_token"):
        raise ValueError("telegram_bot_token is missing in config.json")
    if not cfg.get("user_ids"):
        raise ValueError("user_ids list is empty in config.json")

    return {**_DEFAULT_CONFIG, **cfg}
