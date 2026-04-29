"""Chat-platform backends - Telegram and Slack adapters.

Use ``create_platforms(config)`` to build the right BotPlatform
instance(s) based on which token fields are present in the user's
config.json. Exactly one of ``telegram_bot_tokens`` or
``slack_bot_token`` must be set - having both is rejected so that
session state isn't fragmented across platforms.
"""

from .base import BotPlatform


def create_platforms(config: dict) -> list[BotPlatform]:
    """Build the BotPlatform list dictated by *config*.

    Telegram supports multiple tokens (one bot instance per token);
    Slack is single-instance. ``config`` is expected to have been
    validated by ``config.load_config`` — this function is a pure
    dispatcher, not a validator.
    """
    tg_tokens = config.get("telegram_bot_tokens") or []
    slack_bot = config.get("slack_bot_token") or ""
    recent_limit = config.get("recent_workspace_limit", 10)
    queue_size = config.get("message_queue_size", 50)

    if tg_tokens:
        # Deferred import to avoid requiring slack_bolt at telegram-only
        # deploys (and vice versa).
        from .telegram import TelegramBot
        return [
            TelegramBot(
                token, config.get("user_ids") or [],
                recent_limit=recent_limit, max_queue_size=queue_size,
            )
            for token in tg_tokens
        ]

    if slack_bot:
        from .slack import SlackBot
        return [
            SlackBot(
                slack_bot,
                config.get("slack_app_token") or "",
                config.get("slack_channel_ids") or [],
                recent_limit=recent_limit, max_queue_size=queue_size,
            ),
        ]

    raise ValueError(
        "config has neither telegram_bot_tokens nor slack_bot_token set"
        " (normally caught by config.load_config)."
    )


__all__ = ["BotPlatform", "create_platforms"]
