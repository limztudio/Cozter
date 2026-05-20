"""Chat-platform backends - Telegram, Slack, Signal, and CLI adapters.

Use ``create_platforms(config)`` to build the right BotPlatform
instance(s) based on which token fields are present in the user's
config.json. Exactly one daemon chat surface must be set so that
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
    signal_phone = config.get("signal_phone_number") or ""
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

    if signal_phone:
        from .signal import SignalBot
        return [
            SignalBot(
                signal_phone,
                config.get("signal_group_urls") or [],
                recent_limit=recent_limit, max_queue_size=queue_size,
                jsonrpc_socket=config["signal_jsonrpc_socket"],
            ),
        ]

    raise ValueError(
        "config has no Telegram, Slack, or Signal platform set"
        " (normally caught by config.load_config)."
    )


__all__ = ["BotPlatform", "create_platforms"]
