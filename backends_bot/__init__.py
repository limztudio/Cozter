"""Chat-platform backends - Telegram, Slack, Signal, and CLI adapters.

Use ``create_platforms(config)`` to build the right BotPlatform
instance(s) based on which token fields are present in the user's
config.json. Multiple daemon chat surfaces may be set at once; every
configured platform starts side by side and shares the same workspaces
(workspace files, sessions, and locks are per-workspace, while each
platform keeps its own current-workspace pointer, queues, and delivery
ledger keyed by its platform id).
"""

from .base import BotPlatform
from ..config import (
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_MESSAGE_QUEUE_SIZE,
    DEFAULT_RECENT_WORKSPACE_LIMIT,
)


def create_platforms(config: dict) -> list[BotPlatform]:
    """Build the BotPlatform list dictated by *config*.

    Telegram supports multiple tokens (one bot instance per token);
    Slack is single-instance. Every configured surface is started: set
    any combination of Telegram, Slack, and Signal to share all
    workspaces across every platform at once. ``config`` is expected to
    have been validated by ``config.load_config`` — this function is a
    pure dispatcher, not a validator.
    """
    tg_tokens = config.get("telegram_bot_tokens") or []
    slack_bot = config.get("slack_bot_token") or ""
    signal_groups = config.get("signal_group_urls") or []
    signal_socket = config.get("signal_jsonrpc_socket") or ""
    recent_limit = config.get(
        "recent_workspace_limit", DEFAULT_RECENT_WORKSPACE_LIMIT,
    )
    queue_size = config.get("message_queue_size", DEFAULT_MESSAGE_QUEUE_SIZE)
    max_upload_bytes = config.get(
        "max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES,
    )

    bots: list[BotPlatform] = []

    if tg_tokens:
        # Deferred import to avoid requiring slack_bolt at telegram-only
        # deploys (and vice versa).
        from .telegram import TelegramBot
        bots.extend(
            TelegramBot(
                token, config.get("user_ids") or [],
                recent_limit=recent_limit,
                max_queue_size=queue_size,
                max_upload_bytes=max_upload_bytes,
            )
            for token in tg_tokens
        )

    if slack_bot:
        from .slack import SlackBot
        bots.append(
            SlackBot(
                slack_bot,
                config.get("slack_app_token") or "",
                config.get("slack_channel_ids") or [],
                recent_limit=recent_limit,
                max_queue_size=queue_size,
                max_upload_bytes=max_upload_bytes,
            ),
        )

    if signal_groups or signal_socket:
        from .signal import SignalBot
        bots.append(
            SignalBot(
                signal_groups,
                recent_limit=recent_limit,
                max_queue_size=queue_size,
                max_upload_bytes=max_upload_bytes,
                jsonrpc_socket=signal_socket,
            ),
        )

    if not bots:
        raise ValueError(
            "config has no Telegram, Slack, or Signal platform set"
            " (normally caught by config.load_config)."
        )
    return bots


__all__ = ["BotPlatform", "create_platforms"]
