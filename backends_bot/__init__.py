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
    if isinstance(tg_tokens, str):
        tg_tokens = [tg_tokens]
    if not isinstance(tg_tokens, list):
        tg_tokens = []
    tg_tokens = [t for t in tg_tokens if isinstance(t, str) and t.strip()]
    slack_bot = config.get("slack_bot_token") or ""
    if not isinstance(slack_bot, str):
        slack_bot = ""
    signal_groups = config.get("signal_group_urls") or []
    if isinstance(signal_groups, str):
        signal_groups = [signal_groups]
    if not isinstance(signal_groups, list):
        signal_groups = []
    signal_groups = [g for g in signal_groups if isinstance(g, str) and g.strip()]
    signal_socket = config.get("signal_jsonrpc_socket") or ""
    if not isinstance(signal_socket, str):
        signal_socket = ""
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
        raw_ids = config.get("user_ids") or []
        if isinstance(raw_ids, (str, int)) and not isinstance(raw_ids, bool):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            raw_ids = []
        user_ids = [
            str(t) for t in raw_ids
            if isinstance(t, (str, int)) and not isinstance(t, bool)
        ]
        bots.extend(
            TelegramBot(
                token, user_ids,
                recent_limit=recent_limit,
                max_queue_size=queue_size,
                max_upload_bytes=max_upload_bytes,
            )
            for token in tg_tokens
        )

    if slack_bot.strip():
        from .slack import SlackBot
        raw_channels = config.get("slack_channel_ids") or []
        if isinstance(raw_channels, str):
            raw_channels = [raw_channels]
        if not isinstance(raw_channels, list):
            raw_channels = []
        channel_ids = [
            c for c in raw_channels if isinstance(c, str) and c.strip()
        ]
        bots.append(
            SlackBot(
                slack_bot,
                config.get("slack_app_token") or "",
                channel_ids,
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
