import json
import os
import sys
from typing import cast

from .utils import CONFIG_DIR
from .utils import normalize_string_list

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

_DEFAULT_CONFIG = {
    "telegram_bot_tokens": [],
    "user_ids": [],
    "slack_bot_token": "",
    "slack_app_token": "",
    "slack_channel_ids": [],
    "signal_group_urls": [],
    "signal_jsonrpc_socket": "",
    "llama_server_url": "http://127.0.0.1:8080",
    "llama_max_agent_turns": 60,
    "llama_tool_repeat_limit": 3,
    "llama_socket_timeout": 1800,
    "llama_max_retries": 2,
    "zai_api_key": "",
    "zai_base_url": "https://api.z.ai/api/paas/v4",
    "zai_socket_timeout": 300,
    "zai_max_retries": 2,
    # Hard backstop on a single tool call, so a wedged plugin/custom tool
    # cannot block the whole turn indefinitely. Agent turns themselves are
    # not wall-clock limited; long-running work is allowed to finish.
    "tool_timeout": 120,
    # Diagnostic interval for the auto-update loop while it is waiting on
    # active turns. Reaching this interval emits stuck-turn diagnostics and
    # then continues waiting; it does not cancel the turn or force an update
    # restart through active work.
    "update_idle_timeout": 1200,
    # Interval (seconds) between automatic faulthandler traceback
    # dumps. 0 disables the periodic dump; the on-demand SIGUSR1
    # dump always works regardless. See __main__._enable_faulthandler.
    "dump_traceback_interval": 0,
    "update_check_interval": 10,
    "recent_workspace_limit": 10,
    "message_queue_size": 50,
    "extra_models": {},
    "max_permission": "full",
    "show_usage": True,
}


def _load_config_object() -> dict:
    """Read config.json and require the top-level JSON value to be an object."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain a JSON object")
    return cfg


def _read_config_value(key: str):
    """Read a single key from config.json on each call.

    Returns ``None`` if the file is missing (CLI mode without setup).
    JSON / OS errors propagate - daemon mode validates the file in
    :func:`load_config` before any getter runs, and a broken config
    in either mode should surface as an error rather than silently
    falling back to defaults.
    """
    if not os.path.exists(CONFIG_PATH):
        return None
    return _load_config_object().get(key)


def _get_nonempty_string(key: str) -> str:
    """Return ``cfg[key]`` if it's a non-blank string, else the default."""
    val = _read_config_value(key)
    if isinstance(val, str):
        val = val.strip()
        if val:
            return val
    return cast(str, _DEFAULT_CONFIG[key])


def _get_positive_int(key: str) -> int:
    """Return ``cfg[key]`` if it's an ``int > 0``, else the default."""
    val = _read_config_value(key)
    if isinstance(val, int) and not isinstance(val, bool) and val > 0:
        return val
    return cast(int, _DEFAULT_CONFIG[key])


def _get_non_negative_int(key: str) -> int:
    """Return ``cfg[key]`` if it's an ``int >= 0``, else the default.

    Like :func:`_get_positive_int` but permits 0 (used as an "off" flag
    for optional periodic features such as the traceback dump).
    """
    val = _read_config_value(key)
    if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
        return val
    return cast(int, _DEFAULT_CONFIG[key])


def get_llama_server_url() -> str:
    return _get_nonempty_string("llama_server_url")


def get_llama_max_agent_turns() -> int:
    """Return the per-turn cap on llama agent-loop iterations."""
    return _get_positive_int("llama_max_agent_turns")


def get_llama_tool_repeat_limit() -> int:
    """Return the cap on identical repeated tool calls within a turn."""
    return _get_positive_int("llama_tool_repeat_limit")


def get_llama_socket_timeout() -> int:
    """Return the per-socket-read timeout in seconds for the llama HTTP call.

    A slow llama-server (heavy model, large context, weak hardware) can
    take many minutes to emit the first byte of a response, especially
    after a tool turn folds a large file's contents back into context.
    The default is intentionally generous; lower it only if you have a
    fast server and want failures to surface quickly.
    """
    return _get_positive_int("llama_socket_timeout")


def get_llama_max_retries() -> int:
    """Retry attempts for transient llama HTTP failures (>= 0; 0 disables).

    Uses its own reader rather than :func:`_get_positive_int` because 0 is
    a meaningful value here ("do not retry"), not an invalid one.
    """
    val = _read_config_value("llama_max_retries")
    if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
        return val
    return cast(int, _DEFAULT_CONFIG["llama_max_retries"])


def get_show_usage() -> bool:
    """Whether to append a per-turn token/cost footer to replies.

    Only backends that report usage (codex's turn.completed, claude_code's
    result) produce one; others are silent regardless. Defaults to True.
    """
    val = _read_config_value("show_usage")
    if isinstance(val, bool):
        return val
    return cast(bool, _DEFAULT_CONFIG["show_usage"])


def get_max_permission() -> str:
    """Highest permission any workspace may use - an operator-wide cap.

    Defaults to ``"full"`` (no cap). Set it to e.g. ``"auto"`` to forbid
    the sandbox-bypassing ``full`` mode across every workspace, or
    ``"deny"`` for a read-only bot. Invalid values fall back to the
    default. Enforced in :mod:`workspace` (clamps the effective permission
    and rejects setting a higher one via ``/permission``).
    """
    val = _read_config_value("max_permission")
    if isinstance(val, str) and val in ("full", "auto", "confirm", "deny"):
        return val
    return cast(str, _DEFAULT_CONFIG["max_permission"])


def get_zai_api_key() -> str:
    """Z.ai (Zhipu GLM) API key for the ``zai`` backend; "" if unset."""
    val = _read_config_value("zai_api_key")
    return val.strip() if isinstance(val, str) else ""


def get_zai_base_url() -> str:
    """Base URL for Z.ai's OpenAI-compatible endpoint (includes the version)."""
    return _get_nonempty_string("zai_base_url")


def get_zai_socket_timeout() -> int:
    """Per-socket-read timeout (seconds) for zai HTTP calls."""
    return _get_positive_int("zai_socket_timeout")


def get_zai_max_retries() -> int:
    """Retry attempts for transient zai HTTP failures (>= 0; 0 disables)."""
    val = _read_config_value("zai_max_retries")
    if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
        return val
    return cast(int, _DEFAULT_CONFIG["zai_max_retries"])


def get_tool_timeout() -> int:
    """Wall-clock ceiling (seconds) for a single agent tool call.

    Even built-ins like ``bash`` enforce their own per-call timeout, but
    a plugin or a custom tool can hang (blocking I/O, infinite loop) and
    block the whole turn. This wraps every ``execute_tool`` call as a
    safety net. Defaults to 120s, matching bash's hard cap.
    """
    return _get_positive_int("tool_timeout")


def get_update_idle_timeout() -> int:
    """Diagnostic interval while the update loop waits for active turns.

    Long-running turns are allowed to finish. If the wait reaches this
    interval, the loop dumps diagnostics and continues waiting instead of
    restarting through active work. Defaults to 1200s.
    """
    return _get_positive_int("update_idle_timeout")


def get_dump_traceback_interval() -> int:
    """Interval (seconds) between automatic thread-traceback dumps.

    Implemented via ``faulthandler.dump_traceback_later`` so a wedged
    daemon (deadlock, busy loop in C) leaves periodic evidence in the
    log even though no Python exception is raised. 0 disables the
    periodic dump; the on-demand ``SIGUSR1`` dump always works.
    """
    return _get_non_negative_int("dump_traceback_interval")


def get_extra_models(backend_name: str) -> list[str]:
    """Extra model IDs to offer for *backend_name*, from config.json.

    The built-in per-backend lists in ``backends_agent/`` are a curated
    snapshot and go stale as providers ship new models. This lets users
    expose newer or private/self-hosted models without editing source::

        {
            "extra_models": {
                "codex": ["my-private-codex-model"],
                "copilot": ["..."],
            }
        }

    Malformed entries (missing key, non-object value, non-string items)
    are ignored, returning an empty list.
    """
    val = _read_config_value("extra_models")
    if not isinstance(val, dict):
        return []
    return normalize_string_list(val.get(backend_name))


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
        print(f"Config file created at: {CONFIG_PATH}")
        print(
            "Fill in either 'telegram_bot_tokens' + 'user_ids'"
            " or 'slack_bot_token' + 'slack_app_token' +"
            " 'slack_channel_ids', or 'signal_group_urls' +"
            " 'signal_jsonrpc_socket', then restart."
        )
        sys.exit(0)

    try:
        cfg = _load_config_object()
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: config.json is corrupted or unreadable: {e}")
        print(f"Fix or delete {CONFIG_PATH}, then restart.")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"Fix or delete {CONFIG_PATH}, then restart.")
        sys.exit(1)

    cfg = {**_DEFAULT_CONFIG, **cfg}

    # Filter whitespace-only / empty tokens so users who leave placeholders
    # in the file get a "not configured" error rather than a runtime
    # auth-failure later.
    cfg["telegram_bot_tokens"] = normalize_string_list(
        cfg.get("telegram_bot_tokens") or []
    )
    slack_bot_raw = cfg.get("slack_bot_token") or ""
    cfg["slack_bot_token"] = (
        slack_bot_raw.strip() if isinstance(slack_bot_raw, str) else ""
    )
    slack_app_raw = cfg.get("slack_app_token") or ""
    cfg["slack_app_token"] = (
        slack_app_raw.strip() if isinstance(slack_app_raw, str) else ""
    )
    cfg["signal_group_urls"] = normalize_string_list(
        cfg.get("signal_group_urls") or [], allow_scalar=True,
    )
    signal_socket_raw = cfg.get("signal_jsonrpc_socket") or ""
    cfg["signal_jsonrpc_socket"] = (
        os.path.expandvars(os.path.expanduser(signal_socket_raw.strip()))
        if isinstance(signal_socket_raw, str) else ""
    )

    has_telegram = bool(cfg["telegram_bot_tokens"])
    has_slack = bool(cfg["slack_bot_token"])
    has_signal = bool(cfg["signal_group_urls"] or cfg["signal_jsonrpc_socket"])

    configured = sum(bool(x) for x in (has_telegram, has_slack, has_signal))
    if configured > 1:
        print(
            f"ERROR: {CONFIG_PATH} has more than one chat platform set."
        )
        print(
            "Pick one of Telegram, Slack, or Signal - sessions and"
            " workspace state aren't shared across platforms."
        )
        sys.exit(1)
    if configured == 0:
        print(
            f"ERROR: {CONFIG_PATH} must set either 'telegram_bot_tokens'"
            ", 'slack_bot_token', or 'signal_group_urls' +"
            " 'signal_jsonrpc_socket'."
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
        slack_channels = normalize_string_list(
            cfg.get("slack_channel_ids") or []
        )
        cfg["slack_channel_ids"] = slack_channels
        if not slack_channels:
            print(f"ERROR: 'slack_channel_ids' is empty in {CONFIG_PATH}")
            print(
                "Add at least one Slack channel ID (C..., G..., D...,"
                " or MP...) and restart."
            )
            sys.exit(1)
    if has_signal and not cfg["signal_group_urls"]:
        print(f"ERROR: 'signal_group_urls' is empty in {CONFIG_PATH}")
        print("Add at least one Signal group invite URL.")
        sys.exit(1)
    if has_signal and not cfg["signal_jsonrpc_socket"]:
        print(f"ERROR: 'signal_jsonrpc_socket' is empty in {CONFIG_PATH}")
        print("Point it at the shared signal-cli daemon Unix socket.")
        sys.exit(1)

    return cfg
