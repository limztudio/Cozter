import json
import os
import sys
from typing import cast
from urllib.parse import urlsplit

from .utils import CONFIG_DIR
from .utils import normalize_string_list
from .utils import stat_mtime_size
from .utils import validate_json_nesting

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_UPDATE_CHECK_INTERVAL = 300
DEFAULT_RECENT_WORKSPACE_LIMIT = 10
DEFAULT_MESSAGE_QUEUE_SIZE = 5000
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_PERMISSION_LEVELS = frozenset({"full", "auto", "confirm", "deny"})

_DEFAULT_CONFIG = {
    "telegram_bot_tokens": [],
    "user_ids": [],
    "slack_bot_token": "",
    "slack_app_token": "",
    "slack_channel_ids": [],
    "signal_group_urls": [],
    "signal_jsonrpc_socket": "",
    "signal_account": "",
    "signal_own_device_id": 0,
    "llama_server_url": "http://127.0.0.1:8080",
    "llama_max_agent_turns": 60,
    "llama_tool_repeat_limit": 3,
    "llama_max_retries": 2,
    "zai_api_key": "",
    "zai_base_url": "https://api.z.ai/api/paas/v4",
    "zai_max_retries": 2,
    "meta_api_key": "",
    "meta_base_url": "https://api.meta.ai/v1",
    "meta_max_retries": 2,
    "flexible_max_parallel_workers": 3,
    "update_idle_timeout": 1200,
    "dump_traceback_interval": 0,
    "update_check_interval": DEFAULT_UPDATE_CHECK_INTERVAL,
    "recent_workspace_limit": DEFAULT_RECENT_WORKSPACE_LIMIT,
    "message_queue_size": DEFAULT_MESSAGE_QUEUE_SIZE,
    "max_upload_bytes": DEFAULT_MAX_UPLOAD_BYTES,
    "extra_models": {},
    "model_context_windows": {},
    "max_permission": "auto",
    "show_usage": True,
}


_config_cache_path: str | None = None
_config_cache_mtime_ns: int | None = None
_config_cache_size: int | None = None
_config_cache_data: dict | None = None


def _load_config_object() -> dict:
    """Read config.json and require the top-level JSON value to be an object.

    Results are cached by path + mtime + size so runtime getters avoid
    re-parsing the file on every call. A config edit changes mtime/size
    and takes effect on the next getter without a restart.
    """
    global _config_cache_path, _config_cache_mtime_ns
    global _config_cache_size, _config_cache_data
    mtime_ns, size = stat_mtime_size(CONFIG_PATH)
    if (
        mtime_ns is not None
        and _config_cache_data is not None
        and _config_cache_path == CONFIG_PATH
        and mtime_ns == _config_cache_mtime_ns
        and size == _config_cache_size
    ):
        return _config_cache_data
    with open(CONFIG_PATH, encoding="utf-8") as file_handle:
        cfg = json.load(file_handle)
    validate_json_nesting(cfg)
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain a JSON object")
    # An unstatable file must still be read and validated, without trusting
    # or replacing the last successfully stamped cache entry.
    if mtime_ns is not None:
        _config_cache_path = CONFIG_PATH
        _config_cache_mtime_ns = mtime_ns
        _config_cache_size = size
        _config_cache_data = cfg
    return cfg


def _config_dir() -> str:
    """Return the actual parent of the configurable config path."""
    return os.path.dirname(os.path.abspath(CONFIG_PATH)) or "."


def _restrict_config_permissions(path: str, mode: int) -> None:
    """Enforce owner-only POSIX permissions for token-bearing config state.

    POSIX's normal umask commonly makes new files world-readable.  Config
    holds bot and API tokens, so restrict both the directory and file on
    creation and tighten pre-existing installations on their next startup.
    Windows ACLs are not expressible through :func:`os.chmod`, so Windows
    deployments retain their configured directory ACLs instead.
    """
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise RuntimeError(
            f"Could not restrict token config permissions for {path}: {exc}",
        ) from exc


def _create_default_config() -> None:
    """Create the initial config with owner-only file and directory modes."""
    config_dir = _config_dir()
    os.makedirs(config_dir, mode=0o700, exist_ok=True)
    _restrict_config_permissions(config_dir, 0o700)

    # os.open's mode guarantees no group/other read bits.
    fd = os.open(
        CONFIG_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
        json.dump(_DEFAULT_CONFIG, file_handle, indent=2)
    _restrict_config_permissions(CONFIG_PATH, 0o600)


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


def _get_https_url(key: str) -> str:
    """Return a configured HTTPS URL, falling back safely when malformed.

    Some endpoints are called with a bearer token.  Treating an accidental
    ``http://`` override as valid would transmit that secret in clear text,
    so those getters use this narrower form instead of the generic non-empty
    string reader.
    """
    value = _get_nonempty_string(key)
    try:
        parsed = urlsplit(value)
        # urlsplit defers malformed/out-of-range port validation until this
        # property is read. Reject those overrides before request construction.
        parsed.port
    except ValueError:
        return cast(str, _DEFAULT_CONFIG[key])
    if parsed.scheme.lower() == "https" and parsed.hostname:
        return value
    return cast(str, _DEFAULT_CONFIG[key])


def _get_int_at_least(key: str, minimum: int) -> int:
    """Return ``cfg[key]`` if it is an int at least *minimum*, else default."""
    val = _read_config_value(key)
    if (
        isinstance(val, int)
        and not isinstance(val, bool)
        and val >= minimum
    ):
        return val
    return cast(int, _DEFAULT_CONFIG[key])


def get_llama_server_url() -> str:
    return _get_nonempty_string("llama_server_url")


def get_llama_max_agent_turns() -> int:
    """Return the per-turn cap on llama agent-loop iterations."""
    return _get_int_at_least("llama_max_agent_turns", 1)


def get_llama_tool_repeat_limit() -> int:
    """Return the cap on identical repeated tool calls within a turn."""
    return _get_int_at_least("llama_tool_repeat_limit", 1)


def get_llama_max_retries() -> int:
    """Retry attempts for transient llama HTTP failures (>= 0; 0 disables).

    Zero is meaningful here ("do not retry"), so this uses the shared
    integer reader with a zero minimum.
    """
    return _get_int_at_least("llama_max_retries", 0)


def get_show_usage() -> bool:
    """Whether to append a per-turn token/cost footer to replies.

    Only backends that report usage (codex's turn.completed, Claude Code's
    and Grok's result) produce one; others are silent regardless. Defaults
    to True.
    """
    val = _read_config_value("show_usage")
    if isinstance(val, bool):
        return val
    return cast(bool, _DEFAULT_CONFIG["show_usage"])


def get_max_permission() -> str:
    """Highest permission any workspace may use - an operator-wide cap.

    Defaults to ``"auto"``, which forbids the sandbox-bypassing ``"full"``
    mode across every workspace. An operator must explicitly set ``"full"``
    to allow that bypass, or can use ``"deny"`` for a read-only bot. Runtime
    config edits that are malformed fail closed to ``"deny"`` rather than
    accidentally widening a previously restrictive ceiling. Daemon startup
    rejects malformed values with a clear error. Enforced in :mod:`workspace`
    (clamps the effective permission and rejects setting a higher one via
    ``/permission``).
    """
    if not os.path.exists(CONFIG_PATH):
        return cast(str, _DEFAULT_CONFIG["max_permission"])
    # A missing key retains the default; an explicit JSON null is malformed
    # and must not widen a previously restrictive ceiling.
    val = _load_config_object().get(
        "max_permission", _DEFAULT_CONFIG["max_permission"],
    )
    if isinstance(val, str):
        normalized = val.strip()
        if normalized in _PERMISSION_LEVELS:
            return normalized
    return "deny"


def get_zai_api_key() -> str:
    """Z.ai (Zhipu GLM) API key for the ``zai`` backend; "" if unset."""
    return _get_nonempty_string("zai_api_key")


def get_zai_base_url() -> str:
    """HTTPS base URL for Z.ai's OpenAI-compatible bearer-token endpoint."""
    return _get_https_url("zai_base_url")


def get_zai_max_retries() -> int:
    """Retry attempts for transient zai HTTP failures (>= 0; 0 disables)."""
    return _get_int_at_least("zai_max_retries", 0)


def get_meta_api_key() -> str:
    """Meta Model API key for the ``meta`` backend; "" if unset."""
    return _get_nonempty_string("meta_api_key")


def get_meta_base_url() -> str:
    """HTTPS base URL for Meta's OpenAI-compatible bearer-token endpoint."""
    return _get_https_url("meta_base_url")


def get_meta_max_retries() -> int:
    """Retry attempts for transient Meta Model API failures (>= 0)."""
    return _get_int_at_least("meta_max_retries", 0)


def get_update_idle_timeout() -> int:
    """Diagnostic interval while the update loop waits for active turns.

    Long-running turns are allowed to finish. If the wait reaches this
    interval, the loop dumps diagnostics and continues waiting instead of
    restarting through active work. Defaults to 1200s.
    """
    return _get_int_at_least("update_idle_timeout", 1)


def get_flexible_max_parallel_workers() -> int:
    """Concurrent independent read-only workers (1 disables; valid range 1-4)."""
    value = _read_config_value("flexible_max_parallel_workers")
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 4:
        return value
    return cast(int, _DEFAULT_CONFIG["flexible_max_parallel_workers"])


def get_dump_traceback_interval() -> int:
    """Interval (seconds) between automatic thread-traceback dumps.

    Implemented via ``faulthandler.dump_traceback_later`` so a wedged
    daemon (deadlock, busy loop in C) leaves periodic evidence in the
    log even though no Python exception is raised. 0 disables the
    periodic dump; the on-demand ``SIGUSR1`` dump always works.
    """
    return _get_int_at_least("dump_traceback_interval", 0)


def get_extra_models(backend_name: str) -> list[str]:
    """Extra model IDs to offer for *backend_name*, from config.json.

    The built-in per-backend lists in ``backends_agent/`` are a curated
    snapshot and go stale as providers ship new models. This lets users
    expose newer or private/self-hosted models without editing source::

        {
            "extra_models": {
                "codex": ["my-private-codex-model"],
                "llama": ["my-local-model"],
            }
        }

    Malformed entries (missing key, non-object value, non-string items)
    are ignored, returning an empty list. Copilot's picker ignores extras
    because it requires the workspace's authenticated catalog.
    """
    val = _read_config_value("extra_models")
    if not isinstance(val, dict):
        return []
    return normalize_string_list(val.get(backend_name))


def get_model_context_window(
    backend_name: str, model: str | None,
) -> int | None:
    """Return an operator-configured context window for one backend/model.

    Compaction checks this operator override before adapter metadata, so it
    can also correct a provider-specific deployment. It covers private,
    auto-routed, and self-hosted models whose providers do not expose a
    numeric context capacity. Exact model IDs win over a backend-local
    ``"*"`` default::

        {
            "model_context_windows": {
                "llama": {"qwen3-coder": 32768, "auto": 32768},
                "zai": {"private-glm": 128000}
            }
        }

    Invalid values are ignored so malformed configuration falls back to the
    model adapter (and ultimately the message-interval safeguard).
    """
    windows = _read_config_value("model_context_windows")
    if not isinstance(windows, dict):
        return None
    per_backend = windows.get(backend_name)
    if not isinstance(per_backend, dict):
        return None
    keys = [model] if isinstance(model, str) and model else []
    keys.append("*")
    for key in keys:
        value = per_backend.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        try:
            _create_default_config()
        except FileExistsError:
            # Raced by another startup; continue validation.
            pass
        except (OSError, RuntimeError) as exc:
            print(f"ERROR: could not create a private config file: {exc}")
            sys.exit(1)
        else:
            print(f"Config file created at: {CONFIG_PATH}")
            print(
                "Fill in any combination of 'telegram_bot_tokens' +"
                " 'user_ids', 'slack_bot_token' + 'slack_app_token' +"
                " 'slack_channel_ids', and/or 'signal_group_urls' +"
                " 'signal_jsonrpc_socket', then restart."
            )
            sys.exit(0)

    try:
        _restrict_config_permissions(_config_dir(), 0o700)
        _restrict_config_permissions(CONFIG_PATH, 0o600)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    try:
        cfg = _load_config_object()
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, RecursionError) as e:
        print(f"ERROR: config.json is corrupted or unreadable: {e}")
        print(f"Fix or delete {CONFIG_PATH}, then restart.")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"Fix or delete {CONFIG_PATH}, then restart.")
        sys.exit(1)

    cfg = {**_DEFAULT_CONFIG, **cfg}

    raw_max_permission = cfg.get("max_permission")
    if isinstance(raw_max_permission, str):
        normalized_max_permission = raw_max_permission.strip()
    else:
        normalized_max_permission = ""
    if normalized_max_permission not in _PERMISSION_LEVELS:
        print(
            f"ERROR: 'max_permission' in {CONFIG_PATH} must be one of "
            "deny, confirm, auto, or full."
        )
        sys.exit(1)
    cfg["max_permission"] = normalized_max_permission

    # Constructor-consumed: normalize so bad JSON can't crash startup.
    for key in (
        "update_check_interval",
        "recent_workspace_limit",
        "message_queue_size",
        "max_upload_bytes",
    ):
        value = cfg.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            cfg[key] = _DEFAULT_CONFIG[key]

    # Filter blank tokens.
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
    signal_account = cfg.get("signal_account", "")
    if not isinstance(signal_account, str):
        print(f"ERROR: 'signal_account' in {CONFIG_PATH} must be a string.")
        sys.exit(1)
    cfg["signal_account"] = signal_account.strip()
    signal_device = cfg.get("signal_own_device_id")
    if signal_device is not None and (
        not isinstance(signal_device, int) or isinstance(signal_device, bool) or signal_device < 0
    ):
        print(f"ERROR: 'signal_own_device_id' in {CONFIG_PATH} must be a non-negative integer or null.")
        sys.exit(1)
    cfg["signal_own_device_id"] = signal_device or 0

    # Coerce IDs to strings.
    raw_user_ids = cfg.get("user_ids")
    if isinstance(raw_user_ids, (str, int)) and not isinstance(
        raw_user_ids, bool,
    ):
        raw_user_ids = [raw_user_ids]
    if isinstance(raw_user_ids, list):
        normalized_ids: list[str] = []
        for uid in raw_user_ids:
            if not isinstance(uid, (str, int)) or isinstance(uid, bool):
                continue
            text = str(uid).strip()
            if text:
                normalized_ids.append(text)
        cfg["user_ids"] = normalized_ids
    else:
        cfg["user_ids"] = []

    has_telegram = bool(cfg["telegram_bot_tokens"])
    has_slack = bool(cfg["slack_bot_token"])
    has_signal = bool(cfg["signal_group_urls"] or cfg["signal_jsonrpc_socket"] or cfg["signal_account"])

    configured = sum(bool(flag) for flag in (has_telegram, has_slack, has_signal))
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
                "ERROR: 'slack_app_token' (xapp-...) is required for"
                f" Socket Mode in {CONFIG_PATH}."
            )
            sys.exit(1)
        # Drop non-string/blank entries.
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
