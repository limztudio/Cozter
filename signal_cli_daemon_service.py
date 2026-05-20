"""Systemd entrypoint for the shared signal-cli JSON-RPC daemon.

This module is intentionally separate from the bot backend: systemd owns
the daemon lifecycle, and Cozter or any other local script only connects
to the configured socket.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("/home/shared/utilities/SignalDaemon/config.json")


def main() -> None:
    config = _load_config()
    phone_number = _required_config_string(
        config, "phone_number", "signal_phone_number"
    )
    socket_path = _required_config_string(
        config, "socket_path", "signal_jsonrpc_socket"
    )
    signal_cli = _optional_config_string(config, "signal_cli_path") or os.environ.get(
        "SIGNAL_CLI_PATH", "signal-cli"
    )
    executable = shutil.which(signal_cli)
    if executable is None:
        raise SystemExit(f"signal-cli executable not found: {signal_cli}")

    _prepare_socket_path(socket_path)
    os.execv(
        executable,
        [
            executable,
            "-a",
            phone_number,
            "daemon",
            "--socket",
            socket_path,
            "--receive-mode",
            "manual",
            "--ignore-stories",
            "--no-receive-stdout",
        ],
    )


def _load_config() -> dict[str, Any]:
    config_path = Path(
        os.environ.get(
            "SIGNAL_DAEMON_CONFIG_PATH",
            DEFAULT_CONFIG_PATH,
        )
    )
    try:
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} must contain a JSON object")
    return config


def _required_config_string(config: dict[str, Any], *keys: str) -> str:
    value = next(
        (
            config.get(key)
            for key in keys
            if isinstance(config.get(key), str) and config.get(key).strip()
        ),
        "",
    )
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"Missing required config value: {keys[0]}")
    return os.path.expandvars(os.path.expanduser(value.strip()))


def _optional_config_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        return ""
    return os.path.expandvars(os.path.expanduser(value.strip()))


def _prepare_socket_path(socket_path: str) -> None:
    socket_file = Path(socket_path)
    socket_file.parent.mkdir(parents=True, exist_ok=True)
    if not socket_file.exists():
        return
    mode = socket_file.stat().st_mode
    if not stat.S_ISSOCK(mode):
        raise SystemExit(f"Socket path exists but is not a socket: {socket_path}")
    if _socket_accepts_connections(socket_path):
        raise SystemExit(f"Signal daemon socket is already active: {socket_path}")
    socket_file.unlink()


def _socket_accepts_connections(socket_path: str) -> bool:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(1.0)
        client.connect(socket_path)
    except OSError:
        return False
    finally:
        client.close()
    return True


if __name__ == "__main__":
    main()
