import os
import subprocess
import sys
import time

# Re-launch as module if run as `python Cozter` (no package context).
# Forward any CLI args (e.g. -cli) so flags survive the re-exec.
if __name__ == "__main__" and not __package__:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_name = os.path.basename(module_dir)
    parent_dir = os.path.dirname(module_dir)
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", module_name, *sys.argv[1:]],
            cwd=parent_dir,
        )
    )


def _install_deps() -> None:
    """Auto-install missing requirements before any project imports."""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
        )


_install_deps()

import asyncio  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from logging.handlers import RotatingFileHandler  # noqa: E402

from . import config as cfg  # noqa: E402
from . import updater  # noqa: E402
from .backends_bot import BotPlatform, create_platforms  # noqa: E402

MODULE_ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(MODULE_ROOT, ".log")


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, "cozter.log")
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def log_crash(exc: BaseException) -> str:
    """Write a timestamped crash report to .log/ and return the file path."""
    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now()
    crash_file = os.path.join(
        LOG_DIR, f"crash_{now.strftime('%Y%m%d_%H%M%S')}.log"
    )
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    with open(crash_file, "w", encoding="utf-8") as f:
        f.write(f"Crash at {now.isoformat()}\n")
        f.write(f"Exception: {type(exc).__name__}: {exc}\n\n")
        f.writelines(tb)
    return crash_file


logger = logging.getLogger(__name__)


async def update_loop(
    bots: list[BotPlatform], interval: int, restart_code: int = 0,
) -> None:
    """Periodically fetch, pull, and restart if disk HEAD changed.

    restart_code is the exit code used when an update is found. The
    daemon uses 0 (systemd's Restart=always respawns); CLI mode uses
    99 so the in-process respawn loop picks it up.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            changed = await asyncio.to_thread(updater.fetch_and_pull)
            if changed:
                logger.info("New version detected, restarting...")
                await asyncio.to_thread(updater.install_requirements)
                for bot in bots:
                    await bot.notify_users(
                        "Cozter is restarting for an update..."
                    )
                    await bot.stop()
                updater.restart_script(restart_code)
        except Exception:
            logger.exception("Update check failed")


def _cli_mode_requested() -> bool:
    """Return True if the user passed -cli / --cli on the command line."""
    flags = {"-cli", "--cli"}
    return any(arg in flags for arg in sys.argv[1:])


# Three-phase CLI lifecycle, signalled via env vars:
#   - launcher (no env): user invoked the script from a shell. Open a
#     new console window running the respawner.
#   - respawner (COZTER_CLI_RESPAWNER=1): we're inside the new console.
#     Repeatedly invoke the bot subprocess; restart it on exit code
#     ``CLI_RESTART_EXIT_CODE`` (auto-update path). Any other exit code
#     ends the loop so a real crash stays visible.
#   - bot (COZTER_CLI_CHILD=1): run main_cli with update_loop attached.
_CLI_CHILD_ENV = "COZTER_CLI_CHILD"
_CLI_RESPAWNER_ENV = "COZTER_CLI_RESPAWNER"


def _cli_child_mode() -> bool:
    return os.environ.get(_CLI_CHILD_ENV) == "1"


def _cli_respawner_mode() -> bool:
    return os.environ.get(_CLI_RESPAWNER_ENV) == "1"


def _cli_respawner_loop() -> int:
    """Spawn the bot subprocess in a loop; relaunch on exit-99.

    Runs in the new console window. Inherits stdin/stdout/stderr so the
    bot has direct access to the user's terminal. Returns the bot's
    final non-restart exit code.
    """
    bot_env = {**os.environ, _CLI_CHILD_ENV: "1"}
    # Don't leak the respawner marker into the bot process - it would
    # only confuse mode detection if anything ever inspected it.
    bot_env.pop(_CLI_RESPAWNER_ENV, None)
    cmd = [sys.executable, "-m", "Cozter", *sys.argv[1:]]
    while True:
        rc = subprocess.call(cmd, env=bot_env)
        if rc != updater.CLI_RESTART_EXIT_CODE:
            return rc


def _windows_native_path() -> str:
    """Return PATH as Windows itself would set it for a fresh cmd window.

    When the launcher is started from a bash-like shell (msys2, Git Bash,
    WSL interop), ``os.environ['PATH']`` is in Unix format (``/c/...``)
    that ``cmd.exe`` cannot resolve - so a freshly spawned cmd window
    can't find ``codex.exe`` etc even though the user can run them in
    their normal terminal. We re-read PATH from the registry (system +
    user) so the spawned console has the same PATH it would have if the
    user opened ``cmd.exe`` from the Start menu.
    """
    import winreg
    parts: list[str] = []
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as k:
            value, _ = winreg.QueryValueEx(k, "Path")
            if value:
                parts.append(value)
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
            value, _ = winreg.QueryValueEx(k, "Path")
            if value:
                parts.append(value)
    except OSError:
        pass
    return ";".join(parts)


def _spawn_cli_in_new_console() -> bool:
    """Spawn the script again in a fresh OS console window.

    Returns True if a child was successfully started, False if no
    suitable mechanism is available on this host. The caller should
    fall back to running the CLI bot in the current terminal if False.
    """
    import platform
    import shlex
    import shutil as _shutil

    py = sys.executable
    inner_args = [py, "-m", "Cozter", *sys.argv[1:]]
    cwd = os.getcwd()
    # Launch into respawner mode so the new console can auto-restart
    # the bot when an update is pulled.
    env = {**os.environ, _CLI_RESPAWNER_ENV: "1"}
    env.pop(_CLI_CHILD_ENV, None)
    system = platform.system()

    if system == "Windows":
        # Replace the inherited (possibly bash-mangled) PATH with what
        # the registry says PATH should be, so the new cmd window sees
        # the same tools the user gets from a stock Start-menu cmd.
        native_path = _windows_native_path()
        if native_path:
            env["PATH"] = native_path
        # CREATE_NEW_CONSOLE attaches the child to a fresh console
        # window. Wrapping with ``cmd /k`` keeps that window open after
        # Python exits so the user can read any final output (including
        # a crash trace) before closing manually.
        try:
            cmdline = "cmd.exe /k " + subprocess.list2cmdline(inner_args)
            subprocess.Popen(
                cmdline,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=cwd, env=env,
            )
            return True
        except Exception:
            logger.exception("Failed to spawn Windows CLI console")
            return False

    # POSIX: try common terminal emulators in priority order. The
    # trailing ``echo; read`` keeps the window open after Python exits
    # so crash traces are visible.
    inner_str = " ".join(shlex.quote(a) for a in inner_args)
    inner_str += "; echo; echo '(press Enter to close)'; read"

    if system == "Darwin":
        script = (
            f'tell application "Terminal" to do script '
            f'"cd {shlex.quote(cwd)} && {inner_str}"'
        )
        if _shutil.which("osascript"):
            try:
                subprocess.Popen(
                    ["osascript", "-e", script], env=env,
                )
                return True
            except Exception:
                logger.exception("Failed to spawn macOS Terminal")
                return False
        return False

    candidates = [
        ["gnome-terminal", "--", "bash", "-c", inner_str],
        ["konsole", "-e", "bash", "-c", inner_str],
        ["xfce4-terminal", "-e", "bash -c " + shlex.quote(inner_str)],
        ["alacritty", "-e", "bash", "-c", inner_str],
        ["kitty", "bash", "-c", inner_str],
        ["xterm", "-e", "bash -c " + shlex.quote(inner_str)],
    ]
    for cmd in candidates:
        if _shutil.which(cmd[0]):
            try:
                subprocess.Popen(cmd, cwd=cwd, env=env)
                return True
            except Exception:
                logger.exception(
                    "Failed to spawn terminal via %s", cmd[0],
                )
                continue
    return False


async def main_cli(*, auto_update: bool = True) -> None:
    """Run the local stdin/stdout chat surface.

    auto_update=True attaches the update_loop and exits with
    ``CLI_RESTART_EXIT_CODE`` when new code is pulled so the outer
    respawner relaunches the bot. Disabled in the fallback path where
    no respawner is wrapping us - an unhandled exit would just kill
    the visible session.
    """
    from .backends_bot.cli import CliBot

    # CLI auto-update uses the same interval as daemon mode when a
    # config.json happens to be present; otherwise fall back to the
    # built-in default so the user gets updates without setup.
    try:
        cli_interval = cfg.load_config().get("update_check_interval", 10)
    except SystemExit:
        # load_config sys.exits when neither platform is configured;
        # the CLI doesn't need it, just use the default.
        cli_interval = 10

    updater.init_startup_commit()
    logger.info(
        "Cozter CLI mode (version %s, %s)",
        updater.get_current_version(),
        updater.get_last_commit_date(),
    )

    bot = CliBot()
    await bot.start()
    update_task: asyncio.Task | None = None
    if auto_update:
        update_task = asyncio.create_task(update_loop(
            [bot], cli_interval,
            restart_code=updater.CLI_RESTART_EXIT_CODE,
        ))
    try:
        await bot.wait_until_exit()
    finally:
        if update_task is not None:
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass
        await bot.stop()


async def main() -> None:
    if _cli_mode_requested():
        # Dispatch by lifecycle phase (see _cli_respawner_mode docstring).
        if _cli_child_mode():
            await main_cli()
            return
        if _cli_respawner_mode():
            # Sync subprocess loop; not actually async work, just exit
            # cleanly through asyncio.
            sys.exit(_cli_respawner_loop())
        if _spawn_cli_in_new_console():
            return
        logger.warning(
            "Could not open a new console window; running CLI in the"
            " current terminal instead (no auto-update).",
        )
        await main_cli(auto_update=False)
        return

    config = cfg.load_config()
    interval = config["update_check_interval"]

    updater.init_startup_commit()
    version = updater.get_current_version()
    commit_date = updater.get_last_commit_date()

    bots = create_platforms(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _signal_handler())

    for bot in bots:
        await bot.start()

    # Each platform decides how to greet its notify targets on startup.
    for bot in bots:
        try:
            await bot.send_startup_messages(version, commit_date)
        except Exception:
            logger.exception("Failed to send startup message via bot")

    logger.info(
        "Version: %s | Updated: %s | Bots: %d",
        version, commit_date, len(bots),
    )

    update_task = asyncio.create_task(update_loop(bots, interval))

    await stop_event.wait()

    logger.info("Shutting down...")
    update_task.cancel()
    try:
        await update_task
    except asyncio.CancelledError:
        pass
    for bot in bots:
        await bot.notify_users("Cozter is shutting down.")
        await bot.stop()


def run() -> None:
    setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        crash_path = log_crash(exc)
        if _cli_mode_requested():
            # CLI mode is interactive; auto-restarting would be jarring.
            logger.critical(
                "Cozter crashed - crash log written to %s", crash_path,
            )
            return
        logger.critical(
            "Unhandled exception - crash log written to %s."
            " Restarting in 5s...",
            crash_path,
        )
        time.sleep(5)
        updater.restart_script()


if __name__ == "__main__":
    run()
