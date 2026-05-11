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


async def update_loop(bots: list[BotPlatform], interval: int) -> None:
    """Periodically fetch, pull, and restart if disk HEAD changed."""
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
                updater.restart_script()  # exits; systemd respawns
        except Exception:
            logger.exception("Update check failed")


def _cli_mode_requested() -> bool:
    """Return True if the user passed -cli / --cli on the command line."""
    flags = {"-cli", "--cli"}
    return any(arg in flags for arg in sys.argv[1:])


async def main_cli() -> None:
    """Run the local stdin/stdout chat surface.

    No config.json, no Telegram/Slack tokens, no update loop - just an
    interactive REPL backed by the same agent/session machinery.
    """
    from .backends_bot.cli import CliBot

    updater.init_startup_commit()
    logger.info(
        "Cozter CLI mode (version %s, %s)",
        updater.get_current_version(),
        updater.get_last_commit_date(),
    )

    bot = CliBot()
    await bot.start()
    try:
        await bot.wait_until_exit()
    finally:
        await bot.stop()


async def main() -> None:
    if _cli_mode_requested():
        await main_cli()
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
