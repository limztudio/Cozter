import os
import subprocess
import sys

# Re-launch as module if run as `python Cozter` (no package context)
if __name__ == "__main__" and not __package__:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_name = os.path.basename(module_dir)
    parent_dir = os.path.dirname(module_dir)
    sys.exit(subprocess.call([sys.executable, "-m", module_name], cwd=parent_dir))

def _install_deps() -> None:
    """Auto-install missing requirements before any project imports."""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
        )

_install_deps()

import asyncio
import logging
import signal
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler

from . import config as cfg
from . import updater
from . import workspace
from .bot import CozterBot

MODULE_ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(MODULE_ROOT, ".log")


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, "cozter.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_file = os.path.join(LOG_DIR, f"crash_{timestamp}.log")
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    with open(crash_file, "w", encoding="utf-8") as f:
        f.write(f"Crash at {datetime.now().isoformat()}\n")
        f.write(f"Exception: {type(exc).__name__}: {exc}\n\n")
        f.writelines(tb)
    return crash_file


logger = logging.getLogger(__name__)


async def update_loop(bots: list[CozterBot], interval: int) -> None:
    """Periodically fetch, pull, and restart if code on disk differs from startup."""
    while True:
        await asyncio.sleep(interval)
        try:
            changed = await asyncio.to_thread(updater.fetch_and_pull)
            if changed:
                logger.info("New version detected, restarting...")
                await asyncio.to_thread(updater.install_requirements)
                for bot in bots:
                    await bot.notify_users("Cozter is restarting for an update...")
                    await bot.stop()
                updater.restart_script()  # exits; systemd respawns
        except Exception:
            logger.exception("Update check failed")


async def main() -> None:
    config = cfg.load_config()
    tokens = config["telegram_bot_tokens"]
    user_ids = config["user_ids"]
    interval = config["update_check_interval"]
    recent_limit = config["recent_workspace_limit"]

    updater.init_startup_commit()
    version = updater.get_current_version()
    commit_date = updater.get_last_commit_date()

    bots = [CozterBot(token, user_ids, recent_limit=recent_limit) for token in tokens]

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

    # Send startup message through every bot
    for uid in user_ids:
        ws = workspace.get_current(uid)
        msg = f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
        if ws:
            msg += f"\nWorkspace: {ws}"
        else:
            msg += "\nNo workspace selected. Use /new or /open."
        for bot in bots:
            try:
                await bot.app.bot.send_message(chat_id=uid, text=msg)
            except Exception as e:
                logger.warning("Failed to notify user %s via bot: %s", uid, e)

    logger.info("Version: %s | Updated: %s | Bots: %d", version, commit_date, len(bots))

    update_task = asyncio.create_task(update_loop(bots, interval))

    await stop_event.wait()

    logger.info("Shutting down...")
    update_task.cancel()
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
        logger.critical(
            "Unhandled exception — crash log written to %s. Restarting in 5s...",
            crash_path,
        )
        import time
        time.sleep(5)
        updater.restart_script()


if __name__ == "__main__":
    run()
