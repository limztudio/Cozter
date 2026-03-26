import os
import subprocess
import sys

# Re-launch as module if run as `python Cozter` (no package context)
if __name__ == "__main__" and not __package__:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_name = os.path.basename(module_dir)
    parent_dir = os.path.dirname(module_dir)
    sys.exit(subprocess.call([sys.executable, "-m", module_name], cwd=parent_dir))

import asyncio
import logging
import signal
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler

from . import auth
from . import config as cfg
from . import updater
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
    file_handler.setLevel(logging.ERROR)
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


async def update_loop(bot: CozterBot, interval: int) -> None:
    """Periodically check for git updates and restart if needed."""
    while True:
        await asyncio.sleep(interval)
        try:
            if updater.has_remote_update():
                logger.info("New version detected, updating...")
                await bot.notify_users("Cozter is shutting down for an update...")
                await bot.stop()
                updater.pull_and_update()
                updater.restart_script()  # replaces process, does not return
        except Exception:
            logger.exception("Update check failed")


def ensure_login() -> None:
    """Check .config/secret for saved tokens; open browser login if needed."""
    if auth.is_logged_in():
        tokens = auth.get_tokens()
        logger.info("Logged in as %s (plan: %s)", tokens.get("email"), tokens.get("plan"))
        refreshed = auth.refresh_if_needed()
        if refreshed:
            return
        logger.warning("Token refresh failed — re-login required.")
        auth.clear_auth()

    while True:
        try:
            saved = auth.browser_login()
            logger.info("Logged in as %s (plan: %s)", saved.get("email"), saved.get("plan"))
            return
        except Exception as e:
            logger.error("Login failed: %s", e)
            print(f"\nLogin failed: {e}")
            print("Retrying in 30 seconds... (Ctrl+C to quit)\n")
            import time
            time.sleep(30)


async def main() -> None:
    config = cfg.load_config()
    token = config["telegram_bot_token"]
    user_ids = config["user_ids"]
    interval = config["update_check_interval"]
    recent_limit = config["recent_workspace_limit"]

    ensure_login()

    version = updater.get_current_version()
    commit_date = updater.get_last_commit_date()

    bot = CozterBot(token, user_ids, recent_limit=recent_limit)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _signal_handler())

    await bot.start()
    await bot.notify_users(
        f"Cozter started.\nVersion: {version}\nUpdated: {commit_date}"
    )
    logger.info("Version: %s | Updated: %s", version, commit_date)

    update_task = asyncio.create_task(update_loop(bot, interval))

    await stop_event.wait()

    logger.info("Shutting down...")
    update_task.cancel()
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
