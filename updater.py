import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.dirname(__file__)
REQUIREMENTS_PATH = os.path.join(MODULE_ROOT, "requirements.txt")

# Commit hash captured at import time - the version this process is running.
_STARTUP_COMMIT: str | None = None


def _get_head_commit() -> str:
    """Return the full hash of the current HEAD commit on disk."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip()


def init_startup_commit() -> None:
    """Snapshot the commit hash this process started with."""
    global _STARTUP_COMMIT
    _STARTUP_COMMIT = _get_head_commit()
    logger.info("Startup commit: %s", _STARTUP_COMMIT)


def get_current_version() -> str:
    """Return the short hash of the current HEAD commit."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip() or "(unknown)"


def get_last_commit_date() -> str:
    """Return the date of the latest commit."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ci"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip() or "(unknown)"


def fetch_and_pull() -> bool:
    """Fetch origin, pull if behind; return True if disk HEAD changed.

    This detects both remote updates and manual pulls that happened while
    the process was running.
    """
    # Fetch latest refs and pull
    try:
        fetch = subprocess.run(
            ["git", "fetch", "origin"],
            cwd=MODULE_ROOT, capture_output=True, text=True,
        )
        if fetch.returncode != 0:
            logger.warning("git fetch failed: %s", fetch.stderr.strip())

        pull = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=MODULE_ROOT, capture_output=True, text=True,
        )
        if pull.returncode != 0:
            logger.warning("git pull failed: %s", pull.stderr.strip())
    except FileNotFoundError:
        logger.error("git not found on PATH, skipping update check")
        return False

    # Compare disk HEAD to the commit we started with
    current = _get_head_commit()
    if current and _STARTUP_COMMIT and current != _STARTUP_COMMIT:
        logger.info("Code changed: %s -> %s", _STARTUP_COMMIT[:8], current[:8])
        return True
    return False


def install_requirements() -> None:
    """Install updated requirements if the file exists."""
    if os.path.exists(REQUIREMENTS_PATH):
        logger.info("Installing updated requirements...")
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH],
            cwd=MODULE_ROOT, capture_output=True, text=True,
        )
        if pip.returncode != 0:
            logger.error("pip install failed: %s", pip.stderr.strip())
            raise RuntimeError("pip install failed")
        logger.info("Requirements installed.")


def restart_script() -> None:
    """Exit so the systemd service (Restart=always) restarts us."""
    logger.info("Restarting (exiting for systemd to respawn)...")
    os._exit(0)
