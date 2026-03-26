import logging
import subprocess
import sys
import os

logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.dirname(__file__)
REQUIREMENTS_PATH = os.path.join(MODULE_ROOT, "requirements.txt")


def get_current_version() -> str:
    """Return the short hash of the current HEAD commit."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip()


def get_last_commit_date() -> str:
    """Return the date of the latest commit."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ci"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip()


def has_remote_update() -> bool:
    """Fetch origin and check if local HEAD is behind."""
    fetch = subprocess.run(
        ["git", "fetch", "origin"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    if fetch.returncode != 0:
        logger.warning("git fetch failed: %s", fetch.stderr.strip())
        return False

    result = subprocess.run(
        ["git", "status", "-uno"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    return "Your branch is behind" in result.stdout


def pull_and_update() -> None:
    """Pull latest changes and install updated requirements."""
    logger.info("Pulling latest changes...")
    pull = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=MODULE_ROOT, capture_output=True, text=True,
    )
    if pull.returncode != 0:
        logger.error("git pull failed: %s", pull.stderr.strip())
        raise RuntimeError("git pull failed")
    logger.info(pull.stdout.strip())

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
    """Restart the current module by spawning a new process and exiting."""
    logger.info("Restarting...")
    module_name = os.path.basename(MODULE_ROOT)
    parent_dir = os.path.dirname(MODULE_ROOT)
    subprocess.Popen(
        [sys.executable, "-m", module_name],
        cwd=parent_dir,
    )
    sys.exit(0)
