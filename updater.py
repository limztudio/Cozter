import contextlib
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.dirname(__file__)
REQUIREMENTS_PATH = os.path.join(MODULE_ROOT, "requirements.txt")

# Commit hash captured at import time - the version this process is running.
_STARTUP_COMMIT: str | None = None


_GIT_TIMEOUT = 30  # seconds — prevents hung network calls from blocking forever

# A Windows supervisor (such as run_cozter.ps1) treats this as a normal
# self-update restart.  It is non-zero so Task Scheduler recovery can also
# restart a directly launched task if configured to do so.
WINDOWS_SUPERVISOR_RESTART_EXIT_CODE = 75
WINDOWS_SUPERVISOR_ENV = "COZTER_WINDOWS_SUPERVISED"


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command with a timeout, returning the CompletedProcess."""
    return subprocess.run(
        ["git", *args],
        cwd=MODULE_ROOT, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )


def _git_str(*args: str, default: str = "") -> str:
    """Run a git command and return stripped stdout; `default` if unavailable."""
    try:
        return _git(*args).stdout.strip() or default
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default


def _get_head_commit() -> str:
    """Return the full hash of the current HEAD commit on disk, or ''."""
    return _git_str("rev-parse", "HEAD")


def _working_tree_dirty() -> bool:
    """True if the checkout has uncommitted (non-ignored) changes.

    Guards the auto-pull: when Cozter manages its own repo and someone is
    editing it (a developer, or an agent turn writing files), a
    ``git pull`` would fight that work. Any git error - lock contention,
    not-a-repo - is treated as dirty so we never mutate an indeterminate
    state.
    """
    try:
        result = _git("status", "--porcelain")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _local_ahead_of_upstream() -> bool:
    """True when automatic pulling is unsafe for the current checkout.

    A checkout with unpushed local commits is being developed on, not
    just deployed - auto-pulling it is at best a no-op and at worst fights
    the developer, so skip the pull. A branch without a tracking upstream
    is also not a deploy target, so skip it instead of issuing a bare
    ``git pull`` that fails on every update interval.
    """
    try:
        upstream_result = _git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    # A non-zero rev-parse normally means this branch has no configured
    # upstream. Do not issue a bare ``git pull`` in that state: it cannot
    # choose a branch and only produces repeated error logs.
    if upstream_result.returncode != 0:
        return True
    upstream = upstream_result.stdout.strip()
    if not upstream:
        return True
    try:
        count_result = _git(
            "rev-list", "--count", f"{upstream}..HEAD",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    # Once an upstream resolved, failure to compare it with HEAD is an
    # indeterminate state. Treat it as locally ahead so auto-update never
    # mutates a checkout whose safety could not be established.
    if count_result.returncode != 0:
        return True
    count = count_result.stdout.strip()
    try:
        return int(count) > 0
    except ValueError:
        return True


def init_startup_commit() -> None:
    """Snapshot the commit hash this process started with."""
    global _STARTUP_COMMIT
    _STARTUP_COMMIT = _get_head_commit()
    logger.info("Startup commit: %s", _STARTUP_COMMIT)


def get_current_version() -> str:
    """Return the short hash of the current HEAD commit."""
    return _git_str("rev-parse", "--short", "HEAD", default="(unknown)")


def get_last_commit_date() -> str:
    """Return the date of the latest commit."""
    return _git_str("log", "-1", "--format=%ci", default="(unknown)")


def fetch_and_pull() -> bool:
    """Fetch origin, pull if behind; return True if disk HEAD changed.

    This detects both remote updates and manual pulls that happened while
    the process was running.

    The auto-pull is skipped when the working tree is dirty or the local
    branch is ahead of its upstream - i.e. the checkout is being developed
    on rather than merely deployed. HEAD-change detection still runs in
    that case, so a developer's own commit or manual pull is picked up and
    a restart is scheduled as before.
    """
    try:
        fetch = _git("fetch", "origin")
        if fetch.returncode != 0:
            logger.warning("git fetch failed: %s", fetch.stderr.strip())

        if _working_tree_dirty():
            logger.debug(
                "Skipping auto-pull: working tree has uncommitted changes",
            )
        elif _local_ahead_of_upstream():
            logger.debug(
                "Skipping auto-pull: local branch is ahead of or lacks a "
                "tracking upstream (development checkout)",
            )
        else:
            pull = _git("pull", "--ff-only")
            if pull.returncode != 0:
                logger.warning("git pull failed: %s", pull.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("git not available or timed out, skipping update check")
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


def restart_script(exit_code: int = 0) -> None:
    """Restart the current process.

    Daemon/autostart mode re-execs this process directly. XDG autostart
    units are generated with ``Restart=no``, so exiting would leave the
    bot stopped after it announces an update restart.

    CLI mode passes exit_code=99 so the in-process respawn loop in
    ``Cozter.__main__._cli_respawner_loop`` re-launches the bot.
    """
    logger.info("Restarting (exit code %d)...", exit_code)
    if exit_code == 0:
        parent_dir = os.path.dirname(MODULE_ROOT)
        if os.name == "nt" and os.environ.get(WINDOWS_SUPERVISOR_ENV) == "1":
            # Windows Task Scheduler should supervise the next process.
            # Avoid os.execv(), whose Windows process handoff is unreliable
            # in this runtime, and let the supervisor relaunch Cozter.
            os._exit(WINDOWS_SUPERVISOR_RESTART_EXIT_CODE)
            return
        if os.name == "nt":
            # Preserve restart behavior for direct/manual Windows launches
            # until they are moved behind run_cozter.ps1.  The parent waits
            # for the replacement so it remains visible to its supervisor.
            rc = subprocess.call(
                [sys.executable, "-m", "Cozter", *sys.argv[1:]],
                cwd=parent_dir,
            )
            os._exit(rc)
            return
        os.chdir(parent_dir)
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        with contextlib.suppress(Exception):
            sys.stderr.flush()
        os.execv(
            sys.executable,
            [sys.executable, "-m", "Cozter", *sys.argv[1:]],
        )
    os._exit(exit_code)


# Exit code signalling "please respawn me" to the CLI mode respawn loop.
# Any other non-zero exit code stops the loop (so crash tracebacks stay
# visible instead of getting overwritten by restart spam).
CLI_RESTART_EXIT_CODE = 99
