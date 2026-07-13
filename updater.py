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


def _head_changed() -> bool:
    """Return whether the checkout no longer matches the running process."""
    current = _get_head_commit()
    if current and _STARTUP_COMMIT and current != _STARTUP_COMMIT:
        logger.info("Code changed: %s -> %s", _STARTUP_COMMIT[:8], current[:8])
        return True
    return False


def _fetch_origin() -> bool:
    """Refresh remote refs without modifying the working tree."""
    try:
        fetch = _git("fetch", "origin")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("git not available or timed out, skipping update check")
        return False
    if fetch.returncode != 0:
        logger.warning("git fetch failed: %s", fetch.stderr.strip())
        return False
    return True


def _remote_update_available() -> bool:
    """Return whether the fetched upstream is ahead without pulling it."""
    try:
        if _working_tree_dirty():
            logger.debug(
                "Skipping auto-pull: working tree has uncommitted changes",
            )
            return False
        if _local_ahead_of_upstream():
            logger.debug(
                "Skipping auto-pull: local branch is ahead of or lacks a "
                "tracking upstream (development checkout)",
            )
            return False

        upstream_result = _git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
        )
        if upstream_result.returncode != 0:
            return False
        upstream = upstream_result.stdout.strip()
        if not upstream:
            return False
        behind_result = _git("rev-list", "--count", f"HEAD..{upstream}")
        if behind_result.returncode != 0:
            logger.warning(
                "Could not compare local HEAD with upstream: %s",
                behind_result.stderr.strip(),
            )
            return False
        try:
            return int(behind_result.stdout.strip()) > 0
        except ValueError:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("git not available or timed out, skipping update check")
        return False


def check_for_update() -> bool:
    """Fetch safely and report whether an update/restart is needed.

    Fetch only touches Git's remote refs, so normal message intake can stay
    live while this check runs. Pulling is deliberately deferred until the
    runtime has paused new turns and existing turns have finished.
    """
    fetched = _fetch_origin()
    if _head_changed():
        return True
    return fetched and _remote_update_available()


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
    if not _fetch_origin():
        return _head_changed()
    try:
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

    return _head_changed()


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

    Windows daemon launches always hand restart responsibility to a
    supervisor. ``run_cozter.ps1`` is the normal supervisor, and the venv
    bootstrap supplies one for a direct ``python -m Cozter`` launch. This
    avoids recursively spawning a Python child and retaining every prior
    Python process as a waiting parent after an update.

    CLI mode passes exit_code=99 so the in-process respawn loop in
    ``Cozter.__main__._cli_respawner_loop`` re-launches the bot.
    """
    logger.info("Restarting (exit code %d)...", exit_code)
    if exit_code == 0:
        if os.name == "nt":
            if os.environ.get(WINDOWS_SUPERVISOR_ENV) != "1":
                logger.warning(
                    "Windows restart requested without %s; exiting for "
                    "the bootstrap supervisor",
                    WINDOWS_SUPERVISOR_ENV,
                )
            # Avoid os.execv(), whose Windows process handoff is unreliable
            # in this runtime. The supervisor relaunches Cozter without
            # retaining this process as another Python ancestor.
            os._exit(WINDOWS_SUPERVISOR_RESTART_EXIT_CODE)
            return
        parent_dir = os.path.dirname(MODULE_ROOT)
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
