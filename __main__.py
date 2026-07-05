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


# Make ``Cozter`` importable from any subprocess we spawn (codex,
# claude_code, copilot CLIs) and from any bash command they run. The
# CLI subprocesses inherit our env, so when the model invokes a plugin
# via ``python -m Cozter.agent_tools.plugins.<name>``, Python can
# resolve the package without the user setting PYTHONPATH manually.
_pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _pkg_parent not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        f"{_pkg_parent}{os.pathsep}{_existing_pythonpath}"
        if _existing_pythonpath else _pkg_parent
    )


_VENV_REEXEC_ENV = "COZTER_VENV_REEXEC"


def _running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _venv_python() -> str:
    venv_dir = os.path.join(os.path.dirname(__file__), ".venv")
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(venv_dir, bin_dir, exe)


def _ensure_venv_and_reexec() -> None:
    """Run daemon mode through a project-local venv on managed Python."""
    if _running_in_venv() or os.environ.get(_VENV_REEXEC_ENV) == "1":
        return

    python = _venv_python()
    if not os.path.exists(python):
        venv_dir = os.path.dirname(os.path.dirname(python))
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    env = {**os.environ, _VENV_REEXEC_ENV: "1"}
    os.execve(python, [python, "-m", "Cozter", *sys.argv[1:]], env)


_ensure_venv_and_reexec()


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
import threading  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from logging.handlers import RotatingFileHandler  # noqa: E402

from . import config as cfg  # noqa: E402
from . import updater  # noqa: E402
from .backends_bot import BotPlatform, create_platforms  # noqa: E402
from .utils import await_cancelled  # noqa: E402

MODULE_ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(MODULE_ROOT, ".log")

# Sleep before respawning after an unhandled exception. Long enough
# that a tight crash loop logs at human-readable speed; short enough
# that recovery feels prompt to a watching user.
_CRASH_RESTART_DELAY_SEC = 5
_UPDATE_IDLE_POLL_SEC = 1
_UPDATE_IDLE_LOG_SEC = 30


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


# File object that faulthandler and the SIGUSR1 dump write into. Opened
# lazily so tests can import this module without a writable .log/ dir,
# and so the path reflects the configured LOG_DIR even if it is created
# after import time.
_dump_file = None


def _get_dump_file():
    """Open (once) and return the diagnostics dump file handle.

    Writes diagnostics into ``.log/diagnostics.log`` so a SIGUSR1 dump
    or a faulthandler crash capture lands somewhere greppable alongside
    the main log, rather than only to the original stderr of the
    daemon (which systemd may rotate away).
    """
    global _dump_file
    if _dump_file is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        _dump_file = open(
            os.path.join(LOG_DIR, "diagnostics.log"), "a", encoding="utf-8",
        )
    return _dump_file


def dump_runtime_diagnostics(
    bots: list[BotPlatform] | None = None, *, reason: str = "",
) -> None:
    """Dump asyncio tasks, active threads, and turn state to the log.

    Triggered on-demand by SIGUSR1, and reusable from the stuck-wait
    critical path. Uses ``Task.print_stack`` (stable across Python
    versions, unlike ``asyncio.dump_traceback``) and renders into the
    diagnostics file plus the logger so it shows up regardless of how
    the daemon was launched.

    *bots* is optional so this can be called before platforms exist or
    from contexts that only want the task/thread dump.
    """
    label = f" ({reason})" if reason else ""
    f = _get_dump_file()
    f.write(f"\n===== diagnostics dump{label} @ {datetime.now().isoformat()} =====\n")
    f.flush()

    # asyncio tasks: each Task carries its own stack; print_stack writes
    # to *f*. Capture names too for a quick at-a-glance summary.
    try:
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        f.write(
            f"-- asyncio tasks ({len(tasks)} pending) --\n"
        )
        for t in tasks:
            try:
                t.print_stack(file=f)
            except Exception as exc:  # pragma: no cover - defensive
                f.write(f"  (failed to print task {t}: {exc})\n")
    except Exception as exc:  # pragma: no cover - defensive
        f.write(f"  (failed to enumerate tasks: {exc})\n")
    f.flush()

    # Native threads (codex/zai subprocess readers, signal threads, etc.).
    threads = threading.enumerate()
    frames = sys._current_frames()
    f.write(f"-- active threads ({len(threads)}) --\n")
    for thread in threads:
        f.write(
            f"\n--- thread {thread.name} "
            f"(ident={thread.ident}, daemon={thread.daemon}) ---\n"
        )
        if thread.ident is None:
            f.write("  (thread has no ident)\n")
            continue
        frame = frames.get(thread.ident)
        if frame is None:
            f.write("  (no Python frame available)\n")
            continue
        f.writelines(traceback.format_stack(frame))
    f.flush()

    # Per-platform turn-tracking state, when platforms are available.
    if bots:
        f.write("-- bot turn state --\n")
        for bot in bots:
            try:
                has = bot.has_active_turns()
                diag = bot.stuck_turn_diagnostics() if has else "<idle>"
            except Exception as exc:  # pragma: no cover - defensive
                has, diag = None, f"<error: {exc}>"
            f.write(
                f"  {_bot_label(bot)}: has_active_turns={has} {diag}\n"
            )
        f.flush()

    logger.info("Runtime diagnostics dumped to diagnostics.log%s", label)


def _enable_faulthandler() -> None:
    """Enable faulthandler so faults dump all threads to the log.

    Catches segfaults (SIGSEGV etc.) and, via dump_traceback_later, a
    wedged main thread that holds the GIL without making progress.
    Writes into the diagnostics file so output survives systemd log
    rotation. Kept best-effort: if faulthandler is unavailable or the
    file isn't writable yet, the daemon still runs.
    """
    try:
        import faulthandler
    except ImportError:  # pragma: no cover - stdlib on all supported Py
        return
    try:
        faulthandler.enable(file=_get_dump_file(), all_threads=True)
    except Exception:
        logger.exception("Failed to enable faulthandler")

    interval = cfg.get_dump_traceback_interval()
    if interval > 0:
        try:
            # repeat=True keeps it firing every ``interval`` seconds for
            # the life of the process; a one-shot dump is rarely enough
            # to catch a transient wedge.
            faulthandler.dump_traceback_later(
                interval, repeat=True, file=_get_dump_file(),
            )
        except Exception:
            logger.exception("Failed to enable periodic traceback dump")


def _bot_label(bot: BotPlatform) -> str:
    try:
        return bot.platform_id
    except Exception:
        return bot.__class__.__name__


async def _wait_for_update_idle(
    bots: list[BotPlatform], *, log_message: str,
) -> None:
    """Wait until no platform is actively producing an agent reply.

    Has a hard ceiling (``config.get_update_idle_timeout``): no turn
    runs longer than ``turn_timeout`` (it is cancelled at that bound),
    so if the wait blows past the ceiling the turn-tracking state
    itself is wedged — a leaked task or an un-released lock — and we
    force-break out with a critical log + per-bot diagnostics rather
    than hanging the daemon on ``Delaying update check`` forever.
    """
    last_log = 0.0
    deadline = loop_start = asyncio.get_running_loop().time()
    deadline += cfg.get_update_idle_timeout()
    while True:
        active = [bot for bot in bots if bot.has_active_turns()]
        if not active:
            return
        now = asyncio.get_running_loop().time()
        if now >= deadline:
            waited = int(now - loop_start)
            logger.critical(
                "Update idle wait exceeded ceiling of %ds after ~%ds; "
                "force-breaking. Stuck turn-tracking state indicates a "
                "leaked task or un-released lock. Diagnostics: %s",
                cfg.get_update_idle_timeout(), waited,
                {bot.platform_id: bot.stuck_turn_diagnostics() for bot in active},
            )
            # Full task/thread dump so the force-break leaves the
            # evidence needed to root-cause the wedged state, not just
            # the summary above.
            dump_runtime_diagnostics(active, reason="update-idle-force-break")
            return
        if now - last_log >= _UPDATE_IDLE_LOG_SEC:
            logger.info(
                "%s: %s",
                log_message,
                ", ".join(_bot_label(bot) for bot in active),
            )
            last_log = now
        await asyncio.sleep(_UPDATE_IDLE_POLL_SEC)


async def _restart_after_update(
    bots: list[BotPlatform], restart_code: int,
) -> None:
    """Pause intake, wait for active replies, then restart for an update."""
    prepared: list[BotPlatform] = []
    try:
        for bot in bots:
            await bot.begin_update_restart()
            prepared.append(bot)
        await _wait_for_update_idle(
            bots,
            log_message=(
                "Update ready; waiting for active turn(s) before restart"
            ),
        )
        await asyncio.to_thread(updater.install_requirements)
    except Exception:
        for bot in prepared:
            try:
                await bot.cancel_update_restart()
            except Exception:
                logger.exception(
                    "Failed to resume bot after aborted update restart",
                )
        raise

    logger.info("Active turns finished; restarting for update...")
    for bot in bots:
        try:
            await bot.notify_users("Cozter is restarting for an update...")
        except Exception:
            logger.exception("Failed to send update restart notification")
    for bot in bots:
        try:
            await bot.stop()
        except Exception:
            logger.exception("Failed to stop bot before update restart")
    updater.restart_script(restart_code)


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
        prepared: list[BotPlatform] = []
        try:
            if any(bot.has_active_turns() for bot in bots):
                await _wait_for_update_idle(
                    bots,
                    log_message=(
                        "Delaying update check until active turn(s) finish"
                    ),
                )
            for bot in bots:
                await bot.begin_update_check()
                prepared.append(bot)
            await _wait_for_update_idle(
                bots,
                log_message=(
                    "Delaying update check until active turn(s) finish"
                ),
            )
            changed = await asyncio.to_thread(updater.fetch_and_pull)
            if changed:
                logger.info(
                    "New version detected; restart pending after active turns",
                )
                await _restart_after_update(bots, restart_code)
            else:
                for bot in prepared:
                    await bot.cancel_update_check()
        except Exception:
            for bot in prepared:
                try:
                    await bot.cancel_update_check()
                except Exception:
                    logger.exception(
                        "Failed to resume bot after aborted update check",
                    )
            logger.exception("Update check failed")


def _cli_mode_requested() -> bool:
    """Return True if the user passed -cli / --cli on the command line."""
    flags = {"-cli", "--cli"}
    return any(arg in flags for arg in sys.argv[1:])


# Two-phase CLI lifecycle, signalled via one env var:
#   - launcher (no env): user ran ``python -m Cozter -cli`` from a
#     shell. Run the respawner loop in *this* terminal.
#   - bot (COZTER_CLI_CHILD=1): the loop spawned us; run main_cli with
#     update_loop attached. On auto-update, exit with
#     ``CLI_RESTART_EXIT_CODE`` so the loop relaunches us in place.
_CLI_CHILD_ENV = "COZTER_CLI_CHILD"


def _cli_child_mode() -> bool:
    return os.environ.get(_CLI_CHILD_ENV) == "1"


def _cli_respawner_loop() -> int:
    """Spawn the bot subprocess in a loop; relaunch on exit-99.

    Runs in the user's current terminal. Inherits stdin/stdout/stderr
    so the bot has direct access to the terminal. Returns the bot's
    final non-restart exit code.
    """
    bot_env = {**os.environ, _CLI_CHILD_ENV: "1"}
    cmd = [sys.executable, "-m", "Cozter", *sys.argv[1:]]
    while True:
        rc = subprocess.call(cmd, env=bot_env)
        if rc != updater.CLI_RESTART_EXIT_CODE:
            return rc


async def main_cli() -> None:
    """Run the local stdin/stdout chat surface in the current terminal.

    Always attaches the update_loop and exits with
    ``CLI_RESTART_EXIT_CODE`` when new code is pulled so the outer
    respawner relaunches the bot in place.
    """
    from .backends_bot.cli import CliBot

    # CLI mode has no config to read - reading one would create a
    # spurious config.json with daemon-only fields on first run and
    # print misleading "fill in your tokens" messages. Match the
    # daemon's default interval directly.
    cli_interval = 10

    updater.init_startup_commit()
    logger.info(
        "Cozter CLI mode (version %s, %s)",
        updater.get_current_version(),
        updater.get_last_commit_date(),
    )

    bot = CliBot()
    await bot.start()
    update_task = asyncio.create_task(update_loop(
        [bot], cli_interval,
        restart_code=updater.CLI_RESTART_EXIT_CODE,
    ))
    try:
        await bot.wait_until_exit()
    finally:
        update_task.cancel()
        await await_cancelled(update_task)
        await bot.stop()


async def main() -> None:
    if _cli_mode_requested():
        # Two phases: this process is either the launcher (run the
        # respawner loop in the current terminal) or the bot child
        # spawned by that loop.
        if _cli_child_mode():
            await main_cli()
            return
        # Sync subprocess loop; exit cleanly through asyncio.
        sys.exit(_cli_respawner_loop())

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

    # SIGUSR1 dumps asyncio tasks + threads + per-bot turn state into
    # diagnostics.log on demand, without restarting. From the host:
    #   kill -USR1 $(systemctl show -p MainPID --value app-Cozter@autostart)
    # It's Windows-incompatible (add_signal_handler raises), so guard it.
    if hasattr(signal, "SIGUSR1"):
        try:
            loop.add_signal_handler(
                signal.SIGUSR1,
                lambda: dump_runtime_diagnostics(bots, reason="SIGUSR1"),
            )
        except (NotImplementedError, ValueError):
            # NotImplementedError on Windows; ValueError if the signal
            # can't be registered in this context. Neither is fatal.
            pass

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
    await await_cancelled(update_task)
    for bot in bots:
        await bot.notify_users("Cozter is shutting down.")
        await bot.stop()


def run() -> None:
    setup_logging()
    # faulthandler before anything else: if the daemon segfaults or
    # wedges in C, this is the only thing that emits where it stuck.
    _enable_faulthandler()
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
            " Restarting in %ds...",
            crash_path, _CRASH_RESTART_DELAY_SEC,
        )
        time.sleep(_CRASH_RESTART_DELAY_SEC)
        updater.restart_script()


if __name__ == "__main__":
    run()
