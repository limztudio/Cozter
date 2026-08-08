"""Regression tests for backend-process cleanup in the agent runtime."""

import asyncio
import os
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

from Cozter import agent, session, utils
from Cozter.backends_agent import base as backend_base
from Cozter.backends_agent.base import ChatEvent, append_detached_task
from Cozter.backends_bot.base import _InjectQueue
from Cozter.tests.helpers import create_python_script_process


class _StreamingBackend:
    name = "cleanup-test"
    executable = sys.executable
    supports_typed_plugins = True

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.cleaned = False

    async def launch(self, *_args, **_kwargs) -> asyncio.subprocess.Process:
        script = (
            "import json, time\n"
            "print(json.dumps({'type': 'message'}), flush=True)\n"
            "time.sleep(60)\n"
        )
        self.proc = await create_python_script_process(script)
        return self.proc

    async def cleanup_process(self, _proc: asyncio.subprocess.Process) -> None:
        self.cleaned = True

    @staticmethod
    def parse_event(_event: dict, result) -> None:
        result.events.append(ChatEvent(kind="text", content="streamed"))


class AgentProcessCleanupTests(unittest.TestCase):
    @staticmethod
    def _process_is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # A reparented child may briefly remain as a zombie after SIGKILL;
        # it cannot keep a pipe open or mutate a workspace.
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                return f.read().split()[2] != "Z"
        except OSError:
            # macOS and other POSIX systems may not mount Linux's /proc;
            # a successful kill(0) still means the process is live there.
            return True

    def test_captured_subprocess_closes_stdin_and_captures_output(self) -> None:
        async def run() -> None:
            script = (
                "import sys\n"
                "print('stdin-closed=' + str(sys.stdin.read() == ''))\n"
                "print('stderr-output', file=sys.stderr)\n"
            )
            proc = await backend_base.create_captured_subprocess(
                [sys.executable, "-c", script],
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            self.assertEqual(stdout.decode(), "stdin-closed=True\n")
            self.assertEqual(stderr.decode(), "stderr-output\n")

        asyncio.run(run())

    def test_backend_launch_failure_becomes_a_user_facing_result(self) -> None:
        class BrokenBackend:
            name = "broken"
            executable = "broken-cli"

            async def launch(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError("stdin closed during startup")

        result, restarting = asyncio.run(agent._drive_backend(
            BrokenBackend(), "/work", "prompt", None, "auto", effort=0,
        ))

        self.assertFalse(restarting)
        self.assertIn("broken could not start", result.text)
        self.assertIn("stdin closed during startup", result.text)

    def test_launch_failure_honors_pending_and_terminal_injects(self) -> None:
        class BrokenBackend:
            name = "broken"
            executable = "broken-cli"

            async def launch(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError("stdin closed during startup")

        class DelayedBrokenBackend(BrokenBackend):
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def launch(self, *args, **kwargs):
                del args, kwargs
                self.started.set()
                await self.release.wait()
                raise RuntimeError("stdin closed during startup")

        async def run() -> None:
            pending_queue = _InjectQueue(maxsize=2)
            pending_backend = DelayedBrokenBackend()
            injected: list[str] = []
            drive = asyncio.create_task(agent._drive_backend(
                pending_backend, "/work", "prompt", None, "auto", effort=0,
                inject_queue=pending_queue,
                injected=injected,
                close_inject_on_completion=True,
            ))
            await asyncio.wait_for(pending_backend.started.wait(), timeout=1)
            self.assertEqual(
                pending_queue.put_if_active("also check the tests"),
                "accepted",
            )
            pending_backend.release.set()
            _result, restarting = await asyncio.wait_for(drive, timeout=1)

            self.assertTrue(restarting)
            self.assertEqual(injected, ["also check the tests"])
            # A restart owns the same queue, so another inject remains valid.
            self.assertEqual(pending_queue.put_if_active("one more"), "accepted")

            terminal_queue = _InjectQueue(maxsize=1)
            _result, restarting = await agent._drive_backend(
                BrokenBackend(), "/work", "prompt", None, "auto", effort=0,
                inject_queue=terminal_queue,
                injected=[],
                close_inject_on_completion=True,
            )

            self.assertFalse(restarting)
            self.assertEqual(
                terminal_queue.put_if_active("too late"), "finished",
            )

        asyncio.run(run())

    def test_closed_stdin_prompt_delivery_reaps_process(self) -> None:
        async def run() -> None:
            created: list[asyncio.subprocess.Process] = []
            real_create = asyncio.create_subprocess_exec

            async def capture_process(*args, **kwargs):
                proc = await real_create(*args, **kwargs)
                created.append(proc)
                return proc

            script = "import os, time; os.close(0); time.sleep(60)"
            with mock.patch.object(
                backend_base.asyncio,
                "create_subprocess_exec",
                side_effect=capture_process,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "closed stdin before Cozter could deliver the prompt",
                ):
                    await asyncio.wait_for(
                        backend_base.create_prompt_subprocess(
                            [sys.executable, "-c", script],
                            "x" * (1024 * 1024),
                        ),
                        timeout=5,
                    )

            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].returncode)

        asyncio.run(run())

    def test_failed_prompt_reaper_closes_unread_pipes(self) -> None:
        """A failed prompt path owns no readers, so it must close both pipes."""
        async def run() -> None:
            proc = mock.Mock()
            proc.returncode = None
            with (
                mock.patch.object(
                    backend_base, "kill_and_wait", new_callable=mock.AsyncMock,
                ) as kill_and_wait,
                mock.patch.object(backend_base, "close_subprocess_pipe") as close,
            ):
                await backend_base._reap_failed_prompt_subprocess(proc)

            kill_and_wait.assert_awaited_once_with(proc)
            self.assertEqual(
                close.call_args_list,
                [mock.call(proc, 1), mock.call(proc, 2)],
            )

        asyncio.run(run())

    def test_event_callback_failure_reaps_backend_process(self) -> None:
        async def run() -> None:
            backend = _StreamingBackend()

            async def fail_callback(_event: ChatEvent) -> None:
                raise RuntimeError("delivery failed")

            with tempfile.TemporaryDirectory() as tmp:
                data = session.create_session(tmp)
                with mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ):
                    with self.assertRaisesRegex(RuntimeError, "delivery failed"):
                        await agent.run(
                            "hello",
                            tmp,
                            1,
                            on_event=fail_callback,
                            backend_name=backend.name,
                            session_id=data["id"],
                        )

            assert backend.proc is not None
            self.assertTrue(backend.cleaned)
            try:
                self.assertIsNotNone(backend.proc.returncode)
            finally:
                if backend.proc.returncode is None:
                    backend.proc.kill()
                    await backend.proc.wait()

        asyncio.run(run())

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_exited_parent_with_inherited_pipes_is_reaped(self) -> None:
        """A child retaining stdout/stderr must not strand the agent turn."""
        async def run() -> tuple[object, bool, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "child.pid")

                class ExitedParentBackend:
                    name = "exited-parent"

                    async def launch(self, *_args, **_kwargs):
                        script = (
                            "import json, subprocess, sys\n"
                            "child = subprocess.Popen(\n"
                            "    [sys.executable, '-c', "
                            "'import time; time.sleep(30)']\n"
                            ")\n"
                            f"open({pid_path!r}, 'w').write(str(child.pid))\n"
                            "print(json.dumps({'type': 'message'}), flush=True)\n"
                        )
                        return await backend_base.create_captured_subprocess(
                            [sys.executable, "-c", script],
                            start_new_session=True,
                        )

                    @staticmethod
                    def parse_event(_event: dict, result) -> None:
                        result.events.append(ChatEvent(kind="text", content="ok"))

                completed = False
                try:
                    result, restarting = await asyncio.wait_for(
                        agent._drive_backend(
                            ExitedParentBackend(), tmp, "prompt", None,
                            "auto", effort=0,
                        ),
                        timeout=2,
                    )
                    with open(pid_path, encoding="utf-8") as f:
                        child_pid = int(f.read())
                    completed = True
                    return result, restarting, child_pid
                finally:
                    # The assertions below should normally see this child
                    # already dead. Keep the regression itself leak-safe if a
                    # later change makes the test fail midway through.
                    if not completed and os.path.exists(pid_path):
                        with open(pid_path, encoding="utf-8") as f:
                            child_pid = int(f.read())
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

        _result, restarting, child_pid = asyncio.run(run())
        try:
            self.assertFalse(restarting)

            deadline = time.monotonic() + 2
            while (
                self._process_is_running(child_pid)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertFalse(
                self._process_is_running(child_pid),
                f"child process {child_pid} survived its exited parent",
            )
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_exited_parent_json_flood_is_bounded_and_reaped(self) -> None:
        """A child cannot keep a completed turn alive with valid JSON."""
        async def run() -> tuple[bool, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "flood-child.pid")
                child_script = (
                    "import json\n"
                    "while True:\n"
                    "    print(json.dumps({'type': 'message'}), flush=True)\n"
                )

                class FloodBackend:
                    name = "json-flood"

                    async def launch(self, *_args, **_kwargs):
                        script = (
                            "import json, subprocess, sys\n"
                            "child = subprocess.Popen(\n"
                            f"    [sys.executable, '-c', {child_script!r}]\n"
                            ")\n"
                            f"open({pid_path!r}, 'w').write(str(child.pid))\n"
                            "print(json.dumps({'type': 'message'}), flush=True)\n"
                        )
                        return await backend_base.create_captured_subprocess(
                            [sys.executable, "-c", script],
                            start_new_session=True,
                        )

                    @staticmethod
                    def parse_event(_event: dict, _result) -> None:
                        # Deliberately retain nothing: this isolates stream
                        # termination from any backend-specific event limits.
                        return

                completed = False
                try:
                    _result, restarting = await asyncio.wait_for(
                        agent._drive_backend(
                            FloodBackend(), tmp, "prompt", None,
                            "auto", effort=0,
                        ),
                        timeout=2,
                    )
                    with open(pid_path, encoding="utf-8") as f:
                        child_pid = int(f.read())
                    completed = True
                    return restarting, child_pid
                finally:
                    if not completed and os.path.exists(pid_path):
                        with open(pid_path, encoding="utf-8") as f:
                            child_pid = int(f.read())
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

        restarting, child_pid = asyncio.run(run())
        try:
            self.assertFalse(restarting)
            deadline = time.monotonic() + 2
            while (
                self._process_is_running(child_pid)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertFalse(
                self._process_is_running(child_pid),
                f"JSON-flood child {child_pid} survived stream cleanup",
            )
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_exited_parent_keeps_finite_buffered_events(self) -> None:
        """Slow event delivery must not discard an ordinary final backlog."""
        async def run() -> list[str]:
            class BufferedBackend:
                name = "buffered-events"

                async def launch(self, *_args, **_kwargs):
                    script = (
                        "import json\n"
                        "for number in range(10):\n"
                        "    print(json.dumps({'type': 'message', "
                        "'number': number}), flush=True)\n"
                    )
                    return await backend_base.create_captured_subprocess(
                        [sys.executable, "-c", script],
                        start_new_session=os.name != "nt",
                    )

                @staticmethod
                def parse_event(event: dict, result) -> None:
                    result.events.append(ChatEvent(
                        kind="text", content=str(event["number"]),
                    ))

            async def slow_delivery(_event: ChatEvent) -> None:
                # The parent exits during this sequence, leaving the rest of
                # its legitimate JSON lines buffered in stdout.
                await asyncio.sleep(0.12)

            result, restarting = await asyncio.wait_for(
                agent._drive_backend(
                    BufferedBackend(), "/work", "prompt", None, "auto",
                    effort=0, on_event=slow_delivery,
                ),
                timeout=3,
            )
            self.assertFalse(restarting)
            return [event.content for event in result.events]

        self.assertEqual(
            asyncio.run(run()), [str(number) for number in range(10)],
        )

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_recognized_detached_task_with_inherited_pipes_is_preserved(
        self,
    ) -> None:
        """A recognized ``claude --bg`` child survives bounded pipe cleanup."""
        async def run() -> tuple[object, bool, int, float]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "detached-child.pid")

                class DetachedTaskBackend:
                    name = "claude"

                    async def launch(self, *_args, **_kwargs):
                        script = (
                            "import json, subprocess, sys\n"
                            "child = subprocess.Popen(\n"
                            "    [sys.executable, '-c', "
                            "'import time; time.sleep(30)']\n"
                            ")\n"
                            f"open({pid_path!r}, 'w').write(str(child.pid))\n"
                            "print(json.dumps({'type': 'detached'}), flush=True)\n"
                        )
                        return await backend_base.create_captured_subprocess(
                            [sys.executable, "-c", script],
                            start_new_session=True,
                        )

                    @staticmethod
                    def parse_event(event: dict, result) -> None:
                        if event.get("type") == "detached":
                            append_detached_task(result, "claude", "task-123")

                completed = False
                try:
                    started = time.monotonic()
                    result, restarting = await asyncio.wait_for(
                        agent._drive_backend(
                            DetachedTaskBackend(), tmp, "prompt", None,
                            "auto", effort=0,
                        ),
                        timeout=3.5,
                    )
                    with open(pid_path, encoding="utf-8") as f:
                        child_pid = int(f.read())
                    completed = True
                    return (
                        result,
                        restarting,
                        child_pid,
                        time.monotonic() - started,
                    )
                finally:
                    if not completed and os.path.exists(pid_path):
                        with open(pid_path, encoding="utf-8") as f:
                            child_pid = int(f.read())
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

        result, restarting, child_pid, elapsed = asyncio.run(run())
        try:
            self.assertFalse(restarting)
            self.assertEqual(
                [(task.backend_name, task.task_id) for task in result.detached_tasks],
                [("claude", "task-123")],
            )
            self.assertLess(elapsed, 3.5)
            self.assertTrue(
                self._process_is_running(child_pid),
                "recognized detached task was killed during stream cleanup",
            )
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_exited_parent_with_closed_pipes_is_reaped(self) -> None:
        """A same-group child cannot evade teardown by closing its pipes."""
        async def run() -> tuple[bool, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "child.pid")

                class ClosedPipeChildBackend:
                    name = "closed-pipe-child"

                    async def launch(self, *_args, **_kwargs):
                        script = (
                            "import json, subprocess, sys\n"
                            "child = subprocess.Popen(\n"
                            "    [sys.executable, '-c', "
                            "'import time; time.sleep(30)'], "
                            "stdout=subprocess.DEVNULL, "
                            "stderr=subprocess.DEVNULL\n"
                            ")\n"
                            f"open({pid_path!r}, 'w').write(str(child.pid))\n"
                            "print(json.dumps({'type': 'message'}), flush=True)\n"
                        )
                        return await backend_base.create_captured_subprocess(
                            [sys.executable, "-c", script],
                            start_new_session=True,
                        )

                    @staticmethod
                    def parse_event(_event: dict, result) -> None:
                        result.events.append(ChatEvent(kind="text", content="ok"))

                completed = False
                try:
                    _result, restarting = await asyncio.wait_for(
                        agent._drive_backend(
                            ClosedPipeChildBackend(), tmp, "prompt", None,
                            "auto", effort=0,
                        ),
                        timeout=2,
                    )
                    with open(pid_path, encoding="utf-8") as f:
                        child_pid = int(f.read())
                    completed = True
                    return restarting, child_pid
                finally:
                    if not completed and os.path.exists(pid_path):
                        with open(pid_path, encoding="utf-8") as f:
                            child_pid = int(f.read())
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

        restarting, child_pid = asyncio.run(run())
        try:
            self.assertFalse(restarting)

            deadline = time.monotonic() + 2
            while (
                self._process_is_running(child_pid)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertFalse(
                self._process_is_running(child_pid),
                f"closed-pipe child {child_pid} survived normal cleanup",
            )
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_internal_drain_reaps_exited_parent_with_inherited_pipes(self) -> None:
        async def run() -> tuple[str, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "child.pid")
                script = (
                    "import json, subprocess, sys\n"
                    "child = subprocess.Popen(\n"
                    "    [sys.executable, '-c', "
                    "'import time; time.sleep(30)'], "
                    "stdout=subprocess.DEVNULL\n"
                    ")\n"
                    f"open({pid_path!r}, 'w').write(str(child.pid))\n"
                    "print(json.dumps({'text': 'summary'}), flush=True)\n"
                )
                proc = await backend_base.create_captured_subprocess(
                    [sys.executable, "-c", script],
                    start_new_session=True,
                )

                class Backend:
                    @staticmethod
                    def extract_agent_text(event: dict) -> str:
                        return event.get("text", "")

                completed = False
                try:
                    text = await asyncio.wait_for(
                        utils.drain_llm_subprocess(
                            proc, Backend(), 10, "internal test",
                        ),
                        timeout=2,
                    )
                    with open(pid_path, encoding="utf-8") as f:
                        child_pid = int(f.read())
                    completed = True
                    return text, child_pid
                finally:
                    if not completed and os.path.exists(pid_path):
                        with open(pid_path, encoding="utf-8") as f:
                            child_pid = int(f.read())
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

        text, child_pid = asyncio.run(run())
        try:
            self.assertEqual(text, "summary")

            deadline = time.monotonic() + 2
            while (
                self._process_is_running(child_pid)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertFalse(
                self._process_is_running(child_pid),
                f"child process {child_pid} survived internal drain cleanup",
            )
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @unittest.skipIf(os.name == "nt", "POSIX setsid behavior")
    def test_detached_stderr_child_cannot_hang_turn_cleanup(self) -> None:
        """A descendant outside the owned group must not hold /stop forever."""
        async def run() -> tuple[float, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "detached-child.pid")

                class DetachedStderrBackend:
                    name = "detached-stderr"

                    async def launch(self, *_args, **_kwargs):
                        script = (
                            "import json, os, subprocess, sys\n"
                            "child = subprocess.Popen(\n"
                            "    [sys.executable, '-c', "
                            "'import time; time.sleep(4)'], "
                            "stdout=subprocess.DEVNULL, preexec_fn=os.setsid\n"
                            ")\n"
                            f"open({pid_path!r}, 'w').write(str(child.pid))\n"
                            "print(json.dumps({'type': 'message'}), flush=True)\n"
                        )
                        return await backend_base.create_captured_subprocess(
                            [sys.executable, "-c", script],
                            start_new_session=True,
                        )

                    @staticmethod
                    def parse_event(_event: dict, result) -> None:
                        result.events.append(ChatEvent(kind="text", content="ok"))

                completed = False
                try:
                    started = time.monotonic()
                    await asyncio.wait_for(
                        agent._drive_backend(
                            DetachedStderrBackend(), tmp, "prompt", None,
                            "auto", effort=0,
                        ),
                        timeout=2.5,
                    )
                    with open(pid_path, encoding="utf-8") as f:
                        child_pid = int(f.read())
                    completed = True
                    return time.monotonic() - started, child_pid
                finally:
                    if not completed and os.path.exists(pid_path):
                        with open(pid_path, encoding="utf-8") as f:
                            child_pid = int(f.read())
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

        elapsed, child_pid = asyncio.run(run())
        try:
            self.assertLess(elapsed, 2.5)
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    unittest.main()
