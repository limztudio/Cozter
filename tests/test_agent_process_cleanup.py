"""Regression tests for backend-process cleanup in the agent runtime."""

import asyncio
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import agent, session
from Cozter.backends_agent import base as backend_base
from Cozter.backends_agent.base import ChatEvent
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


if __name__ == "__main__":
    unittest.main()
