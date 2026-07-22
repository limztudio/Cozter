import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import utils


class StubBackend:
    @staticmethod
    def extract_agent_text(event: dict) -> str | None:
        return event.get("text")


class CleanupStubBackend(StubBackend):
    def __init__(self) -> None:
        self.cleaned = False

    async def cleanup_process(self, _proc: asyncio.subprocess.Process) -> None:
        self.cleaned = True


class InternalBackendStub:
    executable = "internal-backend"

    def __init__(self) -> None:
        self.launch_args: tuple[object, ...] | None = None
        self.launch_kwargs: dict[str, object] | None = None

    async def launch(self, *args, **kwargs) -> object:
        self.launch_args = args
        self.launch_kwargs = kwargs
        return object()


class ProcessDrainTests(unittest.TestCase):
    def test_iter_json_events_discards_an_oversized_line(self) -> None:
        async def run() -> None:
            script = (
                "import json, sys\n"
                "sys.stdout.write('x' * 4096 + '\\n')\n"
                "print(json.dumps({'type': 'done'}))\n"
            )
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            with (
                mock.patch.object(utils, "_MAX_STREAM_LINE_BYTES", 1024),
                self.assertLogs(utils.logger, level="WARNING") as captured,
            ):
                events = [
                    event async for event in utils.iter_json_events(proc.stdout)
                ]
            await proc.wait()

            self.assertEqual(events, [{"type": "done"}])
            self.assertIn("Discarding backend stdout line", captured.output[0])

        asyncio.run(run())

    def test_iter_json_events_skips_invalid_lines(self) -> None:
        async def run() -> None:
            script = (
                "import json\n"
                "print('plain text')\n"
                "print(json.dumps({'type': 'message'}))\n"
                "print(json.dumps(['not', 'object']))\n"
                "print(json.dumps({'type': 'done'}))\n"
            )
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            invalid: list[str] = []
            events = [
                event async for event in utils.iter_json_events(
                    proc.stdout, on_invalid=invalid.append,
                )
            ]
            await proc.wait()

            self.assertEqual(events, [{"type": "message"}, {"type": "done"}])
            self.assertEqual(
                invalid, ["plain text", '["not", "object"]'],
            )

        asyncio.run(run())

    def test_drain_llm_subprocess_consumes_stderr_concurrently(self) -> None:
        async def run() -> None:
            script = (
                "import json, sys\n"
                "sys.stderr.buffer.write(b'x' * (2 * 1024 * 1024))\n"
                "sys.stderr.flush()\n"
                "print(json.dumps({'type': 'message', 'text': 'done'}))\n"
            )
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            backend = CleanupStubBackend()
            text = await utils.drain_llm_subprocess(proc, backend, 5, "test")

            self.assertEqual(text, "done")
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(backend.cleaned)

        asyncio.run(run())

    def test_drain_llm_subprocess_reports_stderr_when_output_is_empty(self) -> None:
        async def run() -> None:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import sys; print('backend diagnostic', file=sys.stderr)",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            log = logging.getLogger("Cozter.tests.empty_llm_output")

            with self.assertLogs(log, level="WARNING") as captured:
                text = await utils.drain_llm_subprocess(
                    proc,
                    StubBackend(),
                    5,
                    "test",
                    log=log,
                )

            self.assertEqual(text, "")
            self.assertTrue(
                any(
                    "backend diagnostic" in line
                    for line in captured.output
                ),
            )

        asyncio.run(run())


class ProcessTerminationTests(unittest.TestCase):
    def test_windows_termination_kills_the_process_tree(self) -> None:
        proc = mock.MagicMock()
        proc.pid = 2468
        completed = mock.Mock(returncode=0)

        with (
            mock.patch.object(utils.os, "name", "nt"),
            mock.patch.object(
                utils.subprocess, "run", return_value=completed,
            ) as taskkill,
        ):
            utils.terminate_process_group(proc)

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "2468", "/T", "/F"],
            stdout=utils.subprocess.DEVNULL,
            stderr=utils.subprocess.DEVNULL,
            timeout=2,
            check=False,
            creationflags=getattr(utils.subprocess, "CREATE_NO_WINDOW", 0),
        )
        proc.kill.assert_not_called()

    def test_windows_tree_kill_failure_falls_back_to_parent_kill(self) -> None:
        proc = mock.MagicMock()
        proc.pid = 2468

        with (
            mock.patch.object(utils.os, "name", "nt"),
            mock.patch.object(
                utils.subprocess,
                "run",
                return_value=mock.Mock(returncode=1),
            ),
        ):
            utils.terminate_process_group(proc)

        proc.kill.assert_called_once_with()

    def test_non_process_pid_never_calls_windows_taskkill(self) -> None:
        proc = mock.MagicMock()
        proc.pid = 0

        with (
            mock.patch.object(utils.os, "name", "nt"),
            mock.patch.object(utils.subprocess, "run") as taskkill,
        ):
            utils.terminate_process_group(proc)

        taskkill.assert_not_called()
        proc.kill.assert_called_once_with()


class JsonHelperTests(unittest.TestCase):
    def test_save_json_object_creates_parent_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "state.json")

            utils.save_json_object(path, {"ok": True})

            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"ok": True})

    def test_normalize_string_list_preserves_requested_semantics(self) -> None:
        self.assertEqual(
            utils.normalize_string_list([" a ", "", 3, "b"]),
            ["a", "b"],
        )
        self.assertEqual(
            utils.normalize_string_list(" a ", allow_scalar=True),
            ["a"],
        )
        self.assertEqual(
            utils.normalize_string_list([" a ", ""], strip=False),
            [" a "],
        )


class BackgroundTaskTests(unittest.TestCase):
    def test_create_background_task_logs_unhandled_exception(self) -> None:
        async def run() -> list[str]:
            log = logging.getLogger("Cozter.tests.background_task")

            async def fail() -> None:
                raise RuntimeError("boom")

            with self.assertLogs(log, level="ERROR") as captured:
                task = utils.create_background_task(
                    fail(), name="test-failure", log=log,
                )
                while not task.done():
                    await asyncio.sleep(0)
                await asyncio.sleep(0)
            return captured.output

        output = asyncio.run(run())
        self.assertTrue(
            any("Background task test-failure failed" in line for line in output),
        )
        self.assertTrue(any("RuntimeError: boom" in line for line in output))


class InternalBackendRunnerTests(unittest.TestCase):
    def test_internal_backend_uses_deny_instead_of_full(self) -> None:
        backend = InternalBackendStub()

        async def drain(*_args, **_kwargs) -> str:
            return "summary"

        with mock.patch.object(utils, "drain_llm_subprocess", new=drain):
            result = asyncio.run(utils.run_internal_backend(
                backend,
                "/work",
                "summarize this transcript",
                "model",
                timeout=1,
                label="internal test",
                log=logging.getLogger("Cozter.tests.internal"),
                missing_executable_message="%s missing",
            ))

        self.assertEqual(result, "summary")
        self.assertEqual(
            backend.launch_args,
            ("/work", "summarize this transcript", "model"),
        )
        self.assertEqual(
            backend.launch_kwargs,
            {"approval": "deny", "compaction": True},
        )

    def test_internal_backend_launch_failure_returns_empty_result(self) -> None:
        class BrokenBackend:
            executable = "internal-backend"

            async def launch(self, *args, **kwargs) -> object:
                del args, kwargs
                raise RuntimeError("stdin closed")

        log = logging.getLogger("Cozter.tests.internal-failure")
        with self.assertLogs(log, level="ERROR") as captured:
            result = asyncio.run(utils.run_internal_backend(
                BrokenBackend(),
                "/work",
                "summarize this transcript",
                "model",
                timeout=1,
                label="internal test",
                log=log,
                missing_executable_message="%s missing",
            ))

        self.assertEqual(result, "")
        self.assertIn("internal test could not start", captured.output[0])


if __name__ == "__main__":
    unittest.main()
