import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest

from Cozter import utils


class StubBackend:
    @staticmethod
    def extract_agent_text(event: dict) -> str | None:
        return event.get("text")


class ProcessDrainTests(unittest.TestCase):
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

            text = await utils.drain_llm_subprocess(
                proc, StubBackend(), 5, "test",
            )

            self.assertEqual(text, "done")
            self.assertEqual(proc.returncode, 0)

        asyncio.run(run())


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


if __name__ == "__main__":
    unittest.main()
