import asyncio
import sys
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


if __name__ == "__main__":
    unittest.main()
