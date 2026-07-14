"""Regression tests for backend-process cleanup in the agent runtime."""

import asyncio
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import agent, session
from Cozter.backends_agent.base import ChatEvent


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
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return self.proc

    async def cleanup_process(self, _proc: asyncio.subprocess.Process) -> None:
        self.cleaned = True

    @staticmethod
    def parse_event(_event: dict, result) -> None:
        result.events.append(ChatEvent(kind="text", content="streamed"))


class AgentProcessCleanupTests(unittest.TestCase):
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
