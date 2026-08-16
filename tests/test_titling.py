"""Regression coverage for session auto-titling."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import session, titling, workspace


class AutoTitlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_summary_keeps_title_prompt_bounded_and_recent(
        self,
    ) -> None:
        data = {
            "summary": "old context " + ("x" * titling.TITLE_CONTEXT_CHARS),
            "messages": [
                {"role": "user", "content": "recent user context"},
                {"role": "assistant", "content": "recent assistant context"},
            ],
        }

        with (
            mock.patch.object(
                titling.backends_agent,
                "get_backend",
                return_value=mock.sentinel.backend,
            ),
            mock.patch.object(
                titling,
                "run_internal_backend",
                new=mock.AsyncMock(return_value="Recent Context Title"),
            ) as run_title,
        ):
            title = await titling.generate(
                "/workspace", "session", "model", backend_name="test",
                _preloaded_data=data,
            )

        self.assertEqual(title, "Recent Context Title")
        prompt = run_title.await_args.args[2]
        self.assertLessEqual(len(prompt), titling.TITLE_CONTEXT_CHARS)
        self.assertIn("Recent messages:\nUser: recent user context", prompt)
        self.assertIn("Assistant: recent assistant context", prompt)

    async def test_stale_auto_title_does_not_overwrite_newer_session_name(
        self,
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_title(*_args, **_kwargs) -> str:
            started.set()
            await release.wait()
            return "Stale Auto Title"

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path)
            session.append_messages(workspace_path, data["id"], [{
                "role": "assistant", "content": "Initial reply",
            }])

            with mock.patch.object(titling, "generate", side_effect=delayed_title):
                task = asyncio.create_task(titling.maybe_auto_title(
                    workspace_path, data["id"], "model", backend_name="codex",
                ))
                await asyncio.wait_for(started.wait(), timeout=1)

                # This is the same write a compaction title (or a manual
                # rename) performs while the fallback title is in flight.
                async with workspace.get_lock(workspace_path):
                    session.set_session_name(
                        workspace_path, data["id"], "Compaction Title",
                    )

                release.set()
                await task

            latest = session.load_session(workspace_path, data["id"])
            assert latest is not None
            self.assertEqual(latest["name"], "Compaction Title")


if __name__ == "__main__":
    unittest.main()
