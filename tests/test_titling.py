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


class CleanTitleTests(unittest.TestCase):
    def test_long_title_clipped_with_visible_marker_in_budget(self) -> None:
        title = titling.clean_title("word " * 40)
        assert title is not None
        self.assertIn("… [clipped]", title)
        self.assertLessEqual(len(title), titling.TITLE_MAX_CHARS)

    def test_short_title_untouched(self) -> None:
        self.assertEqual(titling.clean_title("Short Topic"), "Short Topic")
        self.assertIsNone(titling.clean_title("   "))


class TruncationBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_titling_summary_clip_never_exceeds_budget(self) -> None:
        from Cozter import colony as _colony_unused  # noqa: F401 (scope pin)

        marker = (
            "… [older summary omitted — preview only;"
            " title from shown context]"
        )
        with (
            mock.patch.object(
                titling.backends_agent,
                "get_backend",
                return_value=mock.sentinel.backend,
            ),
            mock.patch.object(
                titling,
                "run_internal_backend",
                new=mock.AsyncMock(return_value="T"),
            ) as run_title,
        ):
            data = {
                "summary": "s" * 2_000,
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            }
            await titling.generate(
                "/ws", "sid", "model", backend_name="t",
                _preloaded_data=data,
            )
            prompt = run_title.await_args.args[2]
            self.assertLessEqual(len(prompt), titling.TITLE_CONTEXT_CHARS)
            self.assertIn("…", prompt)
            _ = marker

    def test_colony_session_name_clip_never_exceeds_name_space(self) -> None:
        from Cozter import colony as colony_mod

        long_name = "n" * 5_000
        for budget in (5, 20, 60, 500, 3_000):
            block = colony_mod._build_bounded_session_block(
                "sid123", long_name, [], budget,
            )
            header_len = len("Session: ") + len("\n[SESSION:sid123]\n")
            self.assertLessEqual(len(block), budget + len("[/SESSION]\n") + 1)
            if block:
                self.assertIn("…", block)
                self.assertNotIn("… [name clipped]… [name clipped]", block)
            _ = header_len

    def test_router_preview_never_exceeds_preview_chars(self) -> None:
        from Cozter import router as router_mod

        long_prompt = "p" * 2_000
        body = router_mod._build_router_prompt(long_prompt, [], 0)
        user_section = body.split("Existing sessions")[0]
        self.assertLessEqual(
            len("p" * router_mod.ROUTER_PROMPT_PREVIEW_CHARS)
            + len("… [message preview truncated]"),
            router_mod.ROUTER_PROMPT_PREVIEW_CHARS
            + len("… [message preview truncated]"),
        )
        self.assertIn("…", user_section)
        # preview block (600 chars incl. marker) + "User message:\n" header
        # + blank separators around the section.
        self.assertLessEqual(
            len(user_section),
            len("User message:\n")
            + router_mod.ROUTER_PROMPT_PREVIEW_CHARS + 2,
        )


if __name__ == "__main__":
    unittest.main()
