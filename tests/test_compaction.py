"""Concurrency coverage for automatic session compaction."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from Cozter import colony, compaction, session, workspace


class CompactionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_compaction_title_does_not_overwrite_manual_rename(
        self,
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_compact(*_args, **_kwargs):
            started.set()
            await release.wait()
            return ("x" * 100, [], "Stale Compaction Title", 6)

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path)
            session.append_messages(workspace_path, data["id"], [
                {"role": "user", "content": f"message-{i}"}
                for i in range(6)
            ])
            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=delayed_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                task = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.wait_for(started.wait(), timeout=1)

                async with workspace.get_lock(workspace_path):
                    session.set_session_name(
                        workspace_path, data["id"], "Manual Session Name",
                    )

                release.set()
                await task

            latest = session.load_session(workspace_path, data["id"])
            assert latest is not None
            self.assertEqual(latest["name"], "Manual Session Name")

    async def test_same_session_compacts_only_once_at_a_time(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def slow_compact(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return ("x" * 100, [], None, 6)

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user", "content": "hello",
            }])
            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=slow_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                first = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.wait_for(started.wait(), timeout=1)
                await compaction.maybe_compact(
                    os.path.join(workspace_path, "."),
                    data["id"], "model", backend_name="backend",
                )
                self.assertEqual(calls, 1)

                release.set()
                await first

                # The guard is released after the first compaction finishes.
                await compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                )
                self.assertEqual(calls, 2)

    async def test_colony_waits_for_a_running_compaction(self) -> None:
        compaction_started = asyncio.Event()
        release_compaction = asyncio.Event()
        colony_started = asyncio.Event()

        async def slow_compact(*_args, **_kwargs):
            compaction_started.set()
            await release_compaction.wait()
            return ("x" * 100, [], None, 6)

        async def fake_consolidate(*_args, **_kwargs) -> bool:
            colony_started.set()
            return False

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user", "content": "hello",
            }])
            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=slow_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
                mock.patch.object(
                    colony, "_consolidate_inner", side_effect=fake_consolidate,
                ),
            ):
                compact_task = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.wait_for(compaction_started.wait(), timeout=1)
                colony_task = asyncio.create_task(colony.consolidate(
                    workspace_path, "model", backend_name="backend",
                ))
                await asyncio.sleep(0)
                self.assertFalse(colony_started.is_set())

                release_compaction.set()
                await compact_task
                await asyncio.wait_for(colony_started.wait(), timeout=1)
                await colony_task

    async def test_compaction_waits_for_a_running_colony_pass(self) -> None:
        colony_started = asyncio.Event()
        release_colony = asyncio.Event()
        compaction_started = asyncio.Event()
        release_compaction = asyncio.Event()

        async def slow_consolidate(*_args, **_kwargs) -> bool:
            colony_started.set()
            await release_colony.wait()
            return False

        async def slow_compact(*_args, **_kwargs):
            compaction_started.set()
            await release_compaction.wait()
            return ("x" * 100, [], None, 6)

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user", "content": "hello",
            }])
            with (
                mock.patch.object(
                    colony, "_consolidate_inner", side_effect=slow_consolidate,
                ),
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=slow_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                colony_task = asyncio.create_task(colony.consolidate(
                    workspace_path, "model", backend_name="backend",
                ))
                await asyncio.wait_for(colony_started.wait(), timeout=1)
                compact_task = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.sleep(0)
                self.assertFalse(compaction_started.is_set())

                release_colony.set()
                await colony_task
                await asyncio.wait_for(compaction_started.wait(), timeout=1)
                release_compaction.set()
                await compact_task

    async def test_known_context_window_compacts_before_message_interval(
        self,
    ) -> None:
        """A known capacity drives compaction even below /compact's fallback."""
        async def fake_compact(*_args, **_kwargs):
            return ("s" * 100, [], None, 6)

        backend = SimpleNamespace(
            context_window_tokens=lambda _model: 500,
        )
        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [
                {"role": "user", "content": "x" * 120}
                for _ in range(6)
            ])
            with (
                mock.patch.object(
                    compaction.config,
                    "get_model_context_window",
                    return_value=None,
                ),
                mock.patch.object(
                    compaction.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(
                    compaction.workspace_mod,
                    "get_compact_interval",
                    return_value=99,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=fake_compact,
                ) as compact,
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                await compaction.maybe_compact(
                    workspace_path,
                    data["id"],
                    "summary-model",
                    backend_name="summary",
                    context_targets=(("known", "model"),),
                )

            compact.assert_awaited_once()
            saved = session.load_session(workspace_path, data["id"])
            assert saved is not None
            self.assertEqual(len(saved["messages"]), 5)

    async def test_unknown_context_window_keeps_message_interval_fallback(
        self,
    ) -> None:
        backend = SimpleNamespace(context_window_tokens=lambda _model: None)
        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [
                {"role": "user", "content": "x" * 120}
                for _ in range(6)
            ])
            with (
                mock.patch.object(
                    compaction.config,
                    "get_model_context_window",
                    return_value=None,
                ),
                mock.patch.object(
                    compaction.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(
                    compaction.workspace_mod,
                    "get_compact_interval",
                    return_value=7,
                ),
                mock.patch.object(compaction, "compact_session") as compact,
            ):
                await compaction.maybe_compact(
                    workspace_path,
                    data["id"],
                    "summary-model",
                    backend_name="summary",
                    context_targets=(("unknown", "model"),),
                )

            compact.assert_not_awaited()

    async def test_context_window_uses_backend_default_for_implicit_model(
        self,
    ) -> None:
        backend = SimpleNamespace(
            default_model="default-model",
            context_window_tokens=lambda _model: None,
        )
        with (
            mock.patch.object(
                compaction.backends_agent, "get_backend", return_value=backend,
            ),
            mock.patch.object(
                compaction.config,
                "get_model_context_window",
                return_value=12_345,
            ) as configured,
        ):
            window = compaction._context_window_tokens((("known", None),))

        self.assertEqual(window, 12_345)
        configured.assert_called_once_with("known", "default-model")

    async def test_known_context_window_also_respects_history_budget(self) -> None:
        """/context remains a guard before raw history would be truncated."""
        backend = SimpleNamespace(
            context_window_tokens=lambda _model: 1_000_000,
        )
        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [
                {"role": "user", "content": "x" * 600},
                {"role": "assistant", "content": "y" * 600},
            ])
            loaded = session.load_session(workspace_path, data["id"])
            assert loaded is not None
            with (
                mock.patch.object(
                    compaction.config,
                    "get_model_context_window",
                    return_value=None,
                ),
                mock.patch.object(
                    compaction.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(
                    compaction.workspace_mod,
                    "get_history_budget",
                    return_value=1_000,
                ),
            ):
                trigger = compaction._token_trigger(
                    loaded,
                    workspace_path,
                    (("known", "model"),),
                )

        assert trigger is not None
        self.assertEqual(trigger[0], "history_chars")

    async def test_oversized_history_keeps_messages_not_sent_to_summary(self) -> None:
        """A budgeted compaction may trim only the prefix it actually saw."""
        async def fake_summary(*_args, **_kwargs) -> str:
            return "[SUMMARY]\n" + ("s" * 100) + "\n[/SUMMARY]"

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            messages = [
                {
                    "role": "user",
                    "content": f"message-{i}:" + ("x" * 2_000),
                }
                for i in range(10)
            ]
            session.append_messages(workspace_path, data["id"], messages)
            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(compaction, "MAX_SUMMARY_CHARS", 17_000),
                mock.patch.object(
                    compaction, "run_internal_backend", side_effect=fake_summary,
                ) as run_summary,
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                await compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="codex",
                )

            prompt = run_summary.await_args.args[2]
            expected_lines = compaction._take_oldest_message_lines(
                messages,
                17_000 - len(compaction.SUMMARY_PROMPT) - 200
                - len("Conversation to summarize:"),
            )
            covered_count = len(expected_lines)
            self.assertGreater(covered_count, compaction.KEEP_RECENT_AFTER_COMPACT)
            self.assertLess(covered_count, len(messages))
            self.assertIn("message-0:", prompt)
            self.assertIn(f"message-{covered_count - 1}:", prompt)
            self.assertNotIn(f"message-{covered_count}:", prompt)

            saved = session.load_session(workspace_path, data["id"])
            assert saved is not None
            retained = saved["messages"]
            self.assertEqual(
                [entry["content"] for entry in retained],
                [entry["content"] for entry in messages[
                    covered_count - compaction.KEEP_RECENT_AFTER_COMPACT:
                ]],
            )
            self.assertEqual(
                saved["compacted_count"],
                covered_count - compaction.KEEP_RECENT_AFTER_COMPACT,
            )

    async def test_oversized_first_message_can_be_compacted(self) -> None:
        """The oldest oversized entry must not permanently block compaction."""
        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user",
                "content": "x" * (compaction.MAX_SUMMARY_CHARS + 1),
            }])
            with (
                mock.patch.object(
                    compaction.backends_agent,
                    "get_backend",
                    return_value=SimpleNamespace(name="test"),
                ),
                mock.patch.object(
                    compaction,
                    "run_internal_backend",
                    new=mock.AsyncMock(return_value="[SUMMARY]\nbrief\n[/SUMMARY]"),
                ) as run_summary,
            ):
                _summary, _long_term, _title, covered = await compaction.compact_session(
                    workspace_path, data["id"], "model", backend_name="test",
                )

            self.assertEqual(covered, 1)
            prompt = run_summary.await_args.args[2]
            self.assertIn("…", prompt)
            self.assertLess(len(prompt), compaction.MAX_SUMMARY_CHARS)

    async def test_oversized_previous_summary_can_be_replaced(self) -> None:
        """A bad historical summary must not block every future compaction."""
        async def fake_summary(*_args, **_kwargs) -> str:
            return "[SUMMARY]\n" + ("s" * 100) + "\n[/SUMMARY]"

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [
                {"role": "user", "content": f"message-{i}"}
                for i in range(6)
            ])
            saved = session.load_session(workspace_path, data["id"])
            assert saved is not None
            saved["summary"] = "old-summary " + (
                "x" * compaction.MAX_SUMMARY_CHARS
            )
            session.save_session(workspace_path, data["id"], saved)

            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "run_internal_backend", side_effect=fake_summary,
                ) as run_summary,
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                await compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="codex",
                )

            prompt = run_summary.await_args.args[2]
            self.assertIn("[previous summary truncated]", prompt)
            self.assertLess(len(prompt), compaction.MAX_SUMMARY_CHARS)
            compacted = session.load_session(workspace_path, data["id"])
            assert compacted is not None
            self.assertEqual(compacted["summary"], "s" * 100)
            self.assertEqual(len(compacted["messages"]), 5)

    async def test_colony_can_retire_items_when_sessions_are_empty(self) -> None:
        """Empty session memory is still evidence for a colony reconciliation."""
        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Retired Topic")
            colony.set_items(workspace_path, ["A stale shared fact."])
            output = (
                "[COLONY]\n[/COLONY]\n\n"
                f"[SESSION:{data['id']}]\n[/SESSION]"
            )
            with mock.patch.object(
                colony, "run_internal_backend", new=mock.AsyncMock(
                    return_value=output,
                ),
            ) as run_consolidate:
                applied = await colony.consolidate(
                    workspace_path, "model", backend_name="codex",
                )

            self.assertTrue(applied)
            prompt = run_consolidate.await_args.args[2]
            self.assertIn(f"[SESSION:{data['id']}]", prompt)
            self.assertEqual(colony.get_items(workspace_path), [])

    async def test_colony_oversized_existing_state_still_includes_sessions(self) -> None:
        """Existing memory cannot consume the whole consolidation budget."""
        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Current Topic")
            loaded = session.load_session(workspace_path, data["id"])
            assert loaded is not None
            loaded["long_term"] = ["Current session evidence."]
            session.save_session(workspace_path, data["id"], loaded)
            colony.set_items(
                workspace_path,
                ["x" * (colony.CONSOLIDATE_MAX_INPUT_CHARS * 2)],
            )
            output = (
                "[COLONY]\n- Rewritten shared fact.\n[/COLONY]\n\n"
                f"[SESSION:{data['id']}]\n- Current session evidence.\n[/SESSION]"
            )
            with mock.patch.object(
                colony, "run_internal_backend", new=mock.AsyncMock(
                    return_value=output,
                ),
            ) as run_consolidate:
                applied = await colony.consolidate(
                    workspace_path, "model", backend_name="codex",
                )

            self.assertTrue(applied)
            prompt = run_consolidate.await_args.args[2]
            self.assertIn(f"[SESSION:{data['id']}]", prompt)
            self.assertLess(
                len(prompt),
                len(colony.CONSOLIDATE_PROMPT) + colony.CONSOLIDATE_MAX_INPUT_CHARS + 10,
            )

    async def test_colony_is_cleared_when_no_sessions_remain(self) -> None:
        """Deleted sessions must not leave stale workspace memory behind."""
        with tempfile.TemporaryDirectory() as workspace_path:
            colony.set_items(workspace_path, ["A stale shared fact."])
            with mock.patch.object(
                colony, "run_internal_backend", new=mock.AsyncMock(),
            ) as run_consolidate:
                applied = await colony.consolidate(
                    workspace_path, "model", backend_name="codex",
                )

            self.assertTrue(applied)
            run_consolidate.assert_not_awaited()
            self.assertEqual(colony.get_items(workspace_path), [])


if __name__ == "__main__":
    unittest.main()
