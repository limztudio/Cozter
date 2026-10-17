"""Native conversations survive jobs while retaining exact generation pins."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import execution_state, session, workspace


_SID = "01a126fa-e54a-7753-8078-3e31ff8fc962"
_OTHER_SID = "f53ba06b-a880-4e59-93ea-bd623b0b922b"
_SETTINGS = ("codex", "chosen", "full", 100)


class NativeConversationStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())

    @asynccontextmanager
    async def scope(self, job="job", conversation="conversation", *, settings=_SETTINGS,
                  worker="main", ws=None, native_worker=None):
        async with execution_state.execution_scope(ws or self.ws, job, "request") as execution:
            if conversation is not None:
                execution_state.bind_conversation(conversation)
            with execution_state.worker_scope(worker):
                async with execution_state.backend_scope(
                    *settings, native=True, native_worker=native_worker,
                ) as state:
                    yield execution, state

    async def test_changed_flexible_tasks_reuse_main_lane_without_changing_job_keys(self) -> None:
        async with self.scope("direct") as (_, direct):
            await direct.save(native_session_id=_SID, native_prompt_ledger={"direct": "seen"})
            original = direct.path
        for index, worker in enumerate(("flexible:build-one", "flexible:fix-two")):
            async with self.scope(
                f"worker-{index}", worker=worker, native_worker="main",
            ) as (execution, state):
                self.assertEqual(state.path, original)
                self.assertEqual(state.get("native_session_id"), _SID)
                self.assertEqual(state.get("native_prompt_ledger"), {"direct": "seen"})
                self.assertIsNone(execution.native_main_record)
                job = await execution.record({
                    "backend": "codex", "model": "chosen", "approval": "full",
                    "effort": 100, "worker": worker,
                })
                self.assertEqual(job.get("native_worker"), "main")
                self.assertEqual(job.get("native_record_id"), state.get("native_record_id"))
                self.assertIsNone(job.get("native_prompt_ledger"))
        async with self.scope("direct-followup") as (_, state):
            self.assertEqual(state.path, original)

    async def test_existing_lane_pin_keeps_its_exact_path_and_home_after_requested_lane_changes(self) -> None:
        async with self.scope("pinned", worker="flexible:task", native_worker="main") as (_, state):
            await state.save(native_session_id=_SID)
            original = state.path
            home = state.directory / "copilot-home"
            home.mkdir()
            (home / "proof").write_text("session-owned", encoding="utf-8")
        async with self.scope(
            "pinned", worker="flexible:task", native_worker="flexible:reader:2",
        ) as (_, state):
            self.assertEqual(state.path, original)
            self.assertEqual(state.get("native_session_id"), _SID)
            self.assertEqual((state.directory / "copilot-home" / "proof").read_text(), "session-owned")
        async with self.scope(
            "new-reader", worker="flexible:different", native_worker="flexible:reader:2",
        ) as (_, state):
            self.assertIsNone(state.get("native_session_id"))

    async def test_legacy_generation_pin_retains_detailed_worker_identity(self) -> None:
        worker = "flexible:legacy-task"
        async with self.scope("legacy", worker=worker) as (execution, state):
            await state.save(native_session_id=_SID)
            original = state.path
            job = await execution.record({
                "backend": "codex", "model": "chosen", "approval": "full",
                "effort": 100, "worker": worker,
            })
            job_path = job.path
        data = json.loads(job_path.read_text(encoding="utf-8"))
        data["values"].pop("native_worker")
        job_path.write_text(json.dumps(data), encoding="utf-8")
        async with self.scope("direct") as (_, state):
            await state.save(native_session_id=_OTHER_SID)
            direct_path = state.path
        async with self.scope("legacy", worker=worker, native_worker="main") as (_, state):
            self.assertEqual(state.path, original)
            self.assertEqual(state.get("native_session_id"), _SID)
        async with self.scope("new", worker="flexible:new-task", native_worker="main") as (_, state):
            self.assertEqual(state.path, direct_path)
            self.assertEqual(state.get("native_session_id"), _OTHER_SID)

    async def test_cached_generation_refreshes_after_another_execution_updates_the_lane(self) -> None:
        async with execution_state.execution_scope(self.ws, "older", "request"):
            execution_state.bind_conversation("conversation")
            with execution_state.worker_scope("flexible:old"):
                async with execution_state.backend_scope(
                    *_SETTINGS, native=True, native_worker="main",
                ) as old:
                    await old.save(native_session_id=_SID, native_prompt_ledger={"first": "seen"})
                async with self.scope("newer", native_worker="main") as (_, newer):
                    self.assertEqual(newer.path, old.path)
                    await newer.save(native_prompt_ledger={"first": "seen", "newer": "seen"})
                async with execution_state.backend_scope(
                    *_SETTINGS, native=True, native_worker="main",
                ) as resumed:
                    self.assertIs(resumed, old)
                    self.assertEqual(resumed.get("native_prompt_ledger"), {"first": "seen", "newer": "seen"})

    async def test_same_native_lane_serializes_selection_work_and_publication(self) -> None:
        first_entered = asyncio.Event()
        second_attempted = asyncio.Event()
        second_entered = asyncio.Event()
        release = asyncio.Event()
        selected = []

        async def first():
            async with self.scope("one", worker="flexible:first", native_worker="main") as (_, state):
                selected.append(state.path)
                state.set(native_session_id=_SID, delivered="first")
                first_entered.set()
                await release.wait()

        async def second():
            second_attempted.set()
            async with self.scope("two", worker="flexible:second", native_worker="main") as (_, state):
                selected.append(state.path)
                self.assertEqual(state.get("native_session_id"), _SID)
                self.assertEqual(state.get("delivered"), "first")
                second_entered.set()

        first_task = asyncio.create_task(first())
        await first_entered.wait()
        second_task = asyncio.create_task(second())
        try:
            await second_attempted.wait()
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(second_entered.wait(), 0.05)
        finally:
            release.set()
            await asyncio.gather(first_task, second_task)
        self.assertEqual(selected[0], selected[1])

    async def test_distinct_reader_lanes_can_execute_in_parallel(self) -> None:
        entered = [asyncio.Event(), asyncio.Event()]
        selected = []

        async def reader(index):
            async with self.scope(
                f"job-{index}", worker=f"flexible:task-{index}",
                native_worker=f"flexible:reader:{index}",
            ) as (_, state):
                selected.append(state.path)
                await state.save(native_session_id=(_SID, _OTHER_SID)[index])
                entered[index].set()
                await asyncio.wait_for(entered[1 - index].wait(), 2)

        await asyncio.gather(reader(0), reader(1))
        self.assertNotEqual(selected[0], selected[1])

    async def test_same_job_key_cannot_choose_conflicting_lanes_concurrently(self) -> None:
        first_entered = asyncio.Event()
        release = asyncio.Event()
        selected = []

        async def worker(lane, ready=None):
            async with self.scope("same-job", worker="flexible:same", native_worker=lane) as (_, state):
                selected.append(state.path)
                if ready is not None:
                    await state.save(native_session_id=_SID)
                    ready.set()
                    await release.wait()
                else:
                    self.assertEqual(state.get("native_session_id"), _SID)

        first = asyncio.create_task(worker("main", first_entered))
        await first_entered.wait()
        second = asyncio.create_task(worker("flexible:reader:1"))
        try:
            await asyncio.sleep(0.05)
            self.assertEqual(len(selected), 1)
        finally:
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual(selected[0], selected[1])

    async def test_cancelled_lane_waiter_does_not_block_later_workers(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with self.scope("holder", native_worker="main") as (_, state):
                await state.save(native_session_id=_SID)
                entered.set()
                await release.wait()

        async def waiter():
            async with self.scope("waiter", worker="flexible:waiting", native_worker="main"):
                self.fail("cancelled waiter entered the leased native session")

        owner = asyncio.create_task(holder())
        await entered.wait()
        waiting = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        waiting.cancel()
        try:
            with self.assertRaises(asyncio.CancelledError):
                await waiting
        finally:
            release.set()
            await owner
        async with self.scope("following", worker="flexible:later", native_worker="main") as (_, state):
            self.assertEqual(state.get("native_session_id"), _SID)

    async def test_native_lane_names_and_saved_lane_pins_are_validated(self) -> None:
        for value in ("", " ", " main", True, 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "native worker lane"):
                async with self.scope(native_worker=value):
                    self.fail("invalid native lane was accepted")
        async with self.scope("invalid-pin") as (execution, _):
            job = await execution.record({
                "backend": "codex", "model": "chosen", "approval": "full",
                "effort": 100, "worker": "main",
            })
            await job.save(native_worker=False)
        with self.assertRaisesRegex(RuntimeError, "invalid native worker lane"):
            async with self.scope("invalid-pin", native_worker="main"):
                self.fail("invalid saved lane was silently replaced")

    async def test_new_job_reuses_exact_confirmed_conversation_generation(self) -> None:
        async with self.scope("first") as (execution, state):
            self.assertIs(execution.native_main_record, state)
            self.assertEqual(state.execution.namespace, "native_sessions")
            await state.save(native_session_id=_SID, native_prompt={"delivered": "hash"})
            path = state.path
            generation = state.get("native_record_id")
            job_record = await execution.record({
                "backend": "codex", "model": "chosen", "approval": "full",
                "effort": 100, "worker": "main",
            })
            self.assertEqual(job_record.get("native_record_id"), generation)
            self.assertIsNone(job_record.get("native_session_id"))
        async with self.scope("second") as (_, state):
            self.assertEqual(state.path, path)
            self.assertEqual(state.get("native_record_id"), generation)
            self.assertEqual(state.get("native_session_id"), _SID)
            self.assertEqual(state.get("native_prompt"), {"delivered": "hash"})

    async def test_job_cleanup_preserves_native_home_but_conversation_cleanup_removes_it(self) -> None:
        async with self.scope("completed") as (_, state):
            await state.save(native_session_id=_SID)
            home = state.directory / "copilot-home" / "session-state" / _SID
            home.mkdir(parents=True)
            (home / "events.jsonl").write_text("native transcript", encoding="utf-8")
            native_path = state.path
        await execution_state.discard_execution(self.ws, "completed")
        self.assertTrue(native_path.is_file())
        self.assertTrue(home.is_dir())
        async with self.scope("following") as (_, state):
            self.assertEqual(state.path, native_path)
            self.assertEqual(state.get("native_session_id"), _SID)
        execution_state.discard_native_conversation(self.ws, "conversation")
        self.assertFalse(native_path.exists())
        self.assertFalse(home.exists())

    async def test_temporary_jobs_keep_conversation_state_after_private_job_cleanup(self) -> None:
        async with self.scope(None) as (execution, state):
            await state.save(native_session_id=_SID)
            native_path = state.path
            job_path = Path(workspace.workspace_state_path(
                self.ws, "executions", execution.directory_id,
            ))
        self.assertFalse(job_path.exists())
        self.assertTrue(native_path.exists())
        async with self.scope(None) as (_, state):
            self.assertEqual(state.path, native_path)
            self.assertEqual(state.get("native_session_id"), _SID)

    async def test_unconfirmed_generation_is_pinned_to_its_job_but_not_reused_by_new_jobs(self) -> None:
        async with self.scope("first") as (_, state):
            generation = state.get("native_record_id")
            path = state.path
        async with self.scope("second") as (_, state):
            self.assertNotEqual(state.get("native_record_id"), generation)
            self.assertIsNone(state.get("native_session_id"))
        async with self.scope("first") as (_, state):
            self.assertEqual(state.get("native_record_id"), generation)
            self.assertEqual(state.path, path)

    async def test_old_job_stays_pinned_after_newer_head_and_cannot_repromote_itself(self) -> None:
        async with self.scope("older") as (_, state):
            older = state.get("native_record_id")
        async with self.scope("newer") as (_, state):
            newer = state.get("native_record_id")
            await state.save(native_session_id=_OTHER_SID)
        self.assertNotEqual(older, newer)
        async with self.scope("older") as (_, state):
            self.assertEqual(state.get("native_record_id"), older)
            await state.save(native_session_id=_SID)
        async with self.scope("following") as (_, state):
            self.assertEqual(state.get("native_record_id"), newer)
            self.assertEqual(state.get("native_session_id"), _OTHER_SID)

    async def test_backend_policy_worker_conversation_and_workspace_boundaries_remain_isolated(self) -> None:
        async with self.scope("seed") as (_, state):
            await state.save(native_session_id=_SID)
        settings = (
            ("claude_code", "chosen", "full", 100),
            ("codex", "other", "full", 100),
            ("codex", "chosen", "auto", 100),
            ("codex", "chosen", "full", 50),
        )
        for index, policy in enumerate(settings):
            async with self.scope(f"variant-{index}", settings=policy) as (_, state):
                self.assertIsNone(state.get("native_session_id"))
        for options in ({"conversation": "other"}, {"worker": "flexible:role"}):
            async with self.scope("variant", **options) as (_, state):
                self.assertIsNone(state.get("native_session_id"))
        with tempfile.TemporaryDirectory() as other:
            async with self.scope("variant", ws=other) as (_, state):
                self.assertIsNone(state.get("native_session_id"))
        async with self.scope("same-conversation") as (_, state):
            self.assertEqual(state.get("native_session_id"), _SID)

    async def test_flexible_scope_does_not_replace_main_acknowledgment_record(self) -> None:
        async with self.scope() as (execution, main):
            with execution_state.worker_scope("flexible:role"):
                async with execution_state.backend_scope(*_SETTINGS, native=True) as worker:
                    self.assertIsNot(worker, main)
                    await worker.save(native_session_id=_OTHER_SID)
            self.assertIs(execution.native_main_record, main)
            await main.save(native_session_id=_SID)
        async with self.scope("next", worker="flexible:role") as (execution, worker):
            self.assertEqual(worker.get("native_session_id"), _OTHER_SID)
            self.assertIsNone(execution.native_main_record)

    async def test_non_native_and_unbound_scopes_keep_job_evidence_separate(self) -> None:
        async with execution_state.execution_scope(self.ws, "job", "request") as execution:
            async with execution_state.backend_scope(*_SETTINGS) as job:
                await job.save(history=["private HTTP evidence"], completed_workers=["done"])
            execution_state.bind_conversation("conversation")
            async with execution_state.backend_scope(*_SETTINGS, native=True) as native:
                self.assertIsNone(native.get("history"))
                self.assertIsNone(native.get("completed_workers"))
                await native.save(native_session_id=_SID)
            async with execution_state.backend_scope(*_SETTINGS) as same_job:
                self.assertEqual(same_job.get("history"), ["private HTTP evidence"])
                self.assertIsNone(same_job.get("native_session_id"))
            self.assertIs(execution.native_main_record, native)
        async with self.scope("unbound", conversation=None) as (execution, state):
            self.assertEqual(state.execution.namespace, "executions")
            self.assertIsNone(execution.native_main_record)
            self.assertIsNone(state.get("native_session_id"))

    async def test_existing_unfinished_native_job_stays_in_its_original_home(self) -> None:
        async with execution_state.execution_scope(self.ws, "old-job", "request"):
            async with execution_state.backend_scope(*_SETTINGS) as state:
                await state.save(native_session_id=_SID)
                original = state.path
                home = state.directory / "copilot-home"
                home.mkdir()
        async with self.scope("old-job") as (_, state):
            self.assertEqual(state.path, original)
            self.assertEqual(state.get("native_session_id"), _SID)
            self.assertTrue(home.is_dir())
            self.assertEqual(state.execution.namespace, "executions")

    async def test_invalid_or_missing_generation_pin_does_not_silently_start_fresh(self) -> None:
        async with self.scope() as (execution, state):
            native_path = state.path
            job = await execution.record({
                "backend": "codex", "model": "chosen", "approval": "full",
                "effort": 100, "worker": "main",
            })
            job_path = job.path
        original = json.loads(job_path.read_text(encoding="utf-8"))
        invalid = json.loads(json.dumps(original))
        invalid["values"]["native_record_id"] = "../other"
        job_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "invalid native generation pin"):
            async with self.scope():
                self.fail("invalid pin entered a native scope")
        job_path.write_text(json.dumps(original), encoding="utf-8")
        native_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "generation is missing"):
            async with self.scope():
                self.fail("missing generation entered a native scope")

    async def test_invalid_conversation_head_is_not_treated_as_a_new_conversation(self) -> None:
        async with self.scope("first") as (execution, state):
            await state.save(native_session_id=_SID)
            conversation = execution.conversation
            head = await conversation.record({"native_head": {
                "backend": "codex", "model": "chosen", "approval": "full",
                "effort": 100, "worker": "main",
            }})
            head_path = head.path
        payload = json.loads(head_path.read_text(encoding="utf-8"))
        payload["values"]["native_record_id"] = "--latest"
        head_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "invalid native generation head"):
            async with self.scope("following"):
                self.fail("invalid head entered a native scope")

    async def test_conversation_cleanup_preserves_other_sessions_and_job_evidence(self) -> None:
        async with self.scope("one", "first") as (_, state):
            await state.save(native_session_id=_SID)
            first = state.path
        async with self.scope("two", "second") as (execution, state):
            await state.save(native_session_id=_OTHER_SID)
            second = state.path
            job = await execution_state.record("plan")
            await job.save(done=True)
            plan_path = job.path
            self.assertEqual(execution.conversation.execution_id, "second")
        execution_state.discard_native_conversation(self.ws, "first")
        self.assertFalse(first.exists())
        self.assertTrue(second.is_file())
        self.assertTrue(plan_path.is_file())

    async def test_public_session_delete_removes_its_native_conversation(self) -> None:
        session_id = session.create_session(self.ws)["id"]
        async with self.scope(conversation=session_id) as (_, state):
            await state.save(native_session_id=_SID)
            native_path = state.path
        self.assertTrue(native_path.is_file())

        self.assertTrue(session.delete_session(self.ws, session_id))

        self.assertIsNone(session.load_session(self.ws, session_id))
        self.assertFalse(native_path.exists())

    async def test_public_session_delete_reports_success_when_native_cleanup_fails(self) -> None:
        for error in (OSError("native file busy"), ValueError("unsafe native path")):
            with self.subTest(error=type(error).__name__):
                session_id = session.create_session(self.ws)["id"]
                async with self.scope(job=session_id, conversation=session_id) as (_, state):
                    await state.save(native_session_id=_SID)
                    native_path = state.path
                with (
                    mock.patch.object(execution_state, "discard_native_conversation", side_effect=error),
                    self.assertLogs(session.logger, level="WARNING") as logs,
                ):
                    self.assertTrue(session.delete_session(self.ws, session_id))
                self.assertIsNone(session.load_session(self.ws, session_id))
                self.assertTrue(native_path.is_file())
                self.assertTrue(any("Failed to clean native state" in line for line in logs.output))
                # A retry can remove the orphan despite the session file
                # already being absent; its boolean still describes unlink.
                self.assertFalse(session.delete_session(self.ws, session_id))
                self.assertFalse(native_path.exists())

    async def test_public_session_delete_attempts_native_cleanup_when_file_is_absent(self) -> None:
        session_id = session.create_session(self.ws)["id"]
        async with self.scope(conversation=session_id) as (_, state):
            await state.save(native_session_id=_SID)
            native_path = state.path
        Path(workspace.workspace_state_path(
            self.ws, "sessions", f"{session_id}.json",
        )).unlink()
        with mock.patch.object(
            execution_state, "discard_native_conversation",
            wraps=execution_state.discard_native_conversation,
        ) as cleanup:
            self.assertFalse(session.delete_session(self.ws, session_id))
        cleanup.assert_called_once_with(self.ws, session_id)
        self.assertFalse(native_path.exists())

    async def test_public_session_delete_prevents_late_state_and_head_writes(self) -> None:
        session_id = session.create_session(self.ws)["id"]
        async with self.scope(conversation=session_id) as (_, state):
            directory = state.path.parents[1]
            state.set(native_session_id=_SID, native_prompt_ledger={"pending": "hash"})
            self.assertTrue(session.delete_session(self.ws, session_id))
            await state.flush()
            self.assertFalse(directory.exists())
        self.assertIsNone(session.load_session(self.ws, session_id))
        self.assertFalse(directory.exists())

    async def test_cleanup_prevents_queued_writes_and_head_publication_from_resurrecting_session(self) -> None:
        async with self.scope() as (_, state):
            path = state.path
            directory = path.parents[1]
            state.set(native_session_id=_SID)
            execution_state.discard_native_conversation(self.ws, "conversation")
            await state.flush()
            self.assertFalse(directory.exists())
        self.assertFalse(directory.exists())

    async def test_cleanup_waits_for_inflight_atomic_write_then_removes_its_result(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original = execution_state._write_snapshot

        def delayed(path, snapshot):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test writer was not released")
            original(path, snapshot)

        async with self.scope() as (_, state):
            path = state.path
            directory = path.parents[1]
            with mock.patch.object(execution_state, "_write_snapshot", delayed):
                state.set(native_session_id=_SID)
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                cleanup = asyncio.create_task(asyncio.to_thread(
                    execution_state.discard_native_conversation, self.ws, "conversation",
                ))
                try:
                    await asyncio.sleep(0)
                    self.assertFalse(cleanup.done())
                finally:
                    release.set()
                await cleanup
                await state.flush()
            self.assertFalse(directory.exists())
        self.assertFalse(directory.exists())

    async def test_cancellation_flushes_native_state_and_head_before_next_job(self) -> None:
        ready = asyncio.Event()

        async def worker():
            async with self.scope("cancelled") as (_, state):
                state.set(native_session_id=_SID, delivered={"section": "hash"})
                ready.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await ready.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(execution_state.current_run())
        async with self.scope("next") as (_, state):
            self.assertEqual(state.get("native_session_id"), _SID)
            self.assertEqual(state.get("delivered"), {"section": "hash"})

    async def test_repeated_cancellation_settles_inflight_native_write_and_head(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original = execution_state._write_snapshot

        def delayed(path, snapshot):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test writer was not released")
            original(path, snapshot)

        async def worker():
            async with self.scope("cancelled") as (_, state):
                with mock.patch.object(execution_state, "_write_snapshot", delayed):
                    state.set(native_session_id=_SID)
                    await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        async with self.scope("next") as (_, state):
            self.assertEqual(state.get("native_session_id"), _SID)

    async def test_namespaces_bindings_and_cleanup_paths_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "namespace"):
            execution_state.Execution(self.ws, "id", "", namespace="../outside")
        async with self.scope() as (execution, state):
            for bad in ("", " ", " conversation", None, True):
                with self.subTest(value=bad), self.assertRaises(ValueError):
                    execution_state.bind_conversation(bad)
            execution_state.bind_conversation("conversation")
            with self.assertRaisesRegex(ValueError, "cannot change"):
                execution_state.bind_conversation("other")
            self.assertIs(execution.native_main_record, state)
        with tempfile.TemporaryDirectory() as outside:
            marker = Path(outside, "keep")
            marker.write_text("preserved", encoding="utf-8")

            def unsafe(_workspace, *parts):
                return str(Path(self.ws, ".cozter", parts[0])) if len(parts) == 1 else outside

            with mock.patch.object(workspace, "workspace_state_path", side_effect=unsafe):
                with self.assertRaisesRegex(ValueError, "unsafe backend continuation cleanup"):
                    execution_state.discard_native_conversation(self.ws, "unsafe")
            self.assertTrue(marker.is_file())


if __name__ == "__main__":
    unittest.main()
