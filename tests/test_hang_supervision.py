"""Hang reviews preserve useful work and settle only a confirmed operation."""

import asyncio
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import config, process_observation, supervision


def decision(verdict="confirmed_hang"):
    return {"verdict": verdict, "evidence": "All participants wait on an unreachable latch count.",
            "diagnosis": "The latch requires eight arrivals but only six participants exist.",
            "repair": "Correct the participant count, rebuild, and resume the failed test.", "progress_paths": []}


def snapshot(*, cpu=1.0, io=0, child_io=0):
    root = process_observation.ProcessIdentity(10, "windows:100")
    child = process_observation.ProcessIdentity(11, "windows:200")
    return process_observation.ProcessSnapshot(root, (
        process_observation.ProcessSample(root, 1, "worker", cpu, io, io, io, io),
        process_observation.ProcessSample(child, 10, "test", cpu, child_io, child_io, child_io, child_io),
    ))


class HangSupervisionPolicyTests(unittest.TestCase):
    def test_diagnosis_parser_rejects_malformed_and_unsupported_evidence(self):
        for raw in ("not JSON", "[]", '{"verdict":"confirmed_hang"}', json.dumps(decision("kill_now")),
                    json.dumps({**decision(), "evidence": " "}), json.dumps({**decision(), "repair": False})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                supervision._parse_diagnosis(raw)
        for verdict in ("progressing", "uncertain", "confirmed_hang", "replan_read"):
            expected = decision(verdict)
            self.assertEqual(supervision._parse_diagnosis("```json\n" + json.dumps(expected) + "\n```"), expected)

    def test_worker_context_and_diagnostic_recursion_are_isolated(self):
        self.assertIsNone(supervision.new_operation("workspace", "bash"))
        with supervision.worker_scope("workspace", summary_backend_name="meta", summary_model="doctor-model"):
            ticket = supervision.new_operation("workspace", "bash")
            self.assertEqual(ticket.worker.backend, "meta")
            self.assertEqual(ticket.worker.model, "doctor-model")
            with supervision.operation_scope(ticket):
                with supervision.diagnostic_scope():
                    self.assertIsNone(supervision.new_operation("workspace", "bash"))
                    supervision.note_output("diagnostic output")
                self.assertEqual(ticket.output, "")
                supervision.note_output("owned output")
                self.assertEqual(ticket.output, "owned output")
        self.assertIsNone(supervision.new_operation("workspace", "bash"))

    def test_changed_command_or_process_invalidates_old_diagnosis_version(self):
        ticket = supervision.Operation(supervision.Worker("workspace", "meta", "model"), "bash")
        with supervision.operation_scope(ticket):
            first = SimpleNamespace(pid=10)
            second = SimpleNamespace(pid=11)
            supervision.attach_process(first, "first command")
            version = ticket.version
            supervision.attach_process(first, "first command")
            self.assertEqual(ticket.version, version)
            supervision.note_command("second command")
            self.assertGreater(ticket.version, version)
            version = ticket.version
            supervision.attach_process(second, "second command")
            self.assertGreater(ticket.version, version)

    def test_invalid_progress_paths_are_observation_unknown_not_exceptions(self):
        with tempfile.TemporaryDirectory() as root:
            ticket = supervision.Operation(supervision.Worker(root, "meta", "model"), "bash")
            ticket.progress_paths = ("bad\x00path", "../outside.txt", "missing.log")
            stamps = supervision._file_stamps(ticket)
            self.assertIsNone(stamps["bad\x00path"])
            self.assertIsNone(stamps["missing.log"])
            self.assertNotIn("../outside.txt", stamps)

    def test_only_owned_builtin_grep_can_opt_in_to_read_replanning(self):
        for kind in ("native", "tool:bash", "tool:write_file", "tool:grep"):
            ticket = supervision.Operation(supervision.Worker("workspace", "meta", "model"), kind)
            with supervision.operation_scope(ticket):
                supervision.mark_replannable_read()
            self.assertEqual(ticket.can_replan, kind == "tool:grep")


class HangSupervisionAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.tasks = []
        self.releases = []
        self.enterContext(mock.patch.object(config, "get_hang_check_interval_seconds", return_value=0.001))
        self.enterContext(mock.patch.object(config, "get_hang_diagnose_after_seconds", return_value=0.01))
        self.enterContext(mock.patch.object(config, "get_hang_review_interval_seconds", return_value=60))
        self.enterContext(mock.patch.object(config, "get_hang_review_timeout_seconds", return_value=0.2))
        self.enterContext(mock.patch.object(config, "get_hang_long_running_review_seconds", return_value=1800))
        self.original_observe = supervision._observe
        self.observe = self.enterContext(mock.patch.object(supervision, "_observe", new=mock.AsyncMock()))
        self.report = self.enterContext(mock.patch.object(supervision, "_write_report", return_value="diagnosis.json"))
        self.original_diagnose = supervision._diagnose
        self.doctor = self.enterContext(mock.patch.object(supervision, "_diagnose", new=mock.AsyncMock(return_value=decision())))

    async def asyncTearDown(self):
        for event in self.releases:
            event.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def event(self):
        value = asyncio.Event()
        self.releases.append(value)
        return value

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def ticket(self, kind="bash"):
        value = supervision.Operation(supervision.Worker(self.ws, "meta", "doctor-model"), kind)
        value.progress_at -= 60
        value.last_review -= 60
        return value

    def read_ticket(self):
        ticket = self.ticket("tool:grep")
        ticket.can_replan = True
        ticket.process = SimpleNamespace(pid=10)
        ticket.identity = snapshot().root
        ticket.snapshot = snapshot()
        return ticket

    async def test_replan_verdict_cannot_stop_native_mutable_or_unmarked_reads(self):
        for kind, marked in (("native", True), ("tool:bash", True), ("tool:write_file", True), ("tool:grep", False)):
            with self.subTest(kind=kind, marked=marked):
                finish, reviewed, applied = self.event(), self.event(), self.event()
                ticket = self.read_ticket()
                ticket.kind, ticket.can_replan = kind, marked

                async def diagnose(*_args):
                    reviewed.set()
                    return decision("replan_read")

                async def observe(_ticket):
                    if reviewed.is_set():
                        applied.set()

                self.doctor.side_effect, self.observe.side_effect = diagnose, observe
                owned = self.spawn(finish.wait())
                monitor = self.spawn(supervision.wait_owned(owned, ticket))
                await asyncio.wait_for(applied.wait(), 2)
                self.assertFalse(owned.done())
                finish.set()
                self.assertIsNone(await asyncio.wait_for(monitor, 2))
                self.assertFalse(owned.cancelled())

    async def test_expensive_read_replanning_settles_owner_and_retains_partial_coverage(self):
        cleanup, release_cleanup = self.event(), self.event()
        ticket = self.read_ticket()
        ticket.command = 'grep path="." pattern="rare_symbol" glob="**/*"'
        ticket.output = "core/module.cpp:7: rare_symbol"
        self.doctor.return_value = decision("replan_read")

        async def work():
            try:
                await asyncio.Future()
            finally:
                cleanup.set()
                await release_cleanup.wait()

        owned = self.spawn(work())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(cleanup.wait(), 2)
        self.assertFalse(monitor.done())
        release_cleanup.set()
        result = await asyncio.wait_for(monitor, 2)
        self.assertTrue(owned.cancelled())
        self.assertIn("not a confirmed deadlock or a complete search", result)
        self.assertIn(ticket.output, result)
        self.assertIn(decision()["repair"], result)
        self.doctor.assert_awaited_once()

    async def test_repeated_uncertain_read_reviews_yield_advice_and_include_prior_findings(self):
        ticket = self.read_ticket()
        evidence_seen = []

        async def diagnose(_ticket, evidence, _timeout):
            evidence_seen.append(evidence)
            return decision("uncertain")

        self.doctor.side_effect = diagnose
        with mock.patch.object(config, "get_hang_review_interval_seconds", return_value=0):
            owned = self.spawn(asyncio.Event().wait())
            result = await asyncio.wait_for(supervision.wait_owned(owned, ticket), 2)
        self.assertTrue(owned.cancelled())
        self.assertEqual(len(evidence_seen), 2)
        self.assertEqual(evidence_seen[1]["review_count"], 2)
        self.assertEqual(evidence_seen[1]["previous_reviews"][0]["verdict"], "uncertain")
        self.assertIn("expensive read-only search was stopped", result)
        self.assertEqual(self.report.call_args.args[2]["review_verdict"], "uncertain")

    async def test_uncertain_write_reviews_and_unavailable_read_diagnoses_cannot_trigger_replanning(self):
        for kind, error in (("tool:write_file", None), ("tool:grep", ValueError("malformed")),
                            ("tool:grep", RuntimeError("provider unavailable"))):
            with self.subTest(kind=kind, error=error):
                finish, twice = self.event(), self.event()
                ticket = self.read_ticket()
                ticket.kind = kind
                reviews = 0

                async def diagnose(*_args):
                    nonlocal reviews
                    reviews += 1
                    if reviews == 2:
                        twice.set()
                    if error:
                        raise error
                    return decision("uncertain")

                self.doctor.side_effect = diagnose
                with mock.patch.object(config, "get_hang_review_interval_seconds", return_value=0):
                    owned = self.spawn(finish.wait())
                    monitor = self.spawn(supervision.wait_owned(owned, ticket))
                    await asyncio.wait_for(twice.wait(), 2)
                    self.assertFalse(owned.done())
                    finish.set()
                    self.assertIsNone(await asyncio.wait_for(monitor, 2))
                self.assertFalse(owned.cancelled())

    async def test_replanning_requires_a_verified_owned_process(self):
        for identity, observed in ((None, snapshot()), (snapshot().root, None)):
            with self.subTest(identity=identity, observed=observed):
                finish, applied = self.event(), self.event()
                ticket = self.read_ticket()
                ticket.identity, ticket.snapshot = identity, observed
                self.doctor.return_value = decision("replan_read")

                async def observe(_ticket):
                    if self.doctor.await_count:
                        applied.set()

                self.observe.side_effect = observe
                self.doctor.reset_mock()
                owned = self.spawn(finish.wait())
                monitor = self.spawn(supervision.wait_owned(owned, ticket))
                await asyncio.wait_for(applied.wait(), 2)
                self.assertFalse(owned.done())
                finish.set()
                self.assertIsNone(await asyncio.wait_for(monitor, 2))
                self.assertFalse(owned.cancelled())

    async def test_new_matches_during_read_diagnosis_make_replan_stale(self):
        finish, entered, release, applied = (self.event() for _ in range(4))
        ticket = self.read_ticket()

        async def diagnose(*_args):
            entered.set()
            await release.wait()
            return decision("replan_read")

        async def observe(_ticket):
            if release.is_set():
                applied.set()

        self.doctor.side_effect, self.observe.side_effect = diagnose, observe
        owned = self.spawn(finish.wait())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(entered.wait(), 2)
        with supervision.operation_scope(ticket):
            supervision.note_output("core/next.cpp:10: rare_symbol")
        release.set()
        await asyncio.wait_for(applied.wait(), 2)
        self.assertFalse(owned.done())
        finish.set()
        self.assertIsNone(await asyncio.wait_for(monitor, 2))
        self.report.assert_not_called()

    async def test_operator_cancel_during_read_cleanup_never_returns_replan_success(self):
        cleanup, release_cleanup = self.event(), self.event()
        self.doctor.return_value = decision("replan_read")

        async def work():
            try:
                await asyncio.Future()
            finally:
                cleanup.set()
                await release_cleanup.wait()

        owned = self.spawn(work())
        monitor = self.spawn(supervision.wait_owned(owned, self.read_ticket()))
        await asyncio.wait_for(cleanup.wait(), 2)
        monitor.cancel()
        release_cleanup.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(monitor, 2)
        self.assertTrue(owned.cancelled())

    async def test_process_io_and_child_churn_are_observed_without_resetting_quiet_clock(self):
        for kind in ("native", "tool:bash"):
            with self.subTest(kind=kind):
                ticket = self.ticket(kind)
                ticket.process = SimpleNamespace(pid=10)
                ticket.identity = snapshot().root
                ticket.snapshot = snapshot()
                quiet_at, version = ticket.progress_at, ticket.version
                cpu_only = snapshot(cpu=100)
                root_io = snapshot(cpu=100, io=500)
                child_io = snapshot(cpu=100, io=500, child_io=500)
                changed_tree = process_observation.ProcessSnapshot(child_io.root, child_io.processes[:1])
                for sample in (cpu_only, root_io, child_io, changed_tree):
                    with (mock.patch.object(process_observation, "snapshot_process_tree", return_value=sample),
                          mock.patch.object(supervision, "_file_stamps", return_value={})):
                        await self.original_observe(ticket)
                    self.assertIs(ticket.snapshot, sample)
                    self.assertEqual(ticket.progress_at, quiet_at)
                    self.assertEqual(ticket.version, version)

    async def test_observed_progress_file_changes_advance_quiet_clock(self):
        ticket = self.ticket()
        ticket.progress_paths = ("build.log",)
        ticket.file_stamps = {"build.log": (1, 10)}
        version = ticket.version
        with mock.patch.object(supervision, "_file_stamps", return_value={"build.log": (2, 20)}):
            await self.original_observe(ticket)
        self.assertEqual(ticket.file_stamps, {"build.log": (2, 20)})
        self.assertGreater(ticket.version, version)

    async def test_failed_process_sampler_replaces_stale_observation_with_unknown(self):
        ticket = self.ticket()
        ticket.process = SimpleNamespace(pid=10)
        ticket.identity = snapshot().root
        ticket.snapshot = snapshot()
        version, quiet_at = ticket.version, ticket.progress_at
        with (mock.patch.object(process_observation, "snapshot_process_tree", side_effect=OSError("fixture unavailable")),
              mock.patch.object(supervision, "_file_stamps", return_value={})):
            await self.original_observe(ticket)
        self.assertIsNone(ticket.snapshot.root)
        self.assertEqual(ticket.snapshot.processes, ())
        self.assertIn("unavailable", ticket.snapshot.error)
        self.assertEqual(ticket.identity, snapshot().root)
        self.assertEqual(ticket.version, version)
        self.assertEqual(ticket.progress_at, quiet_at)

    async def test_failed_fresh_observation_cannot_confirm_a_known_process_hang(self):
        finish, reviewed, observed_after_review = self.event(), self.event(), self.event()
        ticket = self.ticket()
        ticket.process = SimpleNamespace(pid=10)
        ticket.identity = snapshot().root
        ticket.snapshot = snapshot()

        async def diagnose(*_args):
            reviewed.set()
            return decision()

        async def observe(current):
            if reviewed.is_set():
                current.snapshot = process_observation.ProcessSnapshot(None, (), "fixture unavailable")
                observed_after_review.set()

        self.doctor.side_effect = diagnose
        self.observe.side_effect = observe
        owned = self.spawn(finish.wait())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(observed_after_review.wait(), 2)
        self.assertFalse(owned.done())
        finish.set()
        self.assertIsNone(await asyncio.wait_for(monitor, 2))
        self.assertFalse(owned.cancelled())
        self.report.assert_not_called()

    async def test_unknown_initial_process_identity_cannot_confirm_a_hang(self):
        finish, reviewed, observed_after_review = self.event(), self.event(), self.event()
        ticket = self.ticket()
        ticket.process = SimpleNamespace(pid=10)

        async def diagnose(*_args):
            reviewed.set()
            return decision()

        async def observe(current):
            current.snapshot = process_observation.ProcessSnapshot(None, (), "fixture unavailable")
            if reviewed.is_set():
                observed_after_review.set()

        self.doctor.side_effect = diagnose
        self.observe.side_effect = observe
        owned = self.spawn(finish.wait())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(observed_after_review.wait(), 2)
        self.assertIsNone(ticket.identity)
        self.assertFalse(owned.done())
        finish.set()
        self.assertIsNone(await asyncio.wait_for(monitor, 2))
        self.assertFalse(owned.cancelled())
        self.report.assert_not_called()

    async def test_useful_progress_can_continue_without_any_execution_deadline(self):
        finish = self.event()
        ticket = self.ticket()
        probes = 0

        async def observe(current):
            nonlocal probes
            probes += 1
            supervision._advance(current)
            if probes == 5:
                finish.set()

        self.observe.side_effect = observe
        owned = self.spawn(finish.wait())
        result = await asyncio.wait_for(supervision.wait_owned(owned, ticket), 2)
        self.assertIsNone(result)
        self.assertGreaterEqual(probes, 5)
        self.assertFalse(owned.cancelled())
        self.doctor.assert_not_awaited()

    async def test_recently_active_long_task_is_reviewed_and_progressing_work_survives(self):
        finish, reviewed, review_applied = self.event(), self.event(), self.event()
        ticket = self.ticket()
        ticket.started -= 3600
        ticket.last_review = 0
        ticket.progress_at = supervision.time.monotonic()
        post_review_observations = 0

        async def observe(_ticket):
            nonlocal post_review_observations
            if reviewed.is_set():
                post_review_observations += 1
                if post_review_observations == 2:
                    review_applied.set()

        async def diagnose(_ticket, evidence, _timeout):
            self.assertGreaterEqual(evidence["elapsed_seconds"], 3600)
            self.assertLess(evidence["quiet_seconds"], 3600)
            reviewed.set()
            return decision("progressing")

        self.observe.side_effect = observe
        self.doctor.side_effect = diagnose
        with mock.patch.object(config, "get_hang_diagnose_after_seconds", return_value=3600):
            owned = self.spawn(finish.wait())
            monitor = self.spawn(supervision.wait_owned(owned, ticket))
            await asyncio.wait_for(review_applied.wait(), 2)
            self.assertFalse(owned.done())
            self.report.assert_called_once()
            finish.set()
            self.assertIsNone(await asyncio.wait_for(monitor, 2))
        self.assertFalse(owned.cancelled())
        self.doctor.assert_awaited_once()

    async def test_new_short_task_is_not_reviewed_from_older_service_uptime(self):
        finish = self.event()
        ticket = self.ticket()
        ticket.progress_at = supervision.time.monotonic()
        ticket.last_review = ticket.started - 3600
        probes = 0

        async def observe(_ticket):
            nonlocal probes
            probes += 1
            if probes == 5:
                finish.set()

        self.observe.side_effect = observe
        with mock.patch.object(config, "get_hang_diagnose_after_seconds", return_value=3600):
            owned = self.spawn(finish.wait())
            self.assertIsNone(await asyncio.wait_for(supervision.wait_owned(owned, ticket), 2))
        self.assertGreaterEqual(probes, 5)
        self.assertFalse(owned.cancelled())
        self.doctor.assert_not_awaited()

    async def test_cpu_only_busy_operation_is_reviewed_without_automatic_interruption(self):
        finish, reviewed = self.event(), self.event()
        ticket = self.ticket()
        cpu = 0

        async def observe(current):
            nonlocal cpu
            cpu += 100
            current.snapshot = snapshot(cpu=cpu)
            if reviewed.is_set():
                finish.set()

        async def diagnose(_ticket, evidence, _timeout):
            self.assertGreater(evidence["processes"]["processes"][0]["cpu_seconds"], 0)
            reviewed.set()
            return decision("uncertain")

        self.observe.side_effect = observe
        self.doctor.side_effect = diagnose
        owned = self.spawn(finish.wait())
        self.assertIsNone(await asyncio.wait_for(supervision.wait_owned(owned, ticket), 2))
        self.doctor.assert_awaited_once()
        self.assertFalse(owned.cancelled())

    async def test_uncertain_malformed_unavailable_and_cancelled_doctor_preserve_work(self):
        for outcome in (decision("uncertain"), decision("progressing"), ValueError("malformed JSON"),
                        RuntimeError("provider unavailable"), asyncio.CancelledError()):
            with self.subTest(outcome=outcome):
                finish, reviewed, observed_after_review = self.event(), self.event(), self.event()
                ticket = self.ticket()

                async def diagnose(*_args):
                    reviewed.set()
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome

                async def observe(_ticket):
                    if reviewed.is_set():
                        observed_after_review.set()

                self.doctor.side_effect = diagnose
                self.observe.side_effect = observe
                owned = self.spawn(finish.wait())
                monitor = self.spawn(supervision.wait_owned(owned, ticket))
                await asyncio.wait_for(observed_after_review.wait(), 2)
                self.assertFalse(owned.done())
                finish.set()
                self.assertIsNone(await asyncio.wait_for(monitor, 2))
                self.assertFalse(owned.cancelled())

    async def test_completion_wins_and_waits_for_doctor_cleanup(self):
        finish, entered, cleanup, release_cleanup = (self.event() for _ in range(4))
        ticket = self.ticket()

        async def diagnose(*_args):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleanup.set()
                await release_cleanup.wait()

        self.doctor.side_effect = diagnose
        owned = self.spawn(finish.wait())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(entered.wait(), 2)
        finish.set()
        await asyncio.wait_for(cleanup.wait(), 2)
        self.assertFalse(monitor.done())
        release_cleanup.set()
        self.assertIsNone(await asyncio.wait_for(monitor, 2))
        self.assertFalse(owned.cancelled())
        self.report.assert_not_called()

    async def test_output_while_doctor_runs_makes_confirmed_verdict_stale(self):
        finish, entered, release, observed_after_review = (self.event() for _ in range(4))
        ticket = self.ticket()

        async def diagnose(*_args):
            entered.set()
            await release.wait()
            return decision()

        async def observe(_ticket):
            if release.is_set():
                observed_after_review.set()

        self.doctor.side_effect = diagnose
        self.observe.side_effect = observe
        owned = self.spawn(finish.wait())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(entered.wait(), 2)
        with supervision.operation_scope(ticket):
            supervision.note_output(b"new verified progress")
        release.set()
        await asyncio.wait_for(observed_after_review.wait(), 2)
        self.assertFalse(owned.done())
        finish.set()
        self.assertIsNone(await asyncio.wait_for(monitor, 2))
        self.assertFalse(owned.cancelled())
        self.report.assert_not_called()

    async def test_confirmed_hang_returns_recovery_only_after_owned_cleanup_finishes(self):
        cleanup, release_cleanup, settled = (self.event() for _ in range(3))
        ticket = self.ticket()

        async def work():
            try:
                await asyncio.Future()
            finally:
                cleanup.set()
                await release_cleanup.wait()
                settled.set()

        owned = self.spawn(work())
        monitor = self.spawn(supervision.wait_owned(owned, ticket))
        await asyncio.wait_for(cleanup.wait(), 2)
        self.assertFalse(monitor.done())
        self.assertFalse(settled.is_set())
        release_cleanup.set()
        result = await asyncio.wait_for(monitor, 2)
        self.assertTrue(settled.is_set())
        self.assertTrue(owned.cancelled())
        self.assertIn("original task remains unfinished", result)
        self.assertIn("diagnosis.json", result)
        self.assertIn(decision()["repair"], result)

    async def test_operator_cancellation_during_diagnosis_settles_doctor_and_propagates(self):
        finish, entered, cleanup, release_cleanup = (self.event() for _ in range(4))

        async def diagnose(*_args):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleanup.set()
                await release_cleanup.wait()

        self.doctor.side_effect = diagnose
        owned = self.spawn(finish.wait())
        monitor = self.spawn(supervision.wait_owned(owned, self.ticket()))
        await asyncio.wait_for(entered.wait(), 2)
        monitor.cancel()
        await asyncio.wait_for(cleanup.wait(), 2)
        monitor.cancel()
        await asyncio.sleep(0)
        self.assertFalse(monitor.done())
        release_cleanup.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(monitor, 2)
        self.assertFalse(owned.done())  # The external tool owner handles its own cancellation.
        finish.set()
        await owned
        self.report.assert_not_called()

    async def test_operator_cancel_during_confirmed_cleanup_never_returns_success(self):
        cleanup, release_cleanup = self.event(), self.event()

        async def work():
            try:
                await asyncio.Future()
            finally:
                cleanup.set()
                await release_cleanup.wait()

        owned = self.spawn(work())
        monitor = self.spawn(supervision.wait_owned(owned, self.ticket()))
        await asyncio.wait_for(cleanup.wait(), 2)
        monitor.cancel()
        await asyncio.sleep(0)
        monitor.cancel()
        self.assertFalse(monitor.done())
        release_cleanup.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(monitor, 2)
        self.assertTrue(owned.cancelled())

    async def test_parallel_operations_have_independent_progress_and_cancellation(self):
        finish_other = self.event()
        stuck, healthy = self.ticket(), self.ticket()

        async def observe(ticket):
            if ticket is healthy:
                supervision._advance(ticket)

        async def diagnose(ticket, *_args):
            self.assertIs(ticket, stuck)
            return decision()

        self.observe.side_effect = observe
        self.doctor.side_effect = diagnose
        owned_stuck = self.spawn(asyncio.Event().wait())
        owned_healthy = self.spawn(finish_other.wait())
        stuck_monitor = self.spawn(supervision.wait_owned(owned_stuck, stuck))
        healthy_monitor = self.spawn(supervision.wait_owned(owned_healthy, healthy))
        self.assertIn("confirmed hang", await asyncio.wait_for(stuck_monitor, 2))
        self.assertTrue(owned_stuck.cancelled())
        self.assertFalse(owned_healthy.done())
        finish_other.set()
        self.assertIsNone(await asyncio.wait_for(healthy_monitor, 2))
        self.assertFalse(owned_healthy.cancelled())
        self.doctor.assert_awaited_once()

    async def test_diagnostic_runner_uses_configured_model_read_only_and_suspended_execution(self):
        from Cozter import agent_tools, backends_agent, execution_state, utils

        ticket = self.ticket()
        proc = SimpleNamespace(returncode=0)

        async def launch(_workspace, _prompt, model, *, approval, compaction, effort):
            self.assertEqual(model, "doctor-model")
            self.assertEqual(approval, "confirm")
            self.assertFalse(compaction)
            self.assertEqual(effort, 20)
            self.assertTrue(agent_tools.diagnostic_reading_active())
            self.assertIsNone(execution_state.current_run())
            self.assertIsNone(supervision.new_operation(self.ws, "bash"))
            return proc

        backend = SimpleNamespace(launch=mock.AsyncMock(side_effect=launch))
        with (mock.patch.object(backends_agent, "get_backend", return_value=backend) as get_backend,
              mock.patch.object(utils, "drain_llm_subprocess", new=mock.AsyncMock(return_value=json.dumps(decision("uncertain")))),
              mock.patch.object(utils, "cleanup_backend_process", new=mock.AsyncMock()) as cleanup,
              mock.patch.object(utils, "kill_and_wait", new=mock.AsyncMock()) as kill):
            result = await self.original_diagnose(ticket, {"fixture": "evidence"}, 1)
        self.assertEqual(result["verdict"], "uncertain")
        get_backend.assert_called_once_with("meta")
        cleanup.assert_awaited_once()
        kill.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
