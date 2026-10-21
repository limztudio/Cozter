"""Pending outcomes reuse durable history without weakening recovery barriers."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend
from Cozter.tests.test_http_execution_continuation import _call


_TRANSCRIPT_FIELDS = {
    "http_identity", "http_messages", "http_prompt_digest", "http_pending_results",
    "http_argument_repairs",
}


class _ObservedRecord:
    def __init__(self, record, *, expand_pending=False):
        self.record = record
        self.expand_pending = expand_pending
        self.submitted_fields = []

    def get(self, key, default=None):
        return self.record.get(key, default)

    async def save(self, **values):
        self.submitted_fields.append(set(values))
        if self.expand_pending and set(values) == {"http_pending_results"}:
            # Reconstruct exactly the previous full-history barrier so both
            # implementations can be compared using real atomic disk writes.
            values = {key: self.record.get(key) for key in (
                "http_identity", "http_messages", "http_prompt_digest",
            )} | values
        await self.record.save(**values)


class HttpPendingCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def run_backend(self, workspace, *, prompt="synthetic request"):
        await OfflineHttpBackend()._run_agent(
            SimpleNamespace(emit=lambda _event: None), workspace,
            prompt, "model-a", "full", False, 50,
        )

    async def test_pending_barriers_keep_exact_legacy_atomic_snapshot_bytes(self):
        snapshots = {"legacy": [], "candidate": []}
        fields = {}
        calls = [_call(str(index), path=str(index)) for index in range(4)]
        calls.append(_call("write", "write_file"))
        calls.extend(_call(str(index), path=str(index)) for index in range(5, 9))
        calls.append(_call("build", "bash"))
        original_write = execution_state._write_snapshot

        with tempfile.TemporaryDirectory() as workspace:
            for mode in ("legacy", "candidate"):
                rounds = 0

                async def stream(*_args, **_kwargs):
                    nonlocal rounds
                    rounds += 1
                    return ("", "", calls) if rounds == 1 else ("verified", "", [])

                def capture(path, snapshot):
                    original_write(path, snapshot)
                    snapshots[mode].append(Path(path).read_bytes())

                async with execution_state.execution_scope(workspace, mode, "synthetic request"):
                    record = await execution_state.record("offline-checkpoint")
                    observed = _ObservedRecord(record, expand_pending=mode == "legacy")

                    async def execute(name, _args, *_unused):
                        disk = json.loads(record.path.read_bytes())["values"]
                        self.assertEqual(disk["http_messages"][-1]["tool_calls"], calls)
                        if name == "write_file":
                            self.assertEqual(set(disk["http_pending_results"]), {str(i) for i in range(4)})
                        if name == "bash":
                            self.assertEqual(disk["http_pending_results"]["4"], "write_file evidence")
                        return name + " evidence"

                    with (
                        mock.patch.object(oa, "current_run", return_value=observed),
                        mock.patch.object(oa, "_stream_completion", side_effect=stream),
                        mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
                        mock.patch.object(execution_state, "_write_snapshot", side_effect=capture),
                    ):
                        await self.run_backend(workspace, prompt="synthetic request " + "x" * 20_000)
                    fields[mode] = observed.submitted_fields
                    ordered = [message for message in record.get("http_messages") if message["role"] == "tool"]
                    self.assertEqual([message["tool_call_id"] for message in ordered],
                                     [call["id"] for call in calls])
                    self.assertEqual(record.get("http_pending_results"), {})
        self.assertEqual(snapshots["candidate"], snapshots["legacy"])
        self.assertEqual(len(snapshots["candidate"]), 8)
        self.assertEqual(fields["candidate"].count({"http_pending_results"}), 4)
        self.assertEqual(fields["candidate"].count(_TRANSCRIPT_FIELDS), 4)

    async def test_cancelled_read_wave_persists_only_outcomes_and_recovers_finished_evidence(self):
        waiting = asyncio.Event()
        sibling_settled = asyncio.Event()
        calls = [_call("finished"), _call("waiting")]
        requests = []
        with tempfile.TemporaryDirectory() as workspace:
            async with execution_state.execution_scope(workspace, "read-job", "synthetic request"):
                record = await execution_state.record("offline-checkpoint")
                observed = _ObservedRecord(record)

                async def execute(_name, _args, *_unused):
                    if not waiting.is_set():
                        waiting.set()
                        return "completed evidence"
                    try:
                        await asyncio.Event().wait()
                    finally:
                        sibling_settled.set()

                with (
                    mock.patch.object(oa, "current_run", return_value=observed),
                    mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                      return_value=("", "", calls)),
                    mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
                ):
                    owner = asyncio.create_task(self.run_backend(workspace))
                    await asyncio.wait_for(waiting.wait(), 2)
                    await asyncio.sleep(0)
                    owner.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await owner
                self.assertTrue(sibling_settled.is_set())
                self.assertEqual(observed.submitted_fields[-1], {"http_pending_results"})
                saved = json.loads(record.path.read_bytes())["values"]
                self.assertEqual(saved["http_pending_results"], {"0": "completed evidence"})

            async def resume(*args, **_kwargs):
                requests.extend(copy.deepcopy(args[1]["messages"]))
                return "verified saved evidence", "", []

            async with execution_state.execution_scope(workspace, "read-job", "synthetic request"):
                record = await execution_state.record("offline-checkpoint")
                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", side_effect=resume),
                    mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute_tool,
                ):
                    await self.run_backend(workspace)
                execute_tool.assert_not_awaited()
        recovered = [message for message in requests if message["role"] == "tool"]
        self.assertEqual(recovered[0]["content"], "completed evidence")
        self.assertIn("outcome is unknown", recovered[1]["content"])

    async def test_repeated_cancel_settles_pending_write_then_recovers_without_replaying_mutation(self):
        entered = threading.Event()
        release = threading.Event()
        settled = threading.Event()
        calls = [_call("write", "write_file"), _call("next", "bash")]
        mutations = []
        original_write = execution_state._write_snapshot
        requests = []
        with tempfile.TemporaryDirectory() as workspace:
            async with execution_state.execution_scope(workspace, "write-job", "synthetic request"):
                record = await execution_state.record("offline-checkpoint")
                observed = _ObservedRecord(record)

                def block_pending(path, snapshot):
                    if snapshot["values"]["http_pending_results"]:
                        entered.set()
                        if not release.wait(5):
                            raise AssertionError("fixture did not release checkpoint write")
                    try:
                        original_write(path, snapshot)
                    finally:
                        if snapshot["values"]["http_pending_results"]:
                            settled.set()

                async def execute(name, _args, *_unused):
                    mutations.append(name)
                    Path(workspace, "artifact.txt").write_text("saved once", encoding="utf-8")
                    return "mutation completed"

                with (
                    mock.patch.object(oa, "current_run", return_value=observed),
                    mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                      return_value=("", "", calls)),
                    mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
                    mock.patch.object(execution_state, "_write_snapshot", side_effect=block_pending),
                ):
                    owner = asyncio.create_task(self.run_backend(workspace))
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                        owner.cancel("first")
                        await asyncio.sleep(0)
                        owner.cancel("second")
                        for _ in range(10):
                            await asyncio.sleep(0)
                        self.assertFalse(owner.done())
                        self.assertFalse(settled.is_set())
                        self.assertEqual(mutations, ["write_file"])
                        release.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(owner, 3)
                    finally:
                        release.set()
                        await asyncio.gather(owner, return_exceptions=True)
                self.assertTrue(settled.is_set())
                self.assertEqual(observed.submitted_fields[-1], {"http_pending_results"})

            async def resume(*args, **_kwargs):
                requests.extend(copy.deepcopy(args[1]["messages"]))
                return "existing artifact verified", "", []

            async with execution_state.execution_scope(workspace, "write-job", "synthetic request"):
                record = await execution_state.record("offline-checkpoint")
                with (
                    mock.patch.object(oa, "current_run", return_value=record),
                    mock.patch.object(oa, "_stream_completion", side_effect=resume),
                    mock.patch.object(oa.tools, "execute_tool", new_callable=mock.AsyncMock) as execute_tool,
                ):
                    await self.run_backend(workspace)
                execute_tool.assert_not_awaited()
            self.assertEqual(Path(workspace, "artifact.txt").read_text(encoding="utf-8"), "saved once")
        recovered = [message for message in requests if message["role"] == "tool"]
        self.assertEqual(recovered[0]["content"], "mutation completed")
        self.assertIn("outcome is unknown", recovered[1]["content"])
        self.assertEqual(mutations, ["write_file"])
