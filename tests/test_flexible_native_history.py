"""Shared Cozter context reaches isolated Flexible native conversations once."""

import asyncio
from collections import Counter, defaultdict
import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from Cozter import agent, execution_state, flexible
from Cozter.backends_agent.base import AgentResult


class FlexibleNativeHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.native = SimpleNamespace(
            name="codex", default_model="native-model", supports_read_only_workers=True,
            supports_typed_plugins=True, supports_plugin_prelude=False, supports_detached_tasks=False,
        )
        self.http = SimpleNamespace(
            name="meta", supports_read_only_workers=True, supports_typed_plugins=True,
            supports_plugin_prelude=False, supports_detached_tasks=False,
        )
        self.summary = SimpleNamespace(name="summary")
        backends = {"codex": self.native, "meta": self.http, "summary": self.summary}
        self.enterContext(mock.patch.object(
            agent.backends_agent, "get_backend", side_effect=lambda name: backends[name],
        ))
        self.tiers = {tier: ("codex", "native-model") for tier in flexible.TIERS}
        self.enterContext(mock.patch.object(
            agent.workspace_mod, "get_flexible_run_config", side_effect=lambda _ws: dict(self.tiers),
        ))
        self.enterContext(mock.patch.object(agent.config, "get_flexible_max_parallel_workers", return_value=3))
        self.enterContext(mock.patch.object(agent.skills, "current_catalog", return_value=None))
        self.enterContext(mock.patch.object(agent.skills, "catalog_prompt", return_value=""))
        self.enterContext(mock.patch.object(
            agent, "_build_backend_prompt", side_effect=lambda _backend, prompt, **_kwargs: "POLICY\n" + prompt,
        ))
        self.enterContext(mock.patch.object(agent, "run_internal_backend", side_effect=self._internal))
        self.plan = "[PLAN]\n1. [high] Repair module A\n[/PLAN]"
        self.planner_calls = 0
        self.merge_prompts = []
        self.calls = []
        self.data = {
            "summary": "SHARED-SUMMARY-MARKER", "long_term": ["SHARED-MEMORY-MARKER"],
            "messages": [
                {"role": "user", "content": "OLD-SHARED-REQUEST-MARKER"},
                {"role": "assistant", "content": "OLD-SHARED-ANSWER-MARKER"},
            ], "compacted_count": 0,
        }
        self.colony = ["SHARED-COLONY-MARKER"]

    async def _internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            self.planner_calls += 1
            return self.plan
        self.merge_prompts.append(prompt)
        return "MERGED-FINAL-ANSWER-MARKER"

    @staticmethod
    def _index(prompt: str) -> int:
        return int(prompt.split("[Sub-task ")[1].split("/")[0])

    async def _invoke(self, drive, *, job="job", request="CURRENT-REQUEST-MARKER", budget=20_000):
        async with execution_state.execution_scope(self.ws, job, "canonical request"):
            execution_state.bind_conversation("shared-conversation")
            with mock.patch.object(agent, "_drive_backend", side_effect=drive):
                return await agent._run_flexible(
                    "planner history\n" + request, request, self.ws, approval="full", effort=50,
                    collaborative=False, summary_backend_name="summary", summary_model="summary-model",
                    on_event=None, inject_queue=None, injected=[], session_data=copy.deepcopy(self.data),
                    colony_items=list(self.colony), history_budget=budget,
                )

    async def _native_drive(self, backend, _ws, prompt, model, approval, **kwargs):
        context = kwargs["native_context"]
        self.assertIsNotNone(context)
        async with execution_state.backend_scope(
            backend.name, model, approval, kwargs["effort"], native=True,
            native_worker=kwargs.get("native_worker"),
        ) as record:
            self.assertIsNotNone(record)
            rendered = context.render(record)
            self.calls.append({
                "index": self._index(prompt), "prompt": rendered, "record": record,
                "context": context, "approval": approval,
            })
            native_id = record.get("native_session_id") or str(uuid.uuid4())
            await record.save(native_session_id=native_id, native_tools_started=True)
            await context.acknowledge(record)
            return AgentResult(text=f"worker-report-{self._index(prompt)}"), False

    def _append_external_turn(self) -> None:
        self.data["messages"].extend([
            {"role": "user", "content": "INTERVENING-SHARED-REQUEST-MARKER"},
            {"role": "assistant", "content": "INTERVENING-MERGED-ANSWER-MARKER"},
        ])

    async def test_serial_writer_roles_reuse_compatible_lane_and_receive_unseen_merged_history(self) -> None:
        await self._invoke(self._native_drive, job="first")
        first = self.calls[-1]
        for marker in ("SHARED-SUMMARY-MARKER", "SHARED-MEMORY-MARKER", "SHARED-COLONY-MARKER",
                       "OLD-SHARED-REQUEST-MARKER", "OLD-SHARED-ANSWER-MARKER"):
            self.assertIn(marker, first["prompt"])
        self.assertEqual(set(first["record"].get("native_prompt_ledger")["messages"]), {"0", "1"})
        self._append_external_turn()
        self.plan = "[PLAN]\n1. [high] Repair module B\n[/PLAN]"

        await self._invoke(self._native_drive, job="second", request="NEXT-REQUEST-MARKER")

        later = self.calls[-1]
        self.assertEqual(later["record"].path, first["record"].path)
        self.assertEqual(later["record"].get("native_session_id"), first["record"].get("native_session_id"))
        self.assertIn("INTERVENING-SHARED-REQUEST-MARKER", later["prompt"])
        self.assertIn("INTERVENING-MERGED-ANSWER-MARKER", later["prompt"])
        self.assertIn("Repair module B", later["prompt"])
        self.assertIn("NEXT-REQUEST-MARKER", later["prompt"])
        self.assertNotIn("OLD-SHARED-REQUEST-MARKER", later["prompt"])
        self.assertNotIn("OLD-SHARED-ANSWER-MARKER", later["prompt"])
        self.assertNotIn("SHARED-SUMMARY-MARKER", later["prompt"])
        self.assertNotIn("SHARED-MEMORY-MARKER", later["prompt"])
        self.assertNotIn("SHARED-COLONY-MARKER", later["prompt"])

    async def test_parallel_inspectors_track_shared_message_delivery_independently(self) -> None:
        self.plan = ("[PLAN]\n1. [high] [parallel-read] Inspect transport logs\n"
                     "2. [high] [parallel-read] Inspect parser code\n[/PLAN]")
        await self._invoke(self._native_drive, job="first")
        initial = {call["index"]: call for call in self.calls}
        self.assertNotEqual(initial[1]["record"].path, initial[2]["record"].path)
        self._append_external_turn()

        await self._invoke(self._native_drive, job="second", request="NEXT-REQUEST-MARKER")

        self.assertEqual(len(self.calls), 4)
        for call in self.calls[2:]:
            previous = initial[call["index"]]
            self.assertEqual(call["record"].path, previous["record"].path)
            self.assertEqual(call["approval"], "confirm")
            self.assertIn("INTERVENING-SHARED-REQUEST-MARKER", call["prompt"])
            self.assertIn("INTERVENING-MERGED-ANSWER-MARKER", call["prompt"])
            self.assertNotIn("OLD-SHARED-REQUEST-MARKER", call["prompt"])
            self.assertNotIn("OLD-SHARED-ANSWER-MARKER", call["prompt"])

    async def test_shared_context_is_budgeted_without_clipping_current_worker_scope(self) -> None:
        self.data.update(
            summary="SUMMARY-BUDGET-MARKER " + "s" * 3_000,
            long_term=["MEMORY-BUDGET-MARKER " + "m" * 3_000],
            messages=[{"role": "user", "content": f"message-{index} " + "x" * 240}
                      for index in range(15)],
        )
        self.colony = ["COLONY-BUDGET-MARKER " + "c" * 3_000]
        await self._invoke(self._native_drive, budget=1_800)

        call = self.calls[-1]
        context = call["context"]
        shared_size = sum(len(context.sections.get(name, "")) for name in ("colony", "memory", "summary"))
        shared_size += sum(len(text) + 1 for _, _, text in context.messages)
        self.assertLessEqual(shared_size, 1_800)
        self.assertLess(len(context.messages), 15)
        self.assertIn("message-14", call["prompt"])
        self.assertIn("Repair module A", call["prompt"])
        self.assertTrue(call["prompt"].endswith("New message:\nCURRENT-REQUEST-MARKER"))
        self.assertTrue({"colony", "memory", "summary"} <= context.partial_sections)

    async def test_long_checkpoint_reserves_bounded_shared_history_and_resumes_with_only_new_messages(self) -> None:
        request = "LONG-CHECKPOINT-MARKER " + "checkpoint obligation " * 3_000
        self.data["messages"] = [
            {"role": "user", "content": "OLDER-LARGE-HISTORY-MARKER " + "x" * 8_000},
            {"role": "assistant", "content": "RECENT-SHARED-ANSWER-MARKER"},
        ]
        await self._invoke(self._native_drive, job="long-first", request=request, budget=2_000)
        first = self.calls[-1]
        context = first["context"]
        shared_size = sum(len(context.sections.get(name, "")) for name in ("colony", "memory", "summary"))
        shared_size += sum(len(text) + 1 for _, _, text in context.messages)
        self.assertTrue(context.messages)
        self.assertLessEqual(shared_size, 2_000)
        self.assertIn("RECENT-SHARED-ANSWER-MARKER", first["prompt"])
        self.assertTrue(first["prompt"].endswith("New message:\n" + request))

        self._append_external_turn()
        await self._invoke(self._native_drive, job="long-second", request=request, budget=2_000)
        later = self.calls[-1]
        self.assertEqual(later["record"].path, first["record"].path)
        self.assertNotIn("RECENT-SHARED-ANSWER-MARKER", later["prompt"])
        self.assertIn("INTERVENING-MERGED-ANSWER-MARKER", later["prompt"])
        self.assertTrue(later["prompt"].endswith("New message:\n" + request))

        direct = agent._native_turn_prompt(
            self.native, request, self.data, self.colony, budget=2_000,
            collaborative=False, is_scheduled=False,
        )
        self.assertEqual(direct.messages, [])
        self.assertTrue(direct.render(None).endswith("New message:\n" + request))

    async def test_normalized_duplicate_inspectors_do_not_overlap_but_other_reads_still_parallelize(self) -> None:
        self.plan = ("[PLAN]\n1. [high] [parallel-read] Please inspect logs\n"
                     "2. [high] [parallel-read] inspect logs\n"
                     "3. [high] [parallel-read] inspect settings\n[/PLAN]")
        active = Counter()
        peak = Counter()
        total_active = max_total = 0
        calls = []
        probe_keys = {}
        third_started = asyncio.Event()

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            nonlocal total_active, max_total
            index = self._index(prompt)
            role = "logs" if index in (1, 2) else "settings"
            calls.append(index)
            active[role] += 1
            peak[role] = max(peak[role], active[role])
            total_active += 1
            max_total = max(max_total, total_active)
            try:
                if index == 2:
                    await third_started.wait()
                elif index == 3:
                    third_started.set()
                await asyncio.sleep(0)
                probe = await execution_state.record("collision-probe")
                probe_keys[index] = probe.key
                return AgentResult(text="inspection complete"), False
            finally:
                active[role] -= 1
                total_active -= 1

        await asyncio.wait_for(self._invoke(drive), timeout=2)

        self.assertEqual(sorted(calls), [1, 2, 3])
        self.assertEqual(len(peak), 2)
        self.assertEqual(set(peak.values()), {1})
        self.assertEqual(max_total, 2)
        self.assertEqual(probe_keys[1], probe_keys[2])
        self.assertNotEqual(probe_keys[1], probe_keys[3])

    async def test_cancelled_unacknowledged_reader_resumes_without_replanning_or_replaying_cached_sibling(self) -> None:
        self.plan = ("[PLAN]\n1. [high] [parallel-read] Inspect completed component\n"
                     "2. [high] [parallel-read] Inspect pending component\n"
                     "3. [high] Apply verified repairs\n[/PLAN]")
        counts = Counter()
        rendered = defaultdict(list)
        interrupted_ledgers = []

        async def drive(backend, _ws, prompt, model, approval, **kwargs):
            index = self._index(prompt)
            counts[index] += 1
            context = kwargs["native_context"]
            async with execution_state.backend_scope(
                backend.name, model, approval, kwargs["effort"], native=True,
                native_worker=kwargs.get("native_worker"),
            ) as record:
                rendered[index].append(context.render(record))
                await record.save(native_session_id=record.get("native_session_id") or str(uuid.uuid4()),
                                  native_tools_started=False)
                if index == 2 and counts[index] == 1:
                    async with asyncio.timeout(2):
                        while True:
                            saved = await execution_state.current_execution().record({
                                "record": "flexible_plan", "worker": "main",
                            })
                            if saved.get("worker_outputs", {}).get("0", {}).get("complete"):
                                break
                            await asyncio.sleep(0)
                    await context.acknowledge(record)
                    interrupted_ledgers.append(record.get("native_prompt_ledger"))
                    raise asyncio.CancelledError
                await record.save(native_tools_started=True)
                await context.acknowledge(record)
                return AgentResult(text=f"retained-evidence-{index}"), False

        with self.assertRaises(asyncio.CancelledError):
            await self._invoke(drive, job="durable")
        self.assertEqual(interrupted_ledgers, [None])
        self._append_external_turn()

        result, restarting = await self._invoke(drive, job="durable", request="RESUME-SAME-WORK-MARKER")

        self.assertFalse(restarting)
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(counts, {1: 1, 2: 2, 3: 1})
        self.assertIn("OLD-SHARED-REQUEST-MARKER", rendered[2][1])
        self.assertIn("INTERVENING-MERGED-ANSWER-MARKER", rendered[2][1])
        self.assertIn("RESUME-SAME-WORK-MARKER", rendered[2][1])
        for index in (1, 2, 3):
            self.assertIn(f"retained-evidence-{index}", self.merge_prompts[-1])
        self.assertEqual(result.text, "MERGED-FINAL-ANSWER-MARKER")

    async def test_http_worker_keeps_existing_prompt_and_execution_contract(self) -> None:
        self.tiers = {tier: ("meta", "http-model") for tier in flexible.TIERS}

        async def drive(backend, _ws, prompt, _model, _approval, **kwargs):
            self.assertEqual(backend.name, "meta")
            self.assertIsNone(kwargs.get("native_context"))
            self.assertIsNone(kwargs.get("native_worker"))
            self.assertIn("CURRENT-REQUEST-MARKER", prompt)
            self.assertIn("Repair module A", prompt)
            self.assertNotIn("OLD-SHARED-REQUEST-MARKER", prompt)
            self.assertNotIn("SHARED-MEMORY-MARKER", prompt)
            return AgentResult(text="http report"), False

        result, restarting = await self._invoke(drive)

        self.assertFalse(restarting)
        self.assertEqual(result.text, "http report")
