"""Durable retries retain semantic scope without recursively saving resume envelopes."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.tests.test_reply_delivery import _ReplyDeliveryBot


class CheckpointScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_real_allowance_boundaries_keep_canonical_request_and_evidence(self):
        with tempfile.TemporaryDirectory() as ws, mock.patch.object(workspace, "CONFIG_DIR", ws):
            bot = _ReplyDeliveryBot(ws, fail_final=False)
            sid = session.create_session(ws, name="Remaining duplicate cleanup")["id"]
            original = "Remove all duplication, preserve the documented behavior, verify and commit the result."
            addition = "Also retain the public command names."
            entry = await bot._persist_enqueue("u1", original, "chat")
            bot._ensure_message_queue("u1").put_nowait((original, "chat", entry, False))
            await bot._save_job_checkpoint("u1", entry, {
                "state": "running", "request": original, "workspace_path": ws, "session_id": sid,
            })
            prompts = []
            saved_jobs = []

            async def execute(turn, prompt, *_args, **_kwargs):
                prompts.append(prompt)
                turn.session_id = sid
                saved_jobs.append(await bot._load_job_checkpoint("u1", entry))
                if len(prompts) <= 3:
                    budget = agent._TURN_BUDGET.get()
                    budget.reports = [(f"Verified scope {len(prompts)}", f"Evidence artifact {len(prompts)} passed")]
                    if len(prompts) == 2:
                        turn.injected.append(addition)
                    raise agent._TurnBudgetExhaustedError("synthetic execution boundary")
                return AgentResult(
                    text="Original work completed", completion_verified=True, session_id=sid,
                    events=[ChatEvent(kind="text", content="Original work completed")],
                )

            with (
                mock.patch.object(agent, "_run_turn_impl", execute),
                mock.patch.object(workspace, "get_run_config", return_value=("meta", None, "summary", "full", "meta")),
            ):
                await asyncio.wait_for(bot._drain_message_queue("u1"), timeout=3)
                await agent._wait_for_pending_session_writes(ws)
            self.assertEqual(len(prompts), 4)
            self.assertEqual(prompts[0], original)
            for index, prompt in enumerate(prompts[1:], start=1):
                self.assertEqual(prompt.count(original), 1)
                self.assertEqual(prompt.count("[Resume the SAME unfinished job"), 1)
                self.assertEqual(prompt.count("[Saved progress checkpoint]"), 1)
                pending = saved_jobs[index]["checkpoint"]["pending"]
                self.assertEqual(pending, [agent._DEFAULT_PENDING_SCOPE])
                self.assertNotIn("[Resume the SAME unfinished job", "\n".join(pending))
                self.assertNotIn("[Saved progress checkpoint]", "\n".join(pending))
                self.assertLess(len(prompt), 6000)
            final_saved = saved_jobs[-1]
            self.assertTrue(final_saved["request"].startswith(original))
            self.assertEqual(final_saved["request"].count(addition), 1)
            self.assertIn(addition, prompts[-1])
            self.assertEqual([row["report"] for row in final_saved["checkpoint"]["reports"]], [
                "Evidence artifact 1 passed", "Evidence artifact 2 passed", "Evidence artifact 3 passed",
            ])
            self.assertEqual(bot._read_queue_file(), {})

    async def test_unavailable_planner_fallback_does_not_store_full_resume_as_scope(self):
        with tempfile.TemporaryDirectory() as ws:
            for tier in ("low", "mid", "high"):
                workspace.set_flexible_backend_name(ws, tier, "meta")
            original = "Finish the original duplicate cleanup and verification."
            request = original + (
                "\n\n[Resume the SAME unfinished job; later queued jobs remain held.]\n"
                "[Saved progress checkpoint]:\n"
                '{"pending":["Verify the retained repairs"],"reports":[]}'
            )

            async def worker(*_args, **_kwargs):
                self.assertEqual(agent._TURN_BUDGET.get().pending, [agent._DEFAULT_PENDING_SCOPE])
                return AgentResult(text="Source repair evidence retained", error="Tool budget exhausted after 40 tool rounds."), False

            with agent.turn_budget(fresh=True) as budget:
                with (
                    mock.patch.object(agent, "run_internal_backend", mock.AsyncMock(return_value=None)),
                    mock.patch.object(agent, "_drive_backend", worker),
                    self.assertRaises(agent._TurnBudgetExhaustedError),
                ):
                    await agent._run_flexible(
                        request, request, ws, approval="full", effort=100,
                        collaborative=False, summary_backend_name="meta", summary_model="summary",
                        on_event=None, inject_queue=None, injected=[],
                    )
                checkpoint = budget.checkpoint("tool allowance", agent._InterruptedTurn(ws, request))
            self.assertEqual(checkpoint["pending"], [agent._DEFAULT_PENDING_SCOPE])
            self.assertEqual(checkpoint["reports"][0]["instruction"], agent._DEFAULT_PENDING_SCOPE)
            self.assertIn("Source repair evidence retained", checkpoint["reports"][0]["report"])
            self.assertEqual(checkpoint["request"], request)

    def test_checkpoint_sanitizes_only_exact_transport_values_and_preserves_user_text(self):
        original = (
            "Keep the literal documentation examples '[Resume the SAME unfinished job' "
            "and '[Saved progress checkpoint]' unchanged in the user guide."
        )
        semantic_instruction = "Verify the '[Saved progress checkpoint]' documentation example."
        report = "Verified the literal '[Resume the SAME unfinished job' example."
        budget = agent.TurnBudget()
        budget.pending = [original, semantic_instruction]
        budget.reports = [(original, report), (semantic_instruction, report)]
        turn = agent._InterruptedTurn("C:/example", original, injected=["Preserve both examples."])
        checkpoint = budget.checkpoint("execution boundary", turn)
        self.assertEqual(checkpoint["request"], original)
        self.assertEqual(checkpoint["pending"], [agent._DEFAULT_PENDING_SCOPE, semantic_instruction])
        self.assertEqual(checkpoint["reports"][0]["instruction"], agent._DEFAULT_PENDING_SCOPE)
        self.assertEqual(checkpoint["reports"][1]["instruction"], semantic_instruction)
        self.assertEqual([row["report"] for row in checkpoint["reports"]], [report, report])
        self.assertEqual(checkpoint["accepted_additions"], ["Preserve both examples."])

