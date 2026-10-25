"""Plugin work stays bounded as expressions and saved transcripts grow."""

import ast
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import calculator, memory


class CalculatorIntermediateBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_product_chain_stops_at_first_over_budget_intermediate(self) -> None:
        # Each power alone is within the existing 50,000-digit magnitude
        # policy. Previously multiplication grew four million digit integers
        # and blocked the event loop for seconds before output conversion failed.
        expression = "*".join(["99999**9999"] * 80)
        operation = mock.Mock(wraps=calculator._guarded_multiply)
        with mock.patch.dict(calculator._BIN_OPS, {ast.Mult: operation}):
            result = await calculator.CalculatorTool().run(".", {"expression": expression})
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("50,000 digits", result)
        self.assertEqual(operation.call_count, 1)

    async def test_large_valid_integer_remains_available_for_later_math(self) -> None:
        result = await calculator.CalculatorTool().run(".", {
            "expression": "99999**9999 % 97",
        })
        self.assertEqual(result, str(pow(99999, 9999, 97)))

    async def test_existing_power_magnitude_boundary_is_preserved(self) -> None:
        result = await calculator.CalculatorTool().run(".", {
            "expression": "(10**10000)**5 % 97",
        })
        self.assertEqual(result, str(pow(10, 50_000, 97)))

    async def test_addition_cannot_bypass_intermediate_magnitude_budget(self) -> None:
        result = await calculator.CalculatorTool().run(".", {
            "expression": "((10**10000)**5 + 1) % 97",
        })
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("50,000 digits", result)

    async def test_complex_intermediate_can_still_be_reduced_to_real_result(self) -> None:
        result = await calculator.CalculatorTool().run(".", {
            "expression": "abs((-1)**0.5)",
        })
        self.assertEqual(result, "1")


class MemoryDirectReadTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def write_session(directory: Path, session_id: str, *, name: str = "Fixture", created: str = "2026-10-15", content: str = "saved evidence", recorded_id: str | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        Path(directory, f"{session_id}.json").write_text(json.dumps({
            "id": recorded_id if recorded_id is not None else session_id,
            "name": name,
            "created": created,
            "messages": [{"role": "assistant", "content": content}],
        }), encoding="utf-8")

    async def test_exact_id_reads_only_its_own_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            for index in range(64):
                self.write_session(directory, f"{index:012x}", content="x" * 128_000)
            target = "ffffffffffff"
            self.write_session(directory, target, content="the selected evidence")
            with mock.patch.object(
                memory, "_load_session_file", wraps=memory._load_session_file,
            ) as load:
                result = await memory.MemoryTool().run(tmp, {"action": "read", "session": target})
            self.assertIn("the selected evidence", result)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(load.call_args.args[1], target)

    async def test_exact_id_still_wins_over_another_sessions_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            self.write_session(directory, "target", content="id evidence")
            self.write_session(directory, "other", name="target", content="name evidence")
            result = await memory.MemoryTool().run(tmp, {"action": "read", "session": "target"})
            self.assertIn("id evidence", result)
            self.assertNotIn("name evidence", result)

    async def test_newest_alias_retains_priority_over_same_named_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            self.write_session(directory, "last", created="2025-01-01", content="old evidence")
            self.write_session(directory, "newer", content="newest evidence")
            result = await memory.MemoryTool().run(tmp, {"action": "read", "session": "last"})
            self.assertIn("newest evidence", result)
            self.assertNotIn("old evidence", result)

    async def test_invalid_exact_file_keeps_name_fallback_and_identity_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            self.write_session(directory, "target", recorded_id="wrong", content="mismatched evidence")
            self.write_session(directory, "other", name="target", content="valid named evidence")
            result = await memory.MemoryTool().run(tmp, {"action": "read", "session": "target"})
            self.assertIn("valid named evidence", result)
            self.assertNotIn("mismatched evidence", result)

    async def test_exact_read_rejects_session_link_outside_memory_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            directory = Path(tmp, ".cozter", "sessions")
            directory.mkdir(parents=True)
            self.write_session(Path(outside), "escape", content="outside evidence")
            try:
                Path(directory, "escape.json").symlink_to(Path(outside, "escape.json"))
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            result = await memory.MemoryTool().run(tmp, {"action": "read", "session": "escape"})
            self.assertTrue(result.startswith("Error:"), result)
            self.assertNotIn("outside evidence", result)


class MemoryReadPagingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def write_messages(workspace: str, count: int, *, content_chars: int = 400, name: str = "Paging fixture") -> str:
        session_id = "fixture12345"
        directory = Path(workspace, ".cozter", "sessions")
        directory.mkdir(parents=True, exist_ok=True)
        Path(directory, f"{session_id}.json").write_text(json.dumps({
            "id": session_id,
            "name": name,
            "created": "2026-10-15",
            "messages": [
                {"role": "assistant", "content": f"message-{number} " + "x" * content_chars}
                for number in range(1, count + 1)
            ],
        }), encoding="utf-8")
        return session_id

    async def test_budget_limited_pages_recover_every_message_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_id = self.write_messages(tmp, 19)
            tool = memory.MemoryTool()
            offset = 0
            seen: list[int] = []
            page_sizes: list[int] = []
            for _ in range(10):
                output = await tool.run(tmp, {
                    "action": "read", "session": session_id, "limit": 100, "offset": offset,
                })
                self.assertLessEqual(len(output), memory._OUTPUT_BUDGET)
                indices = [int(value) for value in re.findall(r"^(\d+)\. Assistant:", output, re.MULTILINE)]
                self.assertTrue(indices, output)
                self.assertEqual(indices, sorted(indices))
                self.assertEqual(indices[-1], 19 - offset)
                self.assertIn("[line clipped]", output)
                self.assertIn(
                    f"showing last {len(indices)}" if offset == 0 else f"showing {len(indices)} older message(s)",
                    output,
                )
                self.assertFalse(set(indices).intersection(seen))
                seen.extend(indices)
                page_sizes.append(len(indices))
                continuation = re.search(r"\boffset=(\d+)\b", output)
                if continuation is None:
                    break
                self.assertIn("Older messages remain", output)
                self.assertEqual(int(continuation.group(1)), offset + len(indices))
                offset = int(continuation.group(1))
            else:
                self.fail("Memory pages did not reach the oldest message")
            self.assertEqual(sorted(seen), list(range(1, 20)))
            self.assertGreater(len(page_sizes), 1)
            self.assertLess(page_sizes[0], 19)

    async def test_default_small_read_keeps_existing_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_id = self.write_messages(tmp, 3, content_chars=0)
            output = await memory.MemoryTool().run(tmp, {
                "action": "read", "session": session_id,
            })
            self.assertIn("3 message(s), showing last 3)", output)
            self.assertIn("1. Assistant: message-1", output)
            self.assertIn("3. Assistant: message-3", output)
            self.assertNotIn("offset=", output)

    async def test_offset_beyond_end_has_no_repeating_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_id = self.write_messages(tmp, 3)
            for offset in (3, 4, 10 ** 5_000):
                output = await memory.MemoryTool().run(tmp, {
                    "action": "read", "session": session_id, "offset": offset,
                })
                self.assertIn("No older messages", output)
                self.assertNotIn("offset=", output)
                self.assertLessEqual(len(output), memory._OUTPUT_BUDGET)

    async def test_invalid_offsets_are_rejected_before_loading_transcripts(self) -> None:
        for offset in (None, True, False, -1, 1.5, "1", [], float("inf")):
            with self.subTest(offset=offset), mock.patch.object(
                memory, "_load_session_file", side_effect=AssertionError("invalid offset must not read"),
            ):
                result = memory.MemoryTool()._read("unused", {"session": "fixture", "offset": offset})
            self.assertTrue(result.startswith("Error:"), result)
            self.assertIn("non-negative integer", result)

    async def test_oversized_metadata_does_not_offer_non_advancing_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_id = self.write_messages(tmp, 3, name="x" * 4_000)
            output = await memory.MemoryTool().run(tmp, {
                "action": "read", "session": session_id, "limit": 100,
            })
            self.assertTrue(output.startswith("Error:"), output[:100])
            self.assertIn("shorten the session name", output)
            self.assertNotIn("offset=", output)
            self.assertLessEqual(len(output), memory._OUTPUT_BUDGET)


if __name__ == "__main__":
    unittest.main()
