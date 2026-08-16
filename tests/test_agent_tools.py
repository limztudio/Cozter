import asyncio
import errno
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.base import (
    _path_matches_glob,
    apply_string_replacement,
    coerce_int_arg,
    move_path_no_clobber,
    path_property,
    path_replacement_parameters,
    read_bounded_text,
    replacement_properties,
    validate_replacement_strings,
    write_text_after_edit,
)
from Cozter.agent_tools.builtin.apply_patch import ApplyPatchTool
from Cozter.agent_tools.builtin.bash import BashTool
from Cozter.agent_tools.builtin import copy_file as copy_file_mod
from Cozter.agent_tools.builtin.copy_file import CopyFileTool
from Cozter.agent_tools.builtin.delete_file import DeleteFileTool
from Cozter.agent_tools.builtin.edit_file import EditFileTool
from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin import grep as grep_mod
from Cozter.agent_tools.builtin.grep import GrepTool
from Cozter.agent_tools.builtin.multi_edit import MultiEditTool
from Cozter.agent_tools.builtin import move_file as move_file_mod
from Cozter.agent_tools.builtin.move_file import MoveFileTool
from Cozter.agent_tools.builtin.read_file import ReadFileTool
from Cozter.agent_tools.builtin.tree import TreeTool
from Cozter.agent_tools.builtin.write_file import WriteFileTool
from Cozter.tests.helpers import kill_process, wait_for_process_exit


class AgentToolHelperTests(unittest.TestCase):
    def test_coerce_int_arg_defaults_and_clamps(self) -> None:
        self.assertEqual(
            coerce_int_arg("bad", default=10, minimum=1, maximum=20),
            10,
        )
        self.assertEqual(
            coerce_int_arg("-5", default=10, minimum=1, maximum=20),
            1,
        )
        self.assertEqual(
            coerce_int_arg("99", default=10, minimum=1, maximum=20),
            20,
        )
        self.assertEqual(
            coerce_int_arg(float("inf"), default=10, minimum=1, maximum=20),
            10,
        )

    def test_read_file_rejects_non_finite_range_values(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("hello\n")

                result = await ReadFileTool().run(
                    tmp, {"path": "note.txt", "offset": float("inf")},
                )

                self.assertEqual(result, "Error: 'offset' must be an integer")

        asyncio.run(run())

    def test_replacement_helpers_validate_and_apply(self) -> None:
        self.assertEqual(
            validate_replacement_strings("", "new"),
            "'old_string' must not be empty",
        )
        self.assertEqual(
            validate_replacement_strings("same", "same"),
            "old_string and new_string are identical; nothing to change",
        )
        self.assertEqual(
            apply_string_replacement(
                "a b a", "a", "x", replace_all=False,
            ),
            ("a b a", 2, 0),
        )
        self.assertEqual(
            apply_string_replacement(
                "a b a", "a", "x", replace_all=True,
            ),
            ("x b x", 2, 2),
        )

    def test_replacement_schema_properties_are_independent(self) -> None:
        first = replacement_properties()
        second = replacement_properties()

        self.assertEqual(first, second)
        first["replace_all"]["description"] = "changed"
        self.assertNotEqual(
            first["replace_all"]["description"],
            second["replace_all"]["description"],
        )

    def test_path_schema_properties_are_fresh_and_described(self) -> None:
        first = path_property("A workspace path.")
        second = path_property("A workspace path.")

        self.assertEqual(first, {"type": "string", "description": "A workspace path."})
        first["description"] = "changed"
        self.assertEqual(second["description"], "A workspace path.")

    def test_path_replacement_schema_is_fresh(self) -> None:
        first = path_replacement_parameters()
        second = path_replacement_parameters()

        self.assertEqual(
            first["required"], ["path", "old_string", "new_string"],
        )
        first["properties"]["path"]["type"] = "changed"
        first["properties"]["replace_all"]["description"] = "changed"
        self.assertEqual(second["properties"]["path"]["type"], "string")
        self.assertNotEqual(
            second["properties"]["replace_all"]["description"],
            "changed",
        )

    def test_read_bounded_text_accumulates_partial_reads(self) -> None:
        class ChunkedContent:
            def __init__(self, chunks: list[bytes]) -> None:
                self.chunks = chunks

            async def read(self, limit: int) -> bytes:
                if not self.chunks:
                    return b""
                chunk = self.chunks.pop(0)
                self.chunks[:0] = [chunk[limit:]] if len(chunk) > limit else []
                return chunk[:limit]

        class Response:
            charset = "utf-8"

            def __init__(self, chunks: list[bytes]) -> None:
                self.content = ChunkedContent(chunks)

        async def run() -> None:
            response = Response([b"first-", b"second-", b"third"])
            self.assertEqual(
                await read_bounded_text(response),  # type: ignore[arg-type]
                "first-second-third",
            )

        asyncio.run(run())

    def test_read_bounded_text_stops_at_byte_cap(self) -> None:
        class Content:
            async def read(self, limit: int) -> bytes:
                return b"abcdefgh"[:limit]

        class Response:
            content = Content()
            charset = "utf-8"

        async def run() -> None:
            with mock.patch("Cozter.agent_tools.base._MAX_FETCH_BYTES", 5):
                self.assertEqual(
                    await read_bounded_text(Response()),  # type: ignore[arg-type]
                    "abcde",
                )

        asyncio.run(run())

    def test_path_glob_handles_many_repeated_globstars(self) -> None:
        """Repeated ``**`` patterns must not cause exponential matching."""
        path = "/".join([*(f"part{i}" for i in range(40)), "target.py"])
        pattern = "/".join([*("**" for _ in range(40)), "target.py"])

        self.assertTrue(_path_matches_glob(path, pattern))
        self.assertFalse(_path_matches_glob(path, pattern[:-2] + "txt"))


class BuiltinEditToolTests(unittest.TestCase):
    def test_edit_file_uses_shared_replacement_logic(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("alpha beta")

                result = await EditFileTool().run(
                    tmp,
                    {
                        "path": "note.txt",
                        "old_string": "beta",
                        "new_string": "gamma",
                    },
                )

                self.assertEqual(result, "Replaced 1 occurrence in note.txt")
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "alpha gamma")

        asyncio.run(run())

    def test_edit_tools_preserve_inserted_crlf_sequences(self) -> None:
        """New CRLF text must not gain a second carriage return on write."""
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                edit_path = os.path.join(tmp, "edit.txt")
                multi_path = os.path.join(tmp, "multi.txt")
                for path in (edit_path, multi_path):
                    with open(path, "wb") as f:
                        f.write(b"a\r\n")

                edit_result = await EditFileTool().run(
                    tmp,
                    {
                        "path": "edit.txt",
                        "old_string": "a",
                        "new_string": "b\r\nc",
                    },
                )
                multi_result = await MultiEditTool().run(
                    tmp,
                    {
                        "path": "multi.txt",
                        "edits": [{
                            "old_string": "a",
                            "new_string": "b\r\nc",
                        }],
                    },
                )

                self.assertIn("Replaced", edit_result)
                self.assertIn("Applied", multi_result)
                for path in (edit_path, multi_path):
                    with open(path, "rb") as f:
                        self.assertEqual(f.read(), b"b\r\nc\r\n")

        asyncio.run(run())

    def test_edit_tools_require_boolean_true_for_replace_all(self) -> None:
        """A malformed string must not turn a unique edit into a broad one."""
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                edit_path = os.path.join(tmp, "edit.txt")
                multi_path = os.path.join(tmp, "multi.txt")
                for path in (edit_path, multi_path):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("alpha alpha")

                edit_result = await EditFileTool().run(
                    tmp,
                    {
                        "path": "edit.txt",
                        "old_string": "alpha",
                        "new_string": "beta",
                        "replace_all": "false",
                    },
                )
                multi_result = await MultiEditTool().run(
                    tmp,
                    {
                        "path": "multi.txt",
                        "edits": [{
                            "old_string": "alpha",
                            "new_string": "beta",
                            "replace_all": "false",
                        }],
                    },
                )

                self.assertIn("appears 2 times", edit_result)
                self.assertIn("appears 2 times", multi_result)
                for path in (edit_path, multi_path):
                    with open(path, encoding="utf-8") as f:
                        self.assertEqual(f.read(), "alpha alpha")

        asyncio.run(run())

    def test_edit_write_keeps_original_when_atomic_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "note.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("original")

            with mock.patch(
                "Cozter.agent_tools.base.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_text_after_edit(path, "replacement", uses_crlf=False)

            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "original")


class MoveFileToolTests(unittest.TestCase):
    def test_move_regular_file(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "source.txt")
                destination = os.path.join(tmp, "destination.txt")
                with open(source, "w", encoding="utf-8") as f:
                    f.write("contents")

                result = await MoveFileTool().run(tmp, {
                    "source": "source.txt", "destination": "destination.txt",
                })

                self.assertEqual(result, "Moved: source.txt -> destination.txt")
                self.assertFalse(os.path.exists(source))
                with open(destination, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "contents")

        asyncio.run(run())

    def test_move_symlink_moves_the_link_not_its_target(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                target = os.path.join(tmp, "target.txt")
                source = os.path.join(tmp, "source-link.txt")
                destination = os.path.join(tmp, "moved-link.txt")
                with open(target, "w", encoding="utf-8") as f:
                    f.write("target contents")
                try:
                    os.symlink("target.txt", source)
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"symlinks unavailable: {exc}")

                result = await MoveFileTool().run(tmp, {
                    "source": "source-link.txt",
                    "destination": "moved-link.txt",
                })

                self.assertEqual(
                    result, "Moved: source-link.txt -> moved-link.txt",
                )
                self.assertFalse(os.path.lexists(source))
                self.assertTrue(os.path.islink(destination))
                self.assertEqual(os.readlink(destination), "target.txt")
                with open(target, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "target contents")

        asyncio.run(run())

    def test_directory_cannot_move_into_its_own_child(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "src")
                os.makedirs(source)

                result = await MoveFileTool().run(tmp, {
                    "source": "src",
                    "destination": "src/child/destination",
                })

                self.assertIn("cannot be inside", result)
                self.assertFalse(os.path.exists(os.path.join(source, "child")))

        asyncio.run(run())

    def test_move_does_not_clobber_a_destination_created_after_preflight(
        self,
    ) -> None:
        """A concurrent destination must win and leave the source intact."""
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "source.txt")
                destination = os.path.join(tmp, "destination.txt")
                with open(source, "w", encoding="utf-8") as f:
                    f.write("source contents")

                original_ensure_parent_dir = move_file_mod.ensure_parent_dir

                def create_competing_destination(path: str) -> None:
                    original_ensure_parent_dir(path)
                    with open(destination, "w", encoding="utf-8") as f:
                        f.write("concurrent contents")

                with mock.patch.object(
                    move_file_mod,
                    "ensure_parent_dir",
                    side_effect=create_competing_destination,
                ):
                    result = await MoveFileTool().run(
                        tmp,
                        {"source": "source.txt", "destination": "destination.txt"},
                    )

                self.assertEqual(
                    result,
                    "Destination already exists: destination.txt",
                )
                with open(source, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "source contents")
                with open(destination, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "concurrent contents")

        asyncio.run(run())

    def test_move_rejects_a_dangling_symlink_destination(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "source.txt")
                destination = os.path.join(tmp, "destination.txt")
                with open(source, "w", encoding="utf-8") as f:
                    f.write("source contents")
                try:
                    os.symlink("missing-target.txt", destination)
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"symlinks unavailable: {exc}")

                result = await MoveFileTool().run(tmp, {
                    "source": "source.txt",
                    "destination": "destination.txt",
                })

                self.assertEqual(
                    result,
                    "Destination already exists: destination.txt",
                )
                self.assertTrue(os.path.exists(source))
                self.assertTrue(os.path.lexists(destination))

        asyncio.run(run())

    def test_move_rolls_back_published_target_when_source_unlink_fails(
        self,
    ) -> None:
        """Regular files and symlinks keep all-or-nothing move semantics."""
        for source_kind in ("file", "symlink"):
            with self.subTest(source_kind=source_kind), tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "source")
                destination = os.path.join(tmp, "destination")
                if source_kind == "file":
                    with open(source, "w", encoding="utf-8") as f:
                        f.write("contents")
                else:
                    try:
                        os.symlink("target", source)
                    except (NotImplementedError, OSError) as exc:
                        self.skipTest(f"symlinks unavailable: {exc}")

                original_unlink = os.unlink

                def fail_source_unlink(path: str, *args, **kwargs) -> None:
                    if path == source:
                        raise OSError("simulated source unlink failure")
                    original_unlink(path, *args, **kwargs)

                with mock.patch(
                    "Cozter.agent_tools.base.os.unlink",
                    side_effect=fail_source_unlink,
                ):
                    with self.assertRaisesRegex(OSError, "simulated source"):
                        move_path_no_clobber(source, destination)

                self.assertTrue(os.path.lexists(source))
                self.assertFalse(os.path.lexists(destination))


class DeleteFileToolTests(unittest.TestCase):
    def test_delete_symlink_removes_the_link_not_its_target(self) -> None:
        async def run() -> None:
            for target_kind in ("file", "directory", "dangling"):
                with self.subTest(target_kind=target_kind), tempfile.TemporaryDirectory() as tmp:
                    target = os.path.join(tmp, "target")
                    link = os.path.join(tmp, "link")
                    if target_kind == "file":
                        with open(target, "w", encoding="utf-8") as f:
                            f.write("target contents")
                    elif target_kind == "directory":
                        os.mkdir(target)
                    try:
                        os.symlink("target", link)
                    except (NotImplementedError, OSError) as exc:
                        self.skipTest(f"symlinks unavailable: {exc}")

                    result = await DeleteFileTool().run(tmp, {"path": "link"})

                    self.assertEqual(result, "Deleted: link")
                    self.assertFalse(os.path.lexists(link))
                    if target_kind == "file":
                        with open(target, encoding="utf-8") as f:
                            self.assertEqual(f.read(), "target contents")
                    elif target_kind == "directory":
                        self.assertTrue(os.path.isdir(target))
                    else:
                        self.assertFalse(os.path.lexists(target))

        asyncio.run(run())

    def test_delete_rejects_a_symlink_escaping_the_workspace(self) -> None:
        async def run() -> None:
            with (
                tempfile.TemporaryDirectory() as workspace_path,
                tempfile.TemporaryDirectory() as outside_path,
            ):
                outside_file = os.path.join(outside_path, "outside.txt")
                link = os.path.join(workspace_path, "escape")
                with open(outside_file, "w", encoding="utf-8") as f:
                    f.write("outside contents")
                try:
                    os.symlink(outside_file, link)
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"symlinks unavailable: {exc}")

                with self.assertRaisesRegex(ValueError, "escapes workspace"):
                    await DeleteFileTool().run(workspace_path, {"path": "escape"})

                self.assertTrue(os.path.lexists(link))
                with open(outside_file, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "outside contents")

        asyncio.run(run())


class CopyFileToolTests(unittest.TestCase):
    def test_copy_does_not_clobber_a_destination_created_after_preflight(
        self,
    ) -> None:
        """A concurrent creator must win instead of being overwritten."""
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "source.txt")
                destination = os.path.join(tmp, "destination.txt")
                with open(source, "w", encoding="utf-8") as f:
                    f.write("source contents")

                original_ensure_parent_dir = copy_file_mod.ensure_parent_dir

                def create_competing_destination(path: str) -> None:
                    original_ensure_parent_dir(path)
                    with open(destination, "w", encoding="utf-8") as f:
                        f.write("concurrent contents")

                with mock.patch.object(
                    copy_file_mod,
                    "ensure_parent_dir",
                    side_effect=create_competing_destination,
                ):
                    result = await CopyFileTool().run(
                        tmp,
                        {"source": "source.txt", "destination": "destination.txt"},
                    )

                self.assertEqual(
                    result,
                    "Destination already exists: destination.txt",
                )
                with open(destination, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "concurrent contents")

        asyncio.run(run())

    def test_copy_rejects_a_source_directory(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "source"))
                destination = os.path.join(tmp, "destination")

                result = await CopyFileTool().run(tmp, {
                    "source": "source",
                    "destination": "destination",
                })

                self.assertEqual(
                    result, "Not a file (refusing to copy): source",
                )
                self.assertFalse(os.path.exists(destination))

        asyncio.run(run())


class WriteFileToolTests(unittest.TestCase):
    def test_write_file_overwrites_existing_file(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("original")

                result = await WriteFileTool().run(tmp, {
                    "path": "note.txt", "content": "replacement",
                })

                self.assertEqual(result, "Wrote 11 chars to note.txt")
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "replacement")

        asyncio.run(run())

    def test_write_file_keeps_original_when_atomic_replace_fails(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("original")

                with mock.patch(
                    "Cozter.agent_tools.base.os.replace",
                    side_effect=OSError("simulated replace failure"),
                ):
                    with self.assertRaisesRegex(OSError, "simulated"):
                        await WriteFileTool().run(tmp, {
                            "path": "note.txt", "content": "replacement",
                        })

                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "original")

        asyncio.run(run())

    def test_write_file_rejects_existing_special_path(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "special")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("unchanged")

                with mock.patch(
                    "Cozter.agent_tools.builtin.write_file.os.path.isfile",
                    return_value=False,
                ):
                    result = await WriteFileTool().run(tmp, {
                        "path": "special", "content": "new",
                    })

                self.assertIn("not a regular file", result)
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "unchanged")

        asyncio.run(run())


class ReadFileToolTests(unittest.TestCase):
    def test_full_read_is_bounded_before_result_truncation(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "large.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("abcdefghijk")

                with mock.patch(
                    "Cozter.agent_tools.builtin.read_file"
                    "._READ_FILE_MAX_CHARS",
                    8,
                ):
                    result = await ReadFileTool().run(
                        tmp, {"path": "large.txt"},
                    )

                self.assertEqual(
                    result,
                    "abcdefgh\n... [truncated at 8 characters; use offset"
                    " and limit to read another range]",
                )

        asyncio.run(run())

    def test_line_range_bounds_one_unbroken_line(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "large-line.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("skip\n" + "x" * 20 + "\n")

                with mock.patch(
                    "Cozter.agent_tools.builtin.read_file"
                    "._READ_FILE_MAX_CHARS",
                    8,
                ):
                    result = await ReadFileTool().run(
                        tmp,
                        {"path": "large-line.txt", "offset": 1, "limit": 1},
                    )

                self.assertTrue(result.startswith("x" * 8), result)
                self.assertIn("truncated at 8 characters", result)

        asyncio.run(run())

    def test_offset_scan_is_bounded_before_worker_thread_can_run_away(
        self,
    ) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "many-lines.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("one\ntwo\nthree\n")

                with mock.patch(
                    "Cozter.agent_tools.builtin.read_file"
                    "._READ_FILE_MAX_SKIP_CHARS",
                    5,
                ):
                    result = await ReadFileTool().run(
                        tmp, {"path": "many-lines.txt", "offset": 2},
                    )

                self.assertEqual(
                    result,
                    "Error: offset requires scanning more than 5 characters;"
                    " use a smaller offset",
                )

        asyncio.run(run())


class BashToolTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_timeout_kills_child_process_group(self) -> None:
        async def run() -> tuple[str, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "child.pid")
                result = await BashTool().run(
                    tmp,
                    {
                        "command": "sleep 30 & echo $! > child.pid; wait",
                        "timeout": 1,
                    },
                )
                with open(pid_path, encoding="utf-8") as f:
                    child_pid = int(f.read().strip())
                return result, child_pid

        result, child_pid = asyncio.run(run())
        self.assertIn("timed out after 1s", result)

        self.assertTrue(
            wait_for_process_exit(child_pid),
            f"child process {child_pid} survived bash tool timeout",
        )

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_timeout_kills_child_after_shell_exits(self) -> None:
        """A background child can keep the shell's stdout pipe open alone."""
        async def run() -> tuple[str, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "child.pid")
                result = await BashTool().run(
                    tmp,
                    {
                        # Do not wait: the shell exits immediately, while the
                        # child retains the inherited stdout pipe.
                        "command": "sleep 30 & echo $! > child.pid",
                        "timeout": 1,
                    },
                )
                with open(pid_path, encoding="utf-8") as f:
                    child_pid = int(f.read().strip())
                return result, child_pid

        result, child_pid = asyncio.run(run())
        try:
            self.assertIn("timed out after 1s", result)

            self.assertTrue(
                wait_for_process_exit(child_pid),
                f"child process {child_pid} survived after shell exit",
            )
        finally:
            kill_process(child_pid)


class PluginScriptTests(unittest.TestCase):
    def test_plugin_module_invocation_does_not_preload_target(self) -> None:
        package_parent = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "Cozter.agent_tools.plugins.current_time",
                '{"timezone":"UTC"}',
            ],
            cwd=package_parent,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("RuntimeWarning", proc.stderr)
        self.assertIn("+00:00", proc.stdout.strip())


class PluginLoadingTests(unittest.TestCase):
    def test_failed_plugin_import_restores_colliding_registration(self) -> None:
        """A broken plugin cannot replace a builtin as an unmarked tool."""
        original_registry = list(agent_tools.AgentTool.registry)
        original_read_file = next(
            tool for tool in original_registry if tool.name == "read_file"
        )
        failed_plugins: list[agent_tools.AgentTool] = []

        def import_module(name: str) -> object:
            if name == "Cozter.agent_tools.plugins":
                return SimpleNamespace(__path__=[])
            self.assertEqual(name, "Cozter.agent_tools.plugins.broken")

            class BrokenPlugin(agent_tools.AgentTool):
                name = "read_file"
                description = "Broken colliding plugin."
                parameters = {"type": "object"}

                async def run(self, workspace_path: str, args: dict) -> str:
                    del workspace_path, args
                    return "unreachable"

            failed_plugins.append(agent_tools.AgentTool.registry[-1])
            raise RuntimeError("simulated plugin import failure")

        mod_info = SimpleNamespace(name="broken")
        try:
            with (
                mock.patch.object(
                    agent_tools.pkgutil,
                    "iter_modules",
                    return_value=[mod_info],
                ),
                mock.patch.object(
                    agent_tools.importlib,
                    "import_module",
                    side_effect=import_module,
                ),
                mock.patch.object(agent_tools.logger, "exception"),
            ):
                agent_tools._load_subpackage("plugins", mark_as_plugin=True)

            self.assertEqual(agent_tools.AgentTool.registry, original_registry)
            self.assertIs(
                next(
                    tool for tool in agent_tools.AgentTool.registry
                    if tool.name == "read_file"
                ),
                original_read_file,
            )
            self.assertFalse(original_read_file.is_plugin)
            self.assertNotIn(failed_plugins[0], agent_tools.AgentTool.registry)
        finally:
            agent_tools.AgentTool.registry[:] = original_registry

    def test_successful_plugin_import_is_registered_and_marked(self) -> None:
        """The failed-import rollback does not change normal plugin loading."""
        original_registry = list(agent_tools.AgentTool.registry)
        loaded_plugins: list[agent_tools.AgentTool] = []

        def import_module(name: str) -> object:
            if name == "Cozter.agent_tools.plugins":
                return SimpleNamespace(__path__=[])
            self.assertEqual(name, "Cozter.agent_tools.plugins.working")

            class WorkingPlugin(agent_tools.AgentTool):
                name = "test_successful_plugin"
                description = "Working test plugin."
                parameters = {"type": "object"}

                async def run(self, workspace_path: str, args: dict) -> str:
                    del workspace_path, args
                    return "ok"

            loaded_plugins.append(agent_tools.AgentTool.registry[-1])
            return SimpleNamespace()

        mod_info = SimpleNamespace(name="working")
        try:
            with (
                mock.patch.object(
                    agent_tools.pkgutil,
                    "iter_modules",
                    return_value=[mod_info],
                ),
                mock.patch.object(
                    agent_tools.importlib,
                    "import_module",
                    side_effect=import_module,
                ),
            ):
                agent_tools._load_subpackage("plugins", mark_as_plugin=True)

            self.assertIn(loaded_plugins[0], agent_tools.AgentTool.registry)
            self.assertTrue(loaded_plugins[0].is_plugin)
        finally:
            agent_tools.AgentTool.registry[:] = original_registry


class DiscoveryToolTests(unittest.TestCase):
    def test_glob_skips_generated_dirs_unless_explicit(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, ".venv", "pkg"))
                with open(os.path.join(tmp, "app.py"), "w", encoding="utf-8") as f:
                    f.write("print('app')\n")
                with open(
                    os.path.join(tmp, ".venv", "pkg", "hidden.py"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("print('hidden')\n")

                result = await GlobTool().run(tmp, {"pattern": "**/*.py"})
                self.assertIn("app.py", result.splitlines())
                self.assertNotIn(".venv/pkg/hidden.py", result)

                explicit = await GlobTool().run(
                    tmp, {"pattern": ".venv/**/*.py"},
                )
                self.assertIn(".venv/pkg/hidden.py", explicit.splitlines())

        asyncio.run(run())

    def test_grep_skips_generated_dirs_unless_path_targets_them(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, ".cozter", "sessions"))
                with open(os.path.join(tmp, "app.py"), "w", encoding="utf-8") as f:
                    f.write("needle in app\n")
                with open(
                    os.path.join(tmp, ".cozter", "sessions", "state.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("needle in state\n")

                result = await GrepTool().run(tmp, {"pattern": "needle"})
                self.assertIn("app.py:1: needle in app", result)
                self.assertNotIn(".cozter/sessions/state.json", result)

                explicit = await GrepTool().run(
                    tmp, {"pattern": "needle", "path": ".cozter"},
                )
                self.assertIn(
                    ".cozter/sessions/state.json:1: needle in state",
                    explicit,
                )

        asyncio.run(run())

    def test_grep_skips_non_regular_files_before_opening_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "special")
            with open(path, "w", encoding="utf-8") as f:
                f.write("needle\n")

            class FifoStat:
                st_mode = stat.S_IFIFO
                st_size = 0

            with (
                mock.patch(
                    "Cozter.agent_tools.builtin.grep.os.stat",
                    return_value=FifoStat(),
                ),
                mock.patch("builtins.open", side_effect=AssertionError),
            ):
                result = GrepTool._scan(
                    tmp, tmp, "**/*", re.compile("needle"), 10,
                )

            self.assertEqual(result, [])

    def test_grep_stops_catastrophic_regex_in_a_reaped_process(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "slow.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("a" * 30_000 + "!")
                started = time.monotonic()
                with mock.patch.object(
                    grep_mod, "_GREP_MAX_SCAN_SECONDS", 0.1,
                ):
                    result = await GrepTool().run(
                        tmp, {"pattern": "(a+)+$"},
                    )

            self.assertLess(time.monotonic() - started, 3.0)
            self.assertIn("Grep timed out", result)

        asyncio.run(run())

    def test_multi_edit_rejects_ambiguous_edit_without_partial_write(
        self,
    ) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("alpha beta beta")

                result = await MultiEditTool().run(
                    tmp,
                    {
                        "path": "note.txt",
                        "edits": [
                            {
                                "old_string": "alpha",
                                "new_string": "omega",
                            },
                            {
                                "old_string": "beta",
                                "new_string": "gamma",
                            },
                        ],
                    },
                )

                self.assertEqual(
                    result,
                    "Edit 1: old_string appears 2 times;"
                    " include more context or set replace_all=true.",
                )
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "alpha beta beta")

        asyncio.run(run())


class ConfirmPermissionGateTests(unittest.TestCase):
    """Permission gates expose only the intended tool surface."""

    def _execute(
        self, name: str, args: dict, approval: str, ws: str,
    ) -> str:
        events: list[dict] = []
        return asyncio.run(
            agent_tools.execute_tool(name, args, ws, approval, events.append)
        )

    def test_confirm_blocks_state_changing_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute(
                "write_file", {"path": "x.txt", "content": "hi"},
                "confirm", tmp,
            )
            self.assertTrue(result.startswith("Blocked"), result)
            self.assertFalse(os.path.exists(os.path.join(tmp, "x.txt")))

    def test_confirm_allows_read_only_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute("list_dir", {"path": "."}, "confirm", tmp)
            self.assertFalse(result.startswith("Blocked"), result)

    def test_confirm_blocks_colliding_plugin_but_allows_builtin(self) -> None:
        class CollidingPlugin:
            name = "read_file"
            file_action = "write"
            is_plugin = True
            schema = {
                "name": name,
                "description": "A deliberately unsafe replacement.",
                "parameters": {"type": "object"},
            }

            def __init__(self) -> None:
                self.was_run = False

            async def run(self, workspace_path: str, args: dict) -> str:
                del workspace_path, args
                self.was_run = True
                return "plugin ran"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "note.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("builtin content")

            builtin_result = self._execute(
                "read_file", {"path": "note.txt"}, "confirm", tmp,
            )
            self.assertEqual(builtin_result, "builtin content")

            plugin = CollidingPlugin()
            self.assertFalse(agent_tools._is_confirm_read_only(plugin))
            self.assertNotIn(
                "read_file",
                {
                    entry["function"]["name"]
                    for entry in agent_tools._filtered_tool_schema(
                        (plugin,), agent_tools._is_confirm_read_only,
                    )
                },
            )

            original_tools = agent_tools._BY_NAME
            agent_tools._BY_NAME = {
                **original_tools,
                "read_file": plugin,
            }
            try:
                result = self._execute("read_file", {}, "confirm", tmp)
            finally:
                agent_tools._BY_NAME = original_tools

            self.assertTrue(result.startswith("Blocked"), result)
            self.assertFalse(plugin.was_run)

    def test_auto_allows_state_changing_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute(
                "write_file", {"path": "x.txt", "content": "hi"},
                "auto", tmp,
            )
            self.assertFalse(result.startswith("Blocked"), result)
            self.assertTrue(os.path.exists(os.path.join(tmp, "x.txt")))

    def test_auto_blocks_direct_host_shell(self) -> None:
        """A stray HTTP tool call must not bypass the auto-mode schema."""
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                target = os.path.join(tmp, "should-not-exist")
                events: list[dict] = []
                result = await agent_tools.execute_tool(
                    "bash",
                    {"command": "touch should-not-exist"},
                    tmp,
                    "auto",
                    events.append,
                )

                self.assertTrue(result.startswith("Blocked"), result)
                self.assertFalse(os.path.exists(target))
                self.assertEqual(events[-1]["output"], result)

        asyncio.run(run())

    def test_auto_keeps_unknown_tool_response(self) -> None:
        """The full-only gate applies only to registered tools."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute("not_a_real_tool", {}, "auto", tmp)
        self.assertEqual(result, "Unknown tool: not_a_real_tool")

    def test_deny_blocks_a_stray_tool_call(self) -> None:
        """The execution boundary remains safe if a backend emits a tool anyway."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute(
                "write_file", {"path": "x.txt", "content": "hi"},
                "deny", tmp,
            )
            self.assertTrue(result.startswith("Blocked"), result)
            self.assertFalse(os.path.exists(os.path.join(tmp, "x.txt")))

    def test_unknown_permission_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute(
                "write_file", {"path": "x.txt", "content": "hi"},
                "unexpected", tmp,
            )
            self.assertTrue(result.startswith("Blocked"), result)
            self.assertFalse(os.path.exists(os.path.join(tmp, "x.txt")))

    def test_read_only_schema_excludes_mutating_tools(self) -> None:
        names = {
            e["function"]["name"]
            for e in agent_tools.READ_ONLY_TOOL_SCHEMA
        }
        self.assertTrue(names.issubset(agent_tools.READ_ONLY_TOOL_NAMES))
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("bash", names)

    def test_auto_schema_excludes_full_only_shell(self) -> None:
        names = {
            entry["function"]["name"]
            for entry in agent_tools.AUTO_TOOL_SCHEMA
        }
        self.assertIn("write_file", names)
        self.assertNotIn("bash", names)


class ExecuteToolTimeoutTests(unittest.TestCase):
    def test_execute_tool_enforces_configured_timeout(self) -> None:
        class SlowTool:
            file_action = None

            async def run(self, workspace_path: str, args: dict) -> str:
                del workspace_path, args
                await asyncio.sleep(60)
                return "finished"

        async def run() -> tuple[str, list[dict]]:
            events: list[dict] = []
            return (
                await agent_tools.execute_tool(
                    "slow_test", {}, "/tmp", "auto", events.append,
                ),
                events,
            )

        original_tools = agent_tools._BY_NAME
        original_timeout = agent_tools.tool_timeout
        agent_tools._BY_NAME = {**original_tools, "slow_test": SlowTool()}
        agent_tools.tool_timeout = lambda: 0.01
        try:
            result, events = asyncio.run(run())
        finally:
            agent_tools._BY_NAME = original_tools
            agent_tools.tool_timeout = original_timeout

        self.assertIn("Tool slow_test timed out after 0.01s", result)
        self.assertEqual(events[0]["type"], "tool_use")
        self.assertEqual(events[-1]["type"], "tool_result")
        self.assertEqual(events[-1]["output"], result)


class ExecuteToolResultTests(unittest.TestCase):
    def test_execute_tool_normalizes_malformed_call_fields(self) -> None:
        async def run() -> tuple[str, list[dict]]:
            events: list[dict] = []
            return (
                await agent_tools.execute_tool(
                    ["read_file"],  # type: ignore[arg-type]
                    ["not an object"],  # type: ignore[arg-type]
                    "/tmp",
                    "auto",
                    events.append,
                ),
                events,
            )

        result, events = asyncio.run(run())

        self.assertEqual(result, "Unknown tool: ")
        self.assertEqual(events, [
            {
                "type": "tool_use",
                "name": "",
                "input": {},
                "file_action": None,
            },
            {"type": "tool_result", "name": "", "output": result},
        ])

    def test_execute_tool_handles_non_text_result(self) -> None:
        class NonTextTool:
            file_action = None

            async def run(self, workspace_path: str, args: dict) -> object:
                del workspace_path, args
                return None

        async def run() -> tuple[str, list[dict]]:
            events: list[dict] = []
            return (
                await agent_tools.execute_tool(
                    "non_text_test", {}, "/tmp", "auto", events.append,
                ),
                events,
            )

        original_tools = agent_tools._BY_NAME
        agent_tools._BY_NAME = {
            **original_tools,
            "non_text_test": NonTextTool(),
        }
        try:
            result, events = asyncio.run(run())
        finally:
            agent_tools._BY_NAME = original_tools

        self.assertEqual(
            result,
            "Tool non_text_test returned an invalid non-text result "
            "(NoneType).",
        )
        self.assertEqual(events[0]["type"], "tool_use")
        self.assertEqual(events[-1]["type"], "tool_result")
        self.assertEqual(events[-1]["output"], result)


class ParseOpenAICallTests(unittest.TestCase):
    def test_string_arguments(self) -> None:
        name, args = agent_tools.parse_openai_call(
            {"function": {"name": "read_file",
                          "arguments": '{"path": "a.py"}'}},
        )
        self.assertEqual(name, "read_file")
        self.assertEqual(args, {"path": "a.py"})

    def test_dict_arguments_are_accepted(self) -> None:
        # GLM / Z.ai and some local runtimes return an already-parsed
        # object instead of a JSON string; it must not crash.
        _, args = agent_tools.parse_openai_call(
            {"function": {"name": "x", "arguments": {"path": "b.py"}}},
        )
        self.assertEqual(args, {"path": "b.py"})

    def test_missing_or_bad_arguments_yield_empty(self) -> None:
        for raw in (None, "", "not json", "[1, 2]"):
            _, args = agent_tools.parse_openai_call(
                {"function": {"name": "x", "arguments": raw}},
            )
            self.assertEqual(args, {}, f"raw={raw!r}")

    def test_malformed_function_or_name_yields_empty_name(self) -> None:
        for call in (
            {},
            {"function": None},
            {"function": "not an object"},
            {"function": {"name": ["read_file"]}},
        ):
            with self.subTest(call=call):
                self.assertEqual(agent_tools.parse_openai_call(call), ("", {}))


class TreeToolTests(unittest.TestCase):
    @staticmethod
    def _touch(path: str) -> None:
        with open(path, "w", encoding="utf-8"):
            pass

    def test_tree_shows_structure_and_skips_noise(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "src", "pkg"))
                os.makedirs(os.path.join(tmp, ".git", "objects"))
                self._touch(os.path.join(tmp, "README.md"))
                self._touch(os.path.join(tmp, "src", "main.py"))
                self._touch(os.path.join(tmp, "src", "pkg", "util.py"))
                out = await TreeTool().run(tmp, {})
                for marker in ("src/", "main.py", "pkg/", "util.py",
                               "README.md"):
                    self.assertIn(marker, out)
                self.assertNotIn(".git", out)  # noise dir skipped

        asyncio.run(run())

    def test_tree_depth_limits_recursion(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "a", "b", "c"))
                self._touch(os.path.join(tmp, "a", "b", "c", "deep.py"))
                out = await TreeTool().run(tmp, {"depth": 1})
                self.assertIn("a/", out)
                self.assertNotIn("deep.py", out)  # beyond depth 1

        asyncio.run(run())


class ApplyPatchToolTests(unittest.TestCase):
    @staticmethod
    def _write(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _run(self, ws: str, patch: str) -> str:
        return asyncio.run(ApplyPatchTool().run(ws, {"patch": patch}))

    def test_modify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "foo.txt")
            self._write(p, "line1\nline2\nline3\n")
            out = self._run(tmp, (
                "--- a/foo.txt\n+++ b/foo.txt\n@@ -1,3 +1,3 @@\n"
                " line1\n-line2\n+line2-changed\n line3\n"
            ))
            self.assertIn("applied", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "line1\nline2-changed\nline3\n")

    def test_modify_preserves_crlf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "windows.txt")
            with open(p, "wb") as f:
                f.write(b"line1\r\nline2\r\n")

            out = self._run(tmp, (
                "--- a/windows.txt\n+++ b/windows.txt\n"
                "@@ -1,2 +1,2 @@\n line1\n-line2\n+changed\n"
            ))

            self.assertIn("applied", out)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"line1\r\nchanged\r\n")

    def test_modify_applies_requested_final_newline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "newline.txt")
            with open(p, "wb") as f:
                f.write(b"old\n")

            out = self._run(tmp, (
                "--- a/newline.txt\n+++ b/newline.txt\n"
                "@@ -1 +1 @@\n-old\n+tail\n"
                "\\ No newline at end of file\n"
            ))

            self.assertIn("applied", out)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"tail")

    def test_modify_can_restore_a_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "newline.txt")
            with open(p, "wb") as f:
                f.write(b"old")

            out = self._run(tmp, (
                "--- a/newline.txt\n+++ b/newline.txt\n"
                "@@ -1 +1 @@\n-old\n"
                "\\ No newline at end of file\n+tail\n"
            ))

            self.assertIn("applied", out)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"tail\n")

    def test_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/new/dir/created.txt\n"
                "@@ -0,0 +1,2 @@\n+hello\n+world\n"
            ))
            self.assertIn("created", out)
            created = os.path.join(tmp, "new", "dir", "created.txt")
            with open(created, encoding="utf-8") as f:
                self.assertEqual(f.read(), "hello\nworld\n")

    def test_create_rejects_an_old_side_for_dev_null(self) -> None:
        """Malformed creation diffs must not silently discard old-side text."""
        with tempfile.TemporaryDirectory() as tmp:
            created = os.path.join(tmp, "new.txt")
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/new.txt\n"
                "@@ -1 +1 @@\n-old-side-must-not-exist\n+new\n"
            ))

            self.assertIn("could not parse patch", out)
            self.assertIn("creation hunk", out)
            self.assertFalse(os.path.exists(created))

    def test_create_preserves_requested_missing_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            created = os.path.join(tmp, "new.txt")
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/new.txt\n"
                "@@ -0,0 +1 @@\n+tail\n"
                "\\ No newline at end of file\n"
            ))

            self.assertIn("created", out)
            with open(created, "rb") as f:
                self.assertEqual(f.read(), b"tail")

    def test_create_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "existing.txt")
            self._write(p, "keep me\n")
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/existing.txt\n"
                "@@ -0,0 +1 @@\n+replacement\n"
            ))
            self.assertIn("already exists", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep me\n")

    def test_create_keeps_target_absent_when_atomic_create_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "new.txt")
            patch = (
                "--- /dev/null\n+++ b/new.txt\n"
                "@@ -0,0 +1 @@\n+new content\n"
            )

            with mock.patch(
                "Cozter.agent_tools.base.os.link",
                side_effect=OSError("simulated link failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated link"):
                    self._run(tmp, patch)

            self.assertFalse(os.path.exists(p))

    @unittest.skipIf(os.name == "nt", "POSIX mode/umask semantics")
    def test_create_uses_normal_file_creation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_umask = os.umask(0o027)
            try:
                out = self._run(tmp, (
                    "--- /dev/null\n+++ b/new.txt\n"
                    "@@ -0,0 +1 @@\n+new content\n"
                ))
            finally:
                os.umask(previous_umask)

            self.assertIn("created", out)
            self.assertEqual(
                stat.S_IMODE(os.stat(os.path.join(tmp, "new.txt")).st_mode),
                0o640,
            )

    def test_create_falls_back_when_hard_links_are_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "new.txt")
            with mock.patch(
                "Cozter.agent_tools.base.os.link",
                side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
            ):
                out = self._run(tmp, (
                    "--- /dev/null\n+++ b/new.txt\n"
                    "@@ -0,0 +1 @@\n+fallback content\n"
                ))

            self.assertIn("created", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "fallback content\n")

    def test_create_does_not_clobber_concurrent_creator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "new.txt")
            patch = (
                "--- /dev/null\n+++ b/new.txt\n"
                "@@ -0,0 +1 @@\n+patch content\n"
            )

            def concurrent_create(_source: str, destination: str) -> None:
                with open(destination, "w", encoding="utf-8") as f:
                    f.write("other writer\n")
                raise FileExistsError(destination)

            with mock.patch(
                "Cozter.agent_tools.base.os.link",
                side_effect=concurrent_create,
            ):
                out = self._run(tmp, patch)

            self.assertIn("already exists", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "other writer\n")

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "gone.txt")
            self._write(p, "bye\n")
            out = self._run(
                tmp, "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n",
            )
            self.assertIn("deleted", out)
            self.assertFalse(os.path.exists(p))

    def test_delete_requires_matching_hunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "keep.txt")
            self._write(p, "actual content\n")
            out = self._run(tmp, (
                "--- a/keep.txt\n+++ /dev/null\n"
                "@@ -1 +0,0 @@\n-expected content\n"
            ))
            self.assertIn("did not apply", out)
            self.assertIn("not deleted", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "actual content\n")

    def test_multi_hunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.txt")
            self._write(p, "\n".join(f"L{i}" for i in range(1, 11)) + "\n")
            out = self._run(tmp, (
                "--- a/m.txt\n+++ b/m.txt\n"
                "@@ -1,2 +1,2 @@\n L1\n-L2\n+L2x\n"
                "@@ -9,2 +9,2 @@\n L9\n-L10\n+L10x\n"
            ))
            self.assertIn("2 hunk", out)
            with open(p, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("L2x", content)
            self.assertIn("L10x", content)
            self.assertNotIn("\nL2\n", content)

    def test_context_not_found_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "c.txt")
            self._write(p, "alpha\nbeta\n")
            out = self._run(tmp, (
                "--- a/c.txt\n+++ b/c.txt\n@@ -1,2 +1,2 @@\n"
                " nonexistent-context\n-beta\n+gamma\n"
            ))
            self.assertIn("did not apply", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "alpha\nbeta\n")

    def test_incomplete_hunk_is_rejected_without_partial_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "partial.txt")
            self._write(p, "old\nsecond\n")

            out = self._run(tmp, (
                "--- a/partial.txt\n+++ b/partial.txt\n"
                "@@ -1,2 +1,2 @@\n-old\n+new\n"
            ))

            self.assertIn("could not parse patch", out)
            self.assertIn("line counts", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "old\nsecond\n")

    def test_overlong_hunk_is_rejected_without_dropping_body_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "overlong.txt")
            self._write(p, "old\n")

            out = self._run(tmp, (
                "--- a/overlong.txt\n+++ b/overlong.txt\n"
                "@@ -1 +1 @@\n-old\n+new\n+silently-dropped-before\n"
            ))

            self.assertIn("could not parse patch", out)
            self.assertIn("more body lines", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "old\n")

    def test_overlong_hunk_number_is_reported_as_invalid_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/new.txt\n"
                f"@@ -0,0 +1,{('9' * 5_000)} @@\n+new\n"
            ))

            self.assertIn("could not parse patch", out)
            self.assertIn("invalid line count", out)
            self.assertFalse(os.path.exists(os.path.join(tmp, "new.txt")))

    def test_fuzzy_trailing_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "w.txt")
            # File has trailing spaces the patch context omits: the fuzzy
            # fallback still applies (context trailing ws normalizes to the
            # patch's form).
            self._write(p, "keep  \ndrop\n")
            out = self._run(tmp, (
                "--- a/w.txt\n+++ b/w.txt\n@@ -1,2 +1,1 @@\n keep\n-drop\n"
            ))
            self.assertIn("applied", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep\n")

    def test_file_header_markers_inside_hunk_are_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "markers.txt")
            self._write(p, "-- old marker\nplain\n")
            out = self._run(tmp, (
                "--- a/markers.txt\n+++ b/markers.txt\n"
                "@@ -1,2 +1,2 @@\n"
                "--- old marker\n"
                "+++ new marker\n"
                " plain\n"
            ))
            self.assertIn("applied", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "++ new marker\nplain\n")


if __name__ == "__main__":
    unittest.main()
