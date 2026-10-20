"""Verify that trusted execution permissions govern all file-tool paths."""

from __future__ import annotations

import asyncio
import copy
import json
import ntpath
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, config, workspace as workspace_settings
from Cozter.agent_tools import base as tool_base
from Cozter.agent_tools.builtin import copy_file as copy_tool
from Cozter.agent_tools.builtin import glob as glob_tool
from Cozter.agent_tools.builtin.read_file import ReadFileTool
from Cozter.backends_agent import _openai_agent as openai_agent


class _PermissionBackend(openai_agent.OpenAIChatBackend):
    name = "permission_test"
    supports_vision = False

    def _chat_endpoint(self) -> str:
        return "http://unused.invalid/chat/completions"

    def _max_agent_turns(self) -> int:
        return 2


class FileToolPermissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.outside = self.root / "outside"
        self.workspace.mkdir()
        self.outside.mkdir()
        self.inside_file = self.workspace / "inside.txt"
        self.outside_file = self.outside / "outside.txt"
        self.inside_file.write_text("inside content\n", encoding="utf-8")
        self.outside_file.write_text("outside needle\n", encoding="utf-8")

    async def _execute(
        self, name: str, args: dict, approval: str = "full",
    ) -> str:
        events: list[dict] = []
        result = await agent_tools.execute_tool(
            name, args, str(self.workspace), approval, events.append,
        )
        self.assertEqual(events[-1]["type"], "tool_result")
        self.assertEqual(events[-1]["output"], result)
        return result

    def _symlink(self, target: Path, link: Path) -> None:
        try:
            os.symlink(target, link, target_is_directory=target.is_dir())
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

    async def test_full_reads_absolute_and_parent_relative_paths(self) -> None:
        for path in (str(self.outside_file), "../outside/outside.txt"):
            with self.subTest(path=path):
                result = await self._execute("read_file", {"path": path})
                self.assertEqual(result, "outside needle\n")

        # Relative paths still use the configured workspace as their base.
        self.assertEqual(
            await self._execute("read_file", {"path": "inside.txt"}),
            "inside content\n",
        )

    async def test_full_writes_and_edits_outside_workspace(self) -> None:
        path = self.outside / "created" / "note.txt"
        result = await self._execute(
            "write_file", {"path": str(path), "content": "alpha beta\n"},
        )
        self.assertTrue(result.startswith("Wrote"), result)
        self.assertEqual(path.read_text(encoding="utf-8"), "alpha beta\n")

        result = await self._execute("edit_file", {
            "path": str(path), "old_string": "beta", "new_string": "gamma",
        })
        self.assertTrue(result.startswith("Replaced"), result)
        self.assertEqual(path.read_text(encoding="utf-8"), "alpha gamma\n")

        result = await self._execute("multi_edit", {
            "path": str(path),
            "edits": [{"old_string": "alpha", "new_string": "delta"}],
        })
        self.assertTrue(result.startswith("Applied"), result)
        self.assertEqual(path.read_text(encoding="utf-8"), "delta gamma\n")

    async def test_full_creates_directory_and_applies_outside_patch(self) -> None:
        directory = self.outside / "created" / "nested"
        result = await self._execute("make_dir", {"path": str(directory)})
        self.assertTrue(result.startswith("Directory ready"), result)
        self.assertTrue(directory.is_dir())

        result = await self._execute("apply_patch", {"patch": (
            "--- a/../outside/outside.txt\n"
            "+++ b/../outside/outside.txt\n"
            "@@ -1 +1 @@\n-outside needle\n+patched needle\n"
        )})
        self.assertIn("applied", result)
        self.assertEqual(
            self.outside_file.read_text(encoding="utf-8"), "patched needle\n",
        )

    async def test_full_lists_and_searches_outside_workspace(self) -> None:
        nested = self.outside / "nested"
        nested.mkdir()
        (nested / "second.txt").write_text("second needle\n", encoding="utf-8")
        for name in ("list_dir", "tree"):
            with self.subTest(tool=name):
                result = await self._execute(name, {"path": str(self.outside)})
                self.assertIn("outside.txt", result)
                self.assertIn("nested", result)

        for pattern in (
            str(self.outside / "**" / "*.txt"), "../outside/**/*.txt",
        ):
            with self.subTest(pattern=pattern):
                result = await self._execute("glob", {"pattern": pattern})
                self.assertIn("outside.txt", result)
                self.assertIn("second.txt", result)

        result = await self._execute("glob", {
            "path": str(self.outside), "pattern": "**/*.txt",
        })
        self.assertIn("outside.txt", result)
        self.assertIn("second.txt", result)

        # Run the actual spawned worker: thread-only coverage would miss
        # permissions being lost at the multiprocessing boundary.
        result = await self._execute("grep", {
            "path": str(self.outside), "pattern": "needle", "glob": "**/*.txt",
        })
        self.assertIn("outside.txt:1: outside needle", result)
        self.assertIn("second.txt:1: second needle", result)

    async def test_glob_explicit_generated_directory_keeps_nested_named_directories(self) -> None:
        generated = self.workspace / "node_modules"
        nested = generated / "dep" / "node_modules" / "nested.js"
        nested.parent.mkdir(parents=True)
        nested.write_text("nested dependency\n", encoding="utf-8")
        (generated / "top.js").write_text("top dependency\n", encoding="utf-8")
        for permission in ("full", "auto"):
            with self.subTest(permission=permission):
                explicit = await self._execute("glob", {
                    "pattern": "node_modules/**/*.js",
                }, permission)
                self.assertIn("node_modules/top.js", explicit)
                self.assertIn("node_modules/dep/node_modules/nested.js", explicit)
                default = await self._execute("glob", {
                    "pattern": "**/*.js",
                }, permission)
                self.assertTrue(default.startswith("No files matched"), default)

    async def test_compaction_full_turn_disables_actual_tool_execution(self) -> None:
        requests: list[dict] = []
        events: list[dict] = []

        async def stream(*args, **kwargs):
            requests.append(copy.deepcopy(args[1]))
            if len(requests) == 1:
                return "", "", [{
                    "id": "unexpected-write", "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": str(self.outside_file), "content": "changed",
                        }),
                    },
                }]
            return "summary", "", []

        with mock.patch.object(openai_agent, "_stream_completion", side_effect=stream):
            await _PermissionBackend()._run_agent(
                SimpleNamespace(emit=events.append), str(self.workspace),
                "summarize this session", None, "full", True, 0,
            )
        self.assertTrue(requests)
        for request in requests:
            self.assertNotIn("tools", request)
            self.assertIn("Effective permission: deny", request["messages"][0]["content"])
        self.assertEqual(self.outside_file.read_text(encoding="utf-8"), "outside needle\n")
        self.assertTrue(any(
            event.get("type") == "tool_result"
            and event.get("output", "").startswith("Blocked")
            for event in events
        ), events)

    async def test_full_copies_moves_and_deletes_outside_workspace(self) -> None:
        copied = self.outside / "copied.txt"
        moved = self.workspace / "moved.txt"
        result = await self._execute("copy_file", {
            "source": "inside.txt", "destination": str(copied),
        })
        self.assertTrue(result.startswith("Copied"), result)
        self.assertEqual(copied.read_text(encoding="utf-8"), "inside content\n")

        result = await self._execute("move_file", {
            "source": str(copied), "destination": str(moved),
        })
        self.assertTrue(result.startswith("Moved"), result)
        self.assertFalse(copied.exists())
        self.assertEqual(moved.read_text(encoding="utf-8"), "inside content\n")

        result = await self._execute("delete_file", {"path": str(self.outside_file)})
        self.assertTrue(result.startswith("Deleted"), result)
        self.assertFalse(self.outside_file.exists())
        self.assertTrue(self.inside_file.exists())

    async def test_full_transfer_still_refuses_clobbering(self) -> None:
        for name in ("copy_file", "move_file"):
            with self.subTest(tool=name):
                result = await self._execute(name, {
                    "source": "inside.txt", "destination": str(self.outside_file),
                })
                self.assertTrue(result.startswith("Destination already exists"), result)
                self.assertEqual(
                    self.outside_file.read_text(encoding="utf-8"), "outside needle\n",
                )
                self.assertEqual(
                    self.inside_file.read_text(encoding="utf-8"), "inside content\n",
                )

    async def test_full_copy_preserves_binary_read_only_source(self) -> None:
        source = self.outside / "read-only.bin"
        destination = self.workspace / "copied.bin"
        content = bytes(range(256)) * 1_024
        source.write_bytes(content)
        source.chmod(0o444)

        def restore_permissions() -> None:
            # Windows refuses removal of read-only files. Restore permissions
            # before TemporaryDirectory cleanup, including a failed temp copy.
            for directory in (self.workspace, self.outside):
                for path in directory.iterdir():
                    if path.is_file():
                        path.chmod(0o600)

        self.addCleanup(restore_permissions)
        result = await self._execute("copy_file", {
            "source": str(source), "destination": destination.name,
        })
        self.assertTrue(result.startswith("Copied"), result)
        self.assertEqual(source.read_bytes(), content)
        self.assertEqual(destination.read_bytes(), content)
        self.assertFalse(source.stat().st_mode & stat.S_IWRITE)
        self.assertFalse(destination.stat().st_mode & stat.S_IWRITE)
        self.assertEqual(list(self.workspace.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows read-only publication branch")
    async def test_read_only_copy_race_preserves_competing_destination_and_cleans_temp(self) -> None:
        source = self.outside / "read-only.bin"
        destination = self.workspace / "competing.bin"
        source.write_bytes(b"source\x00\xff")
        source.chmod(0o444)
        self.addCleanup(source.chmod, 0o600)
        original_ensure_parent = copy_tool.ensure_parent_dir

        def create_competing_destination(path: str) -> None:
            original_ensure_parent(path)
            destination.write_bytes(b"other writer\x00\xff")

        with mock.patch.object(
            copy_tool, "ensure_parent_dir", side_effect=create_competing_destination,
        ):
            result = await self._execute("copy_file", {
                "source": str(source), "destination": destination.name,
            })
        self.assertTrue(result.startswith("Destination already exists"), result)
        self.assertEqual(destination.read_bytes(), b"other writer\x00\xff")
        self.assertEqual(source.read_bytes(), b"source\x00\xff")
        self.assertFalse(source.stat().st_mode & stat.S_IWRITE)
        self.assertEqual(list(self.workspace.glob("*.tmp")), [])

    async def test_auto_rejects_outside_paths_without_mutations(self) -> None:
        created = self.outside / "new" / "created.txt"
        operations = (
            ("read_file", {"path": str(self.outside_file)}),
            ("write_file", {"path": str(created), "content": "changed"}),
            ("edit_file", {
                "path": str(self.outside_file),
                "old_string": "needle", "new_string": "changed",
            }),
            ("multi_edit", {
                "path": str(self.outside_file),
                "edits": [{"old_string": "needle", "new_string": "changed"}],
            }),
            ("make_dir", {"path": str(created.parent)}),
            ("copy_file", {"source": "inside.txt", "destination": str(created)}),
            ("copy_file", {
                "source": str(self.outside_file), "destination": "copied.txt",
            }),
            ("move_file", {
                "source": str(self.outside_file), "destination": "moved.txt",
            }),
            ("move_file", {"source": "inside.txt", "destination": str(created)}),
            ("delete_file", {"path": str(self.outside_file)}),
            ("list_dir", {"path": str(self.outside)}),
            ("tree", {"path": str(self.outside)}),
            ("grep", {"path": str(self.outside), "pattern": "needle"}),
            ("glob", {"path": str(self.outside), "pattern": "**/*.txt"}),
            ("apply_patch", {"patch": (
                "--- a/../outside/outside.txt\n"
                "+++ b/../outside/outside.txt\n"
                "@@ -1 +1 @@\n-outside needle\n+changed\n"
            )}),
        )
        for name, args in operations:
            with self.subTest(tool=name, args=args):
                result = await self._execute(name, args, "auto")
                self.assertIn("escapes workspace", result)
                self.assertEqual(
                    self.outside_file.read_text(encoding="utf-8"), "outside needle\n",
                )
                self.assertEqual(
                    self.inside_file.read_text(encoding="utf-8"), "inside content\n",
                )
                self.assertFalse(created.parent.exists())
                self.assertEqual(sorted(path.name for path in self.workspace.iterdir()), [
                    "inside.txt",
                ])

        for pattern in (
            str(self.outside / "**" / "*.txt"), "../outside/**/*.txt",
        ):
            result = await self._execute("glob", {"pattern": pattern}, "auto")
            self.assertNotIn("outside.txt\n", result)
            self.assertFalse(result.endswith("outside.txt"), result)

    async def test_confirm_reads_inside_and_rejects_outside_or_mutations(self) -> None:
        result = await self._execute("read_file", {"path": "inside.txt"}, "confirm")
        self.assertEqual(result, "inside content\n")
        for name, args in (
            ("read_file", {"path": str(self.outside_file)}),
            ("list_dir", {"path": str(self.outside)}),
            ("tree", {"path": str(self.outside)}),
            ("grep", {"path": str(self.outside), "pattern": "needle"}),
        ):
            with self.subTest(tool=name):
                self.assertIn("escapes workspace", await self._execute(name, args, "confirm"))

        result = await self._execute("write_file", {
            "path": "inside.txt", "content": "changed",
        }, "confirm")
        self.assertTrue(result.startswith("Blocked"), result)
        self.assertEqual(self.inside_file.read_text(encoding="utf-8"), "inside content\n")

    async def test_deny_and_unknown_permissions_cannot_run_file_tools(self) -> None:
        for permission in ("deny", "unexpected", "FULL", ""):
            with self.subTest(permission=permission):
                result = await self._execute("read_file", {
                    "path": str(self.outside_file), "permission": "full",
                }, permission)
                self.assertTrue(result.startswith("Blocked"), result)
                result = await self._execute("write_file", {
                    "path": "inside.txt", "content": "changed", "approval": "full",
                }, permission)
                self.assertTrue(result.startswith("Blocked"), result)
        self.assertEqual(self.inside_file.read_text(encoding="utf-8"), "inside content\n")

    async def test_tool_arguments_cannot_grant_full_permission(self) -> None:
        args = {
            "path": str(self.outside_file), "permission": "full",
            "approval": "full", "sandbox": False, "full_permission": True,
        }
        for permission in ("auto", "confirm"):
            with self.subTest(permission=permission):
                self.assertIn("escapes workspace", await self._execute(
                    "read_file", args, permission,
                ))
        self.assertIn("escapes workspace", await ReadFileTool().run(
            str(self.workspace), args,
        ))

    async def test_loaded_config_ceiling_controls_outside_tool_access(self) -> None:
        settings_dir = self.workspace / ".cozter"
        settings_dir.mkdir()
        (settings_dir / "settings.json").write_text(
            json.dumps({"permission": "full"}), encoding="utf-8",
        )
        config_path = self.root / "config.json"
        with mock.patch.object(config, "CONFIG_PATH", str(config_path)):
            config_path.write_text(json.dumps({"max_permission": "auto"}), encoding="utf-8")
            effective = workspace_settings.get_permission(str(self.workspace))
            self.assertEqual(effective, "auto")
            self.assertIn("escapes workspace", await self._execute(
                "read_file", {"path": str(self.outside_file)}, effective,
            ))

            config_path.write_text(json.dumps({"max_permission": "full"}), encoding="utf-8")
            effective = workspace_settings.get_permission(str(self.workspace))
            self.assertEqual(effective, "full")
            self.assertEqual(await self._execute(
                "read_file", {"path": str(self.outside_file)}, effective,
            ), "outside needle\n")

    async def test_full_follows_outside_symlink_for_content_tools(self) -> None:
        link = self.workspace / "outside-link.txt"
        self._symlink(self.outside_file, link)
        self.assertEqual(
            await self._execute("read_file", {"path": link.name}), "outside needle\n",
        )
        result = await self._execute("write_file", {
            "path": link.name, "content": "changed needle\n",
        })
        self.assertTrue(result.startswith("Wrote"), result)
        self.assertTrue(link.is_symlink())
        self.assertEqual(self.outside_file.read_text(encoding="utf-8"), "changed needle\n")
        for name, args in (
            ("glob", {"pattern": "**/*.txt"}),
            ("grep", {"pattern": "changed needle"}),
        ):
            with self.subTest(tool=name):
                self.assertIn(link.name, await self._execute(name, args))
                self.assertNotIn(link.name, await self._execute(name, args, "auto"))
        for name, args in (
            ("read_file", {"path": link.name}),
            ("write_file", {"path": link.name, "content": "denied"}),
        ):
            with self.subTest(tool=name):
                self.assertIn("escapes workspace", await self._execute(name, args, "auto"))
        self.assertEqual(self.outside_file.read_text(encoding="utf-8"), "changed needle\n")

    async def test_full_moves_and_deletes_symlink_without_touching_target(self) -> None:
        link = self.workspace / "outside-link.txt"
        moved = self.outside / "moved-link.txt"
        self._symlink(self.outside_file, link)
        result = await self._execute("move_file", {
            "source": link.name, "destination": str(moved),
        })
        self.assertTrue(result.startswith("Moved"), result)
        self.assertFalse(os.path.lexists(link))
        self.assertTrue(moved.is_symlink())
        self.assertEqual(moved.read_text(encoding="utf-8"), "outside needle\n")
        result = await self._execute("delete_file", {"path": str(moved)})
        self.assertTrue(result.startswith("Deleted"), result)
        self.assertFalse(os.path.lexists(moved))
        self.assertEqual(self.outside_file.read_text(encoding="utf-8"), "outside needle\n")

    async def test_concurrent_full_and_auto_calls_keep_permissions_isolated(self) -> None:
        entered = 0
        ready = asyncio.Event()

        async def delayed_read(workspace: str, args: dict) -> str:
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            return await ReadFileTool().run(workspace, args)

        tool = SimpleNamespace(
            name="delayed_read", file_action=None, is_plugin=False, run=delayed_read,
        )
        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"delayed_read": tool}):
            full, auto = await asyncio.gather(
                self._execute("delayed_read", {"path": str(self.outside_file)}, "full"),
                self._execute("delayed_read", {"path": str(self.outside_file)}, "auto"),
            )
        self.assertEqual(full, "outside needle\n")
        self.assertIn("escapes workspace", auto)
        self.assertIn("escapes workspace", await ReadFileTool().run(
            str(self.workspace), {"path": str(self.outside_file)},
        ))

    async def test_failed_full_call_does_not_leak_permission(self) -> None:
        async def failing_read(workspace: str, args: dict) -> str:
            self.assertEqual(await ReadFileTool().run(workspace, args), "outside needle\n")
            raise RuntimeError("intentional test failure")

        tool = SimpleNamespace(name="failing_read", file_action=None, run=failing_read)
        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"failing_read": tool}):
            result = await self._execute("failing_read", {"path": str(self.outside_file)})
        self.assertIn("intentional test failure", result)
        self.assertIn("escapes workspace", await ReadFileTool().run(
            str(self.workspace), {"path": str(self.outside_file)},
        ))
        self.assertIn("escapes workspace", await self._execute(
            "read_file", {"path": str(self.outside_file)}, "auto",
        ))

    async def test_explicit_operation_timeout_does_not_leak_permission(self) -> None:
        async def waiting_read(workspace: str, args: dict) -> str:
            self.assertEqual(await ReadFileTool().run(workspace, args), "outside needle\n")
            raise TimeoutError("requested operation deadline")

        tool = SimpleNamespace(name="waiting_read", file_action=None, run=waiting_read)
        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"waiting_read": tool}), self.assertLogs(
            "Cozter.performance", level="INFO",
        ) as timings:
            result = await self._execute("waiting_read", {"path": str(self.outside_file)})
        self.assertTrue(result.startswith("Tool waiting_read failed"), result)
        self.assertIn("timed out", result)
        self.assertIn("requested operation", result)
        self.assertIn("before retrying", result)
        self.assertTrue(any("stage=tool_start" in entry for entry in timings.output))
        self.assertTrue(any("outcome=timeout" in entry for entry in timings.output))
        self.assertNotIn(str(self.outside_file), "\n".join(timings.output))
        self.assertIn("escapes workspace", await ReadFileTool().run(
            str(self.workspace), {"path": str(self.outside_file)},
        ))

    async def test_cancelled_full_call_resets_permission_in_its_task(self) -> None:
        started = asyncio.Event()

        async def waiting_read(workspace: str, args: dict) -> str:
            self.assertEqual(await ReadFileTool().run(workspace, args), "outside needle\n")
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

        tool = SimpleNamespace(name="waiting_read", file_action=None, run=waiting_read)

        async def cancelled_caller() -> str:
            try:
                await self._execute("waiting_read", {"path": str(self.outside_file)})
            except asyncio.CancelledError:
                # Inspect the same task after the execution scope has unwound.
                return await ReadFileTool().run(
                    str(self.workspace), {"path": str(self.outside_file)},
                )
            raise AssertionError("cancellation was swallowed")

        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"waiting_read": tool}):
            task = asyncio.create_task(cancelled_caller())
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
            finally:
                task.cancel()
            self.assertIn("escapes workspace", await task)


class PermissionPromptTests(unittest.TestCase):
    def _prompt(self, approval: str) -> str:
        schema = openai_agent._tools_for_approval(approval, False)
        names = tuple(entry["function"]["name"] for entry in schema or ())
        return openai_agent._system_prompt("test-workspace", names, approval)

    def test_full_prompt_describes_host_filesystem_access(self) -> None:
        prompt = self._prompt("full")
        self.assertIn("Effective permission: full", prompt)
        self.assertIn("may access paths outside the workspace", prompt)
        self.assertIn("absolute paths", prompt)
        self.assertIn("Relative paths are based on the workspace", prompt)
        self.assertNotIn("Filesystem tools are confined to the workspace", prompt)

    def test_auto_prompt_describes_workspace_boundary(self) -> None:
        prompt = self._prompt("auto")
        self.assertIn("Effective permission: auto", prompt)
        self.assertIn("Filesystem tools are confined to the workspace", prompt)
        self.assertIn("resolved symlink targets", prompt)
        self.assertNotIn("may access paths outside the workspace", prompt)

    def test_confirm_prompt_describes_read_only_boundary(self) -> None:
        prompt = self._prompt("confirm")
        self.assertIn("Effective permission: confirm", prompt)
        self.assertIn("Filesystem tools are confined to the workspace", prompt)
        self.assertIn("Only read-only tools are available", prompt)

    def test_deny_and_unknown_prompts_disable_tools(self) -> None:
        for approval in ("deny", "unexpected"):
            with self.subTest(approval=approval):
                prompt = self._prompt(approval)
                self.assertIn("Effective permission: deny", prompt)
                self.assertIn("Tool execution is disabled", prompt)
                self.assertIn("No tools this turn", prompt)
                self.assertNotIn("may access paths outside the workspace", prompt)


class WindowsDiscoveryPathTests(unittest.TestCase):
    """Use Windows path semantics without a second drive or network share."""

    def test_glob_treats_bracketed_unc_anchor_as_literal_search_root(self) -> None:
        workspace = r"C:\workspace"
        for anchor in (r"\\server\share[1]", r"\\?\UNC\server\share[1]"):
            with self.subTest(anchor=anchor):
                split_calls = 0

                def bounded_split(path: str) -> tuple[str, str]:
                    nonlocal split_calls
                    split_calls += 1
                    # Bound this regression even if splitting a magical
                    # anchor stops making progress and would loop forever.
                    if split_calls > 32:
                        raise AssertionError("glob prefix splitting did not terminate")
                    return ntpath.split(path)

                path_api = SimpleNamespace(
                    splitdrive=ntpath.splitdrive, split=bounded_split,
                    join=ntpath.join,
                )
                target = anchor + r"\note.txt"
                with (
                    mock.patch.object(glob_tool, "os", SimpleNamespace(path=path_api)),
                    mock.patch.object(
                        glob_tool, "resolve_inside_workspace",
                        side_effect=lambda _workspace, path: ntpath.normpath(path),
                    ) as resolve,
                    mock.patch.object(
                        glob_tool, "iter_workspace_files",
                        return_value=[(target, target, "note.txt")],
                    ) as scan,
                ):
                    result = glob_tool.GlobTool._find_matches(
                        workspace, anchor + r"\**\*.txt", 100,
                    )
                self.assertEqual(result, [target])
                resolve.assert_called_once_with(workspace, anchor + "\\")
                scan.assert_called_once_with(
                    workspace, ntpath.normpath(anchor + "\\"), r"**\*.txt",
                    discovery_pattern=anchor + r"\**\*.txt",
                    sort_entries=False, source_files=None,
                )

    def test_full_iterator_labels_external_link_to_another_drive_as_absolute(self) -> None:
        workspace = r"C:\workspace"
        outside = r"D:\outside"
        link = outside + r"\link.txt"
        target = workspace + r"\target.txt"

        def realpath(path: str) -> str:
            return target if path == link else ntpath.normpath(path)

        path_api = SimpleNamespace(
            realpath=realpath, join=ntpath.join,
            relpath=ntpath.relpath, commonpath=ntpath.commonpath,
            normcase=ntpath.normcase,
        )
        mocked_os = SimpleNamespace(
            path=path_api, sep="\\",
            walk=mock.Mock(return_value=[(outside, [], ["link.txt"])]),
        )
        with (
            mock.patch.object(tool_base, "os", mocked_os),
            mock.patch.object(
                tool_base, "is_path_within",
                side_effect=lambda path, root: ntpath.commonpath((path, root)) == root,
            ),
            tool_base.tool_permission_scope("full"),
        ):
            files = list(tool_base.iter_workspace_files(workspace, outside, "**/*.txt"))
        mocked_os.walk.assert_called_once_with(outside)
        self.assertEqual(files, [(link, "D:/outside/link.txt", "link.txt")])


if __name__ == "__main__":
    unittest.main()
