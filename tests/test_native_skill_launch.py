"""Native Copilot skills remain discoverable with private session homes."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter.backends_agent import copilot


class NativeSkillLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_home_keeps_existing_and_personal_skill_search_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp, "personal")
            (home / "skills").mkdir(parents=True)
            existing = Path(temp, "extra skills")
            existing.mkdir()
            process = mock.Mock()
            backend = copilot.CopilotBackend()
            with (
                mock.patch.dict(os.environ, {"COPILOT_HOME": str(home), "COPILOT_SKILLS_DIRS": str(existing)}),
                mock.patch.object(copilot, "executable_command", return_value=["copilot"]),
                mock.patch.object(copilot, "create_prompt_subprocess", new=mock.AsyncMock(return_value=process)) as launch,
            ):
                await backend.launch(temp, "offline task", "fixture-model", "full")
                env = launch.call_args.kwargs["env"]
                self.assertNotEqual(env["COPILOT_HOME"], str(home))
                self.assertEqual(env["COPILOT_SKILLS_DIRS"].split(","), [str(existing), str(home / "skills")])
                self.assertIn("--yolo", launch.call_args.args[0])
                self.assertIn("--no-remote-export", launch.call_args.args[0])
                await backend.cleanup_process(process)

    def test_personal_skill_directory_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            skills_dir = Path(temp, "skills")
            skills_dir.mkdir()
            with mock.patch.dict(os.environ, {"COPILOT_HOME": temp}):
                env = {"COPILOT_SKILLS_DIRS": str(skills_dir)}
                copilot._preserve_personal_skill_roots(env)
                self.assertEqual(env["COPILOT_SKILLS_DIRS"], str(skills_dir))

    def test_missing_personal_directory_leaves_existing_search_path_unchanged(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"COPILOT_HOME": temp}):
            env = {"COPILOT_SKILLS_DIRS": "existing"}
            copilot._preserve_personal_skill_roots(env)
            self.assertEqual(env, {"COPILOT_SKILLS_DIRS": "existing"})

    def test_native_copilot_home_refreshes_config_and_settings_across_launches(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            source_home = Path(temp, "personal")
            source_home.mkdir()
            (source_home / "config.json").write_text('{"theme":"dark"}', encoding="utf-8")
            (source_home / "settings.json").write_text('{"skillDirectories":["/skills/one"]}', encoding="utf-8")
            exec_dir = Path(temp, "execution")
            exec_dir.mkdir()
            record = SimpleNamespace(directory=exec_dir)

            with mock.patch.dict(os.environ, {"COPILOT_HOME": str(source_home)}):
                native_home = Path(copilot._native_copilot_home(record))
                session_marker = native_home / "session-state" / "session.json"
                session_marker.parent.mkdir(parents=True, exist_ok=True)
                session_marker.write_text('{"id":"keep-me"}', encoding="utf-8")
                self.assertEqual(
                    (native_home / "settings.json").read_text(encoding="utf-8"),
                    '{"skillDirectories":["/skills/one"]}',
                )

                # User runs `copilot skill add /skills/two` between turns.
                (source_home / "settings.json").write_text(
                    '{"skillDirectories":["/skills/one","/skills/two"]}',
                    encoding="utf-8",
                )
                (source_home / "config.json").write_text('{"theme":"light"}', encoding="utf-8")
                copilot._native_copilot_home(record)

                self.assertEqual(
                    (native_home / "settings.json").read_text(encoding="utf-8"),
                    '{"skillDirectories":["/skills/one","/skills/two"]}',
                )
                self.assertEqual(
                    (native_home / "config.json").read_text(encoding="utf-8"),
                    '{"theme":"light"}',
                )
                self.assertEqual(session_marker.read_text(encoding="utf-8"), '{"id":"keep-me"}')


if __name__ == "__main__":
    unittest.main()
