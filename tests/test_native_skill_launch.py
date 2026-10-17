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


if __name__ == "__main__":
    unittest.main()
