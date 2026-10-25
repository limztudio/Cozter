"""Tests for the per-workspace run lock that serializes agent turns."""

import os
import tempfile
import unittest

from Cozter import workspace


class RunLockTests(unittest.TestCase):
    def test_workspace_aliases_share_file_and_run_locks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "workspace")
            os.mkdir(target)
            dotted_alias = os.path.join(target, ".")

            self.assertIsNot(
                workspace.get_run_lock(target), workspace.get_lock(target),
            )
            self.assertIsNot(
                workspace.get_run_lock(target), workspace.get_run_lock(tmp),
            )
            self.assertIs(
                workspace.get_lock(target), workspace.get_lock(dotted_alias),
            )
            self.assertIs(
                workspace.get_run_lock(target),
                workspace.get_run_lock(dotted_alias),
            )

            link = os.path.join(tmp, "workspace-link")
            try:
                os.symlink(target, link, target_is_directory=True)
            except (AttributeError, OSError) as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            self.assertIs(
                workspace.get_lock(target), workspace.get_lock(link),
            )
            self.assertIs(
                workspace.get_run_lock(target), workspace.get_run_lock(link),
            )


if __name__ == "__main__":
    unittest.main()
