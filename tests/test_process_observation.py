"""Read-only process sampling rejects recycled identities and unknown counters."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import process_observation as observation


def sample(pid: int, parent: int, *, started: int | None = None, cpu: float | None = 1.0,
           reads: int | None = 20, writes: int | None = 30) -> observation.ProcessSample:
    return observation.ProcessSample(observation.ProcessIdentity(pid, f"windows:{started if started is not None else pid * 100}"),
                                     parent, f"process-{pid}", cpu, reads, writes, 2, 3)


class ProcessObservationTests(unittest.TestCase):
    @contextmanager
    def fixture(self, samples, *, errors=None):
        table = {pid: (value.parent_pid, value.name) for pid, value in samples.items()}
        with (mock.patch.object(observation, "_sample_process", side_effect=lambda pid: (samples.get(pid), "")),
              mock.patch.object(observation, "_process_table", return_value=(table, errors or []))):
            yield

    def test_tree_contains_only_owned_descendants_in_parent_first_order(self) -> None:
        samples = {1: sample(1, 0), 2: sample(2, 1), 3: sample(3, 2), 4: sample(4, 0)}
        with self.fixture(samples):
            snapshot = observation.snapshot_process_tree(1, samples[1].identity)
        self.assertEqual(snapshot.root, samples[1].identity)
        self.assertEqual([value.identity.pid for value in snapshot.processes], [1, 2, 3])
        self.assertEqual(snapshot.error, "")

    def test_root_pid_reuse_or_invalid_input_returns_no_owned_processes(self) -> None:
        root = sample(5, 0)
        with self.fixture({5: root}):
            for wrong in (observation.ProcessIdentity(5, "windows:old"), observation.ProcessIdentity(4, root.identity.started)):
                snapshot = observation.snapshot_process_tree(5, wrong)
                self.assertIsNone(snapshot.root)
                self.assertEqual(snapshot.processes, ())
                self.assertIn("identity changed", snapshot.error)
            for invalid in (True, False, 0, -1, "5", None):
                snapshot = observation.snapshot_process_tree(invalid)
                self.assertIsNone(snapshot.root)
                self.assertEqual(snapshot.processes, ())

    def test_identity_alive_requires_same_live_creation_identity(self) -> None:
        root = sample(5, 0)
        with self.fixture({5: root}):
            self.assertTrue(observation.identity_alive(root.identity))
            self.assertFalse(observation.identity_alive(observation.ProcessIdentity(5, "windows:old")))
            self.assertFalse(observation.identity_alive(observation.ProcessIdentity(6, root.identity.started)))

    def test_root_exiting_during_snapshot_invalidates_entire_result(self) -> None:
        root = sample(5, 0)
        with (mock.patch.object(observation, "_sample_process", side_effect=[(root, ""), (None, "")]),
              mock.patch.object(observation, "_process_table", return_value=({5: (0, "root")}, []))):
            snapshot = observation.snapshot_process_tree(5)
        self.assertIsNone(snapshot.root)
        self.assertEqual(snapshot.processes, ())
        self.assertIn("exited or changed", snapshot.error)

    def test_child_born_before_parent_is_not_owned_even_when_parent_pid_matches(self) -> None:
        samples = {5: sample(5, 0, started=500), 6: sample(6, 5, started=400)}
        with self.fixture(samples):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual([value.identity.pid for value in snapshot.processes], [5])
        self.assertIn("ancestry changed", snapshot.error)

    def test_reparented_or_recycled_child_is_removed_on_final_recheck(self) -> None:
        root, child = sample(5, 0), sample(6, 5)
        first_table = {5: (0, "root"), 6: (5, "child")}
        second_table = {5: (0, "root"), 6: (99, "foreign")}
        with (mock.patch.object(observation, "_sample_process", side_effect=lambda pid: ({5: root, 6: child}.get(pid), "")),
              mock.patch.object(observation, "_process_table", side_effect=[(first_table, []), (second_table, [])])):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual([value.identity.pid for value in snapshot.processes], [5])
        self.assertIn("ancestry changed", snapshot.error)

    def test_sampled_child_parent_must_match_enumerated_ancestry(self) -> None:
        root, child = sample(5, 0), sample(6, 99)
        with (mock.patch.object(observation, "_sample_process", side_effect=lambda pid: ({5: root, 6: child}.get(pid), "")),
              mock.patch.object(observation, "_process_table", return_value=({5: (0, "root"), 6: (5, "child")}, []))):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual([value.identity.pid for value in snapshot.processes], [5])
        self.assertIn("parent changed", snapshot.error)

    def test_missing_counters_stay_unknown_in_partial_snapshot(self) -> None:
        root = sample(5, 0, cpu=None, reads=None, writes=None)
        with (mock.patch.object(observation, "_sample_process", return_value=(root, "Counters unavailable")),
              mock.patch.object(observation, "_process_table", return_value=({5: (0, "root")}, []))):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual(snapshot.root, root.identity)
        self.assertIsNone(snapshot.processes[0].cpu_seconds)
        self.assertIsNone(snapshot.processes[0].read_bytes)
        self.assertIsNone(snapshot.processes[0].write_bytes)
        self.assertIn("Counters unavailable", snapshot.error)

    def test_permission_denied_child_is_unknown_not_zero_activity(self) -> None:
        root = sample(5, 0)

        def read(pid):
            return (root, "") if pid == 5 else (None, "Process identity unavailable for PID 6")

        with (mock.patch.object(observation, "_sample_process", side_effect=read),
              mock.patch.object(observation, "_process_table", return_value=({5: (0, "root"), 6: (5, "child")}, []))):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual(len(snapshot.processes), 1)
        self.assertIn("unavailable", snapshot.error)

    def test_enumeration_failure_keeps_root_with_explicit_unknown_ancestry(self) -> None:
        root = sample(5, 0)
        with (mock.patch.object(observation, "_sample_process", return_value=(root, "")),
              mock.patch.object(observation, "_process_table", side_effect=PermissionError)):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual(snapshot.root, root.identity)
        self.assertEqual(snapshot.processes, (root,))
        self.assertIn("ancestry unavailable", snapshot.error)

    def test_tree_is_bounded_and_truncation_is_explicit(self) -> None:
        samples = {1: sample(1, 0), **{pid: sample(pid, 1) for pid in range(2, 800)}}
        with self.fixture(samples):
            snapshot = observation.snapshot_process_tree(1)
        self.assertEqual(len(snapshot.processes), observation.MAX_TREE_PROCESSES)
        self.assertIn("limit reached", snapshot.error)

    def test_cyclic_parent_metadata_does_not_repeat_or_own_older_ancestor(self) -> None:
        samples = {5: sample(5, 6), 6: sample(6, 5)}
        with self.fixture(samples):
            snapshot = observation.snapshot_process_tree(5)
        self.assertEqual([value.identity.pid for value in snapshot.processes], [5, 6])

    @unittest.skipUnless(os.name == "nt" or sys.platform.startswith("linux"), "Native observer supports Windows and Linux")
    def test_real_owned_child_identity_and_counters_without_command_line(self) -> None:
        executable = getattr(sys, "_base_executable", sys.executable)
        child = subprocess.Popen([executable, "-u", "-c", "import sys; print('ready', flush=True); sys.stdin.buffer.read(1)"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            self.assertEqual(child.stdout.readline().strip(), b"ready")
            snapshot = observation.snapshot_process_tree(child.pid)
            self.assertIsNotNone(snapshot.root, snapshot.error)
            self.assertEqual(snapshot.root.pid, child.pid)
            self.assertTrue(observation.identity_alive(snapshot.root))
            process = next(value for value in snapshot.processes if value.identity.pid == child.pid)
            self.assertIsNotNone(process.cpu_seconds)
            self.assertIsNotNone(process.read_bytes)
            self.assertIsNotNone(process.write_ops)
            self.assertTrue(process.name)
            wrong = observation.ProcessIdentity(child.pid, snapshot.root.started + "-old")
            self.assertEqual(observation.snapshot_process_tree(child.pid, wrong).processes, ())
            self.assertFalse(hasattr(process, "command_line"))
        finally:
            if child.stdin is not None:
                child.stdin.close()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            if child.stdout is not None:
                child.stdout.close()
        if snapshot.root is not None:
            self.assertFalse(observation.identity_alive(snapshot.root))


class LinuxProcFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        directory = self.root / "42"
        directory.mkdir()
        fields = ["S", "7", *("0" for _ in range(18))]
        fields[11], fields[12], fields[19] = "100", "50", "321"
        (directory / "stat").write_text("42 (name with ) parentheses) " + " ".join(fields), encoding="utf-8")
        (directory / "io").write_text("read_bytes: 1000\nwrite_bytes: 2000\nsyscr: 12\nsyscw: 34\n", encoding="ascii")
        self.enterContext(mock.patch.object(observation, "_PROC_ROOT", self.root))
        self.enterContext(mock.patch.object(observation, "_linux_boot_id", return_value="fixture-boot"))
        self.enterContext(mock.patch.object(observation.os, "sysconf", return_value=100, create=True))

    def test_linux_stat_handles_parentheses_and_maps_exact_cpu_io_fields(self) -> None:
        value, error = observation._linux_sample(42)
        self.assertEqual(error, "")
        self.assertEqual(value.identity, observation.ProcessIdentity(42, "linux:fixture-boot:321"))
        self.assertEqual(value.parent_pid, 7)
        self.assertEqual(value.name, "name with ) parentheses")
        self.assertEqual(value.cpu_seconds, 1.5)
        self.assertEqual((value.read_bytes, value.write_bytes, value.read_ops, value.write_ops), (1000, 2000, 12, 34))

    def test_missing_linux_io_is_unknown_and_does_not_erase_creation_identity(self) -> None:
        (self.root / "42" / "io").unlink()
        value, error = observation._linux_sample(42)
        self.assertEqual(value.identity.started, "linux:fixture-boot:321")
        self.assertIsNone(value.read_bytes)
        self.assertIsNone(value.read_ops)
        self.assertIn("I/O counters unavailable", error)

    def test_linux_enumeration_ignores_non_process_directories(self) -> None:
        (self.root / "not-a-process").mkdir()
        table, errors = observation._linux_process_table()
        self.assertEqual(table, {42: (7, "name with ) parentheses")})
        self.assertEqual(errors, [])

    def test_zombies_are_not_live_creation_identities(self) -> None:
        path = self.root / "42" / "stat"
        path.write_text(path.read_text().replace(") S ", ") Z "), encoding="utf-8")
        self.assertEqual(observation._linux_sample(42), (None, ""))


if __name__ == "__main__":
    unittest.main()
