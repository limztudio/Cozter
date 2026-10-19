"""Read-only, bounded observations of one owned process tree.

Creation identities distinguish a live process from a recycled PID. Missing
counters remain unknown; observing a process never grants permission to stop it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys


MAX_TREE_PROCESSES = 512
_MAX_TABLE_PROCESSES = 65_536
_PROC_ROOT = Path("/proc")


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    started: str


@dataclass(frozen=True)
class ProcessSample:
    identity: ProcessIdentity
    parent_pid: int
    name: str
    cpu_seconds: float | None
    read_bytes: int | None
    write_bytes: int | None
    read_ops: int | None
    write_ops: int | None


@dataclass(frozen=True)
class ProcessSnapshot:
    root: ProcessIdentity | None
    processes: tuple[ProcessSample, ...]
    error: str = ""


@dataclass(frozen=True)
class _LinuxStat:
    parent_pid: int
    name: str
    state: str
    cpu_ticks: int
    started: int


def _linux_stat(pid: int) -> _LinuxStat:
    text = (_PROC_ROOT / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    opening, closing = text.find("("), text.rfind(")")
    if opening < 0 or closing <= opening or int(text[:opening].strip()) != pid:
        raise ValueError("Invalid process stat identity")
    fields = text[closing + 1:].split()
    sample = _LinuxStat(int(fields[1]), text[opening + 1:closing], fields[0],
                        int(fields[11]) + int(fields[12]), int(fields[19]))
    if sample.parent_pid < 0 or sample.cpu_ticks < 0 or sample.started < 0:
        raise ValueError("Invalid process counters")
    return sample


@lru_cache(maxsize=1)
def _linux_boot_id() -> str:
    value = (_PROC_ROOT / "sys" / "kernel" / "random" / "boot_id").read_text(encoding="ascii").strip()
    if not value or len(value) > 128:
        raise ValueError("Invalid boot identity")
    return value


def _linux_process_table() -> tuple[dict[int, tuple[int, str]], list[str]]:
    table: dict[int, tuple[int, str]] = {}
    errors: list[str] = []
    inspected = 0
    with os.scandir(_PROC_ROOT) as directories:
        for entry in directories:
            if not entry.name.isdecimal():
                continue
            if inspected >= _MAX_TABLE_PROCESSES:
                errors.append("Process enumeration limit reached")
                break
            inspected += 1
            pid = int(entry.name)
            try:
                sample = _linux_stat(pid)
                table[pid] = (sample.parent_pid, sample.name)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, IndexError):
                # Unreadable ancestry prevents claiming the tree is complete.
                if not errors:
                    errors.append("Some process ancestry is unavailable")
    return table, errors


def _linux_sample(pid: int) -> tuple[ProcessSample | None, str]:
    sample = _linux_stat(pid)
    if sample.state in {"Z", "X", "x"}:
        return None, ""
    identity = ProcessIdentity(pid, f"linux:{_linux_boot_id()}:{sample.started}")
    errors = []
    cpu_seconds = None
    try:
        ticks = getattr(os, "sysconf")("SC_CLK_TCK")
        if ticks <= 0:
            raise ValueError("Invalid CPU clock resolution")
        cpu_seconds = sample.cpu_ticks / ticks
    except (OSError, ValueError, AttributeError):
        errors.append(f"CPU counters unavailable for PID {pid}")
    counters: dict[str, int] = {}
    try:
        for line in (_PROC_ROOT / str(pid) / "io").read_text(encoding="ascii").splitlines():
            name, value = line.split(":", 1)
            counters[name] = int(value.strip())
    except (OSError, ValueError):
        errors.append(f"I/O counters unavailable for PID {pid}")
    fields = tuple(counters.get(name) for name in ("read_bytes", "write_bytes", "syscr", "syscw"))
    if any(value is None for value in fields) and not errors:
        errors.append(f"Incomplete I/O counters for PID {pid}")
    return ProcessSample(identity, sample.parent_pid, sample.name, cpu_seconds, *fields), "; ".join(errors)


class _WindowsApi:
    """Small Win32 query binding, initialized lazily on Windows only."""

    def __init__(self) -> None:
        import ctypes

        class FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes",
            )]

        class ProcessEntry(ctypes.Structure):
            _fields_ = [
                ("size", ctypes.c_uint32), ("usage", ctypes.c_uint32), ("pid", ctypes.c_uint32),
                ("heap", ctypes.c_size_t), ("module", ctypes.c_uint32), ("threads", ctypes.c_uint32),
                ("parent_pid", ctypes.c_uint32), ("priority", ctypes.c_int32), ("flags", ctypes.c_uint32),
                ("name", ctypes.c_wchar * 260),
            ]

        self.FileTime, self.IoCounters, self.ProcessEntry = FileTime, IoCounters, ProcessEntry
        kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
        kernel.CreateToolhelp32Snapshot.argtypes = [dword, dword]
        kernel.CreateToolhelp32Snapshot.restype = handle
        kernel.Process32FirstW.argtypes = [handle, ctypes.POINTER(ProcessEntry)]
        kernel.Process32FirstW.restype = boolean
        kernel.Process32NextW.argtypes = [handle, ctypes.POINTER(ProcessEntry)]
        kernel.Process32NextW.restype = boolean
        kernel.OpenProcess.argtypes = [dword, boolean, dword]
        kernel.OpenProcess.restype = handle
        kernel.GetProcessTimes.argtypes = [handle, *[ctypes.POINTER(FileTime)] * 4]
        kernel.GetProcessTimes.restype = boolean
        kernel.GetProcessIoCounters.argtypes = [handle, ctypes.POINTER(IoCounters)]
        kernel.GetProcessIoCounters.restype = boolean
        kernel.GetExitCodeProcess.argtypes = [handle, ctypes.POINTER(dword)]
        kernel.GetExitCodeProcess.restype = boolean
        kernel.CloseHandle.argtypes = [handle]
        kernel.CloseHandle.restype = boolean
        self.kernel = kernel


@lru_cache(maxsize=1)
def _windows_api() -> _WindowsApi:
    return _WindowsApi()


def _windows_process_table() -> tuple[dict[int, tuple[int, str]], list[str]]:
    import ctypes

    api = _windows_api()
    snapshot = api.kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot is None or snapshot == ctypes.c_void_p(-1).value:
        raise OSError("Process enumeration unavailable")
    table: dict[int, tuple[int, str]] = {}
    errors: list[str] = []
    try:
        entry = api.ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        present = api.kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while present:
            if len(table) >= _MAX_TABLE_PROCESSES:
                errors.append("Process enumeration limit reached")
                break
            table[entry.pid] = (entry.parent_pid, entry.name)
            present = api.kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if not present and getattr(ctypes, "get_last_error")() != 18:  # ERROR_NO_MORE_FILES
            errors.append("Process enumeration is incomplete")
    finally:
        api.kernel.CloseHandle(snapshot)
    return table, errors


def _windows_sample(pid: int) -> tuple[ProcessSample | None, str]:
    import ctypes

    api = _windows_api()
    handle = api.kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise OSError("Process identity unavailable")
    try:
        code = ctypes.c_uint32()
        if not api.kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError("Process state unavailable")
        if code.value != 259:  # STILL_ACTIVE
            return None, ""
        created, exited, kernel, user = (api.FileTime() for _ in range(4))
        if not api.kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError("Process creation identity unavailable")
        started = (created.high << 32) | created.low
        cpu = (((kernel.high << 32) | kernel.low) + ((user.high << 32) | user.low)) / 10_000_000
        identity = ProcessIdentity(pid, f"windows:{started}")
        counters = api.IoCounters()
        if not api.kernel.GetProcessIoCounters(handle, ctypes.byref(counters)):
            return ProcessSample(identity, -1, "", cpu, None, None, None, None), f"I/O counters unavailable for PID {pid}"
        return ProcessSample(identity, -1, "", cpu, counters.read_bytes, counters.write_bytes,
                             counters.read_ops, counters.write_ops), ""
    finally:
        api.kernel.CloseHandle(handle)


def _process_table() -> tuple[dict[int, tuple[int, str]], list[str]]:
    if os.name == "nt":
        return _windows_process_table()
    if sys.platform.startswith("linux"):
        return _linux_process_table()
    raise OSError("Process observation is unavailable on this platform")


def _sample_process(pid: int) -> tuple[ProcessSample | None, str]:
    try:
        if os.name == "nt":
            return _windows_sample(pid)
        if sys.platform.startswith("linux"):
            return _linux_sample(pid)
        return None, "Process observation is unavailable on this platform"
    except (FileNotFoundError, ProcessLookupError):
        return None, ""
    except (OSError, ValueError, IndexError):
        return None, f"Process identity unavailable for PID {pid}"


def identity_alive(identity: ProcessIdentity) -> bool:
    if (not isinstance(identity, ProcessIdentity) or isinstance(identity.pid, bool)
            or not isinstance(identity.pid, int) or identity.pid <= 0
            or not isinstance(identity.started, str) or not identity.started):
        return False
    sample, _error = _sample_process(identity.pid)
    return sample is not None and sample.identity == identity


def _born_after(child: ProcessIdentity, parent: ProcessIdentity) -> bool:
    try:
        child_domain, child_time = child.started.rsplit(":", 1)
        parent_domain, parent_time = parent.started.rsplit(":", 1)
        return child_domain == parent_domain and int(child_time) >= int(parent_time)
    except (ValueError, TypeError):
        return False


def snapshot_process_tree(pid: int, expected: ProcessIdentity | None = None) -> ProcessSnapshot:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return ProcessSnapshot(None, (), "Invalid root PID")
    first, error = _sample_process(pid)
    if first is None:
        return ProcessSnapshot(None, (), error)
    if expected is not None and first.identity != expected:
        return ProcessSnapshot(None, (), "Root process identity changed")
    errors = [error] if error else []
    try:
        table, table_errors = _process_table()
        errors.extend(table_errors)
    except OSError:
        table = {}
        errors.append("Process ancestry unavailable")
    children: dict[int, list[int]] = {}
    for child_pid, (parent_pid, _name) in table.items():
        if child_pid > 0 and child_pid != parent_pid:
            children.setdefault(parent_pid, []).append(child_pid)
    samples: list[ProcessSample] = []
    pending: list[tuple[ProcessSample, ProcessIdentity | None]] = [(first, None)]
    seen: set[int] = set()
    while pending:
        sample, parent = pending.pop()
        current_pid = sample.identity.pid
        if current_pid in seen:
            continue
        seen.add(current_pid)
        if parent is not None and not _born_after(sample.identity, parent):
            errors.append("Process ancestry changed during observation")
            continue
        table_parent, table_name = table.get(current_pid, (sample.parent_pid, sample.name))
        if sample.parent_pid >= 0 and sample.parent_pid != table_parent:
            errors.append("Process parent changed during observation")
            continue
        samples.append(ProcessSample(sample.identity, table_parent, table_name, sample.cpu_seconds,
                                     sample.read_bytes, sample.write_bytes, sample.read_ops, sample.write_ops))
        for child_pid in sorted(children.get(current_pid, ()), reverse=True):
            if child_pid in seen:
                continue
            if len(samples) + len(pending) >= MAX_TREE_PROCESSES:
                if "Process tree limit reached" not in errors:
                    errors.append("Process tree limit reached")
                break
            child, child_error = _sample_process(child_pid)
            if child_error:
                errors.append(child_error)
            if child is not None:
                pending.append((child, sample.identity))
    if not identity_alive(first.identity):
        return ProcessSnapshot(None, (), "Root process exited or changed during observation")
    try:
        current_table, current_errors = _process_table()
        errors.extend(current_errors)
    except OSError:
        current_table = {}
        errors.append("Process ancestry recheck unavailable")
    validated: list[ProcessSample] = []
    validated_pids = {pid}
    for sample in samples:
        sample_pid = sample.identity.pid
        if sample_pid == pid:
            validated.append(sample)
            continue
        current_parent = current_table.get(sample_pid)
        if (current_parent is None or current_parent[0] != sample.parent_pid
                or sample.parent_pid not in validated_pids
                or not identity_alive(sample.identity)):
            errors.append("Process ancestry changed during observation")
            continue
        validated.append(sample)
        validated_pids.add(sample_pid)
    return ProcessSnapshot(first.identity, tuple(validated), "; ".join(dict.fromkeys(errors))[:2_048])
