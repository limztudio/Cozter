"""Keep a Windows launcher's descendants tied to its process lifetime.

Call ``ensure_launch_job`` before the launcher spawns a module or venv
replacement. Assigning the current process first makes child membership
automatic, without the race in assigning an already-running child.
"""

from __future__ import annotations

import ctypes
import os
import threading
import uuid


_LAUNCH_JOB_ENV = "COZTER_LAUNCH_JOB"
_JOB_OBJECT_QUERY = 0x0004
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_ERROR_ALREADY_EXISTS = 183

# Keep open till Windows ends the owner; no atexit close.
_job_handle: int | None = None
_protected = False
_JOB_LOCK = threading.Lock()


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    """Load typed Win32 bindings lazily so importing is safe on any OS."""
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
        "OpenJobObjectW": (
            [ctypes.c_uint32, ctypes.c_int32, ctypes.c_wchar_p], ctypes.c_void_p,
        ),
        "SetInformationJobObject": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
            ctypes.c_int32,
        ),
        "AssignProcessToJobObject": (
            [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int32,
        ),
        "IsProcessInJob": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)],
            ctypes.c_int32,
        ),
        "GetCurrentProcess": ([], ctypes.c_void_p),
        "CloseHandle": ([ctypes.c_void_p], ctypes.c_int32),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
    return api


def _last_error() -> int:
    # These ctypes helpers exist only on Windows; resolve them lazily so
    # imports and static checks on other platforms remain safe.
    return int(getattr(ctypes, "get_last_error")())


def _job_error(operation: str) -> RuntimeError:
    code = _last_error()
    detail = str(getattr(ctypes, "FormatError")(code)).strip()
    return RuntimeError(
        "Cannot protect Cozter's launcher process tree: "
        f"{operation} failed (Windows error {code}: {detail}). "
        "An existing scheduler or parent job may prohibit nested jobs. "
        "Startup stopped before spawning replacements to avoid orphaned processes."
    )


def _belongs_to_launch_job(api, name: str) -> bool:
    """Verify an inherited marker without retaining the parent's job handle."""
    handle = api.OpenJobObjectW(_JOB_OBJECT_QUERY, False, name)
    if not handle:
        # A stale marker never disables protection; create a fresh job instead.
        return False
    try:
        belongs = ctypes.c_int32()
        if not api.IsProcessInJob(api.GetCurrentProcess(), handle, ctypes.byref(belongs)):
            raise _job_error("IsProcessInJob")
        return bool(belongs.value)
    finally:
        api.CloseHandle(handle)


def ensure_launch_job() -> bool:
    """Protect this Windows launch before any children are created.

    The owning process retains one noninheritable job handle. Reexecuted
    descendants verify their inherited job membership, then close the query
    handle immediately; retaining it would keep descendants alive after the
    launcher dies. Returns False on other operating systems. Windows API
    failure raises clearly instead of starting an unprotected process chain.
    """
    global _job_handle, _protected
    if os.name != "nt":
        return False
    with _JOB_LOCK:
        if _protected:
            return True
        api = _kernel32()
        marker = os.environ.get(_LAUNCH_JOB_ENV)
        if marker and _belongs_to_launch_job(api, marker):
            _protected = True
            return True

        name = f"Local\\CozterLaunch-{os.getpid()}-{uuid.uuid4().hex}"
        getattr(ctypes, "set_last_error")(0)
        # NULL security attributes mean the handle cannot be inherited.
        handle = api.CreateJobObjectW(None, name)
        if not handle:
            raise _job_error("CreateJobObjectW")
        if _last_error() == _ERROR_ALREADY_EXISTS:
            api.CloseHandle(handle)
            raise RuntimeError("Cozter's unique launcher job already exists; startup stopped.")
        try:
            information = _ExtendedLimitInformation()
            information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not api.SetInformationJobObject(
                handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(information), ctypes.sizeof(information),
            ):
                raise _job_error("SetInformationJobObject")
            if not api.AssignProcessToJobObject(handle, api.GetCurrentProcess()):
                raise _job_error("AssignProcessToJobObject")
        except BaseException:
            # Assignment has not succeeded, so closing cannot kill this process.
            api.CloseHandle(handle)
            raise

        _job_handle = handle
        _protected = True
        os.environ[_LAUNCH_JOB_ENV] = name
        return True
