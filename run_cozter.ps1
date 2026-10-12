# Restartable Task Scheduler entry: keeps Cozter alive, restarts venv after updates/failures.

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$packageParent = Split-Path -Parent $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$restartDelaySeconds = 5

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Cozter virtual-environment Python was not found: $python"
}

# The scheduled entry is this PowerShell process. Own the job here, before
# Python starts, so terminating the supervisor also terminates its descendants.
# A fresh nested job is necessary even if this launch inherited another marker.
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace Cozter
{
    public static class SupervisorLaunchJob
    {
        // Retain the noninheritable handle until Windows tears down this process.
        // Closing it explicitly would also terminate this supervisor.
        private static IntPtr jobHandle = IntPtr.Zero;
        private static string jobName;
        private static readonly object gate = new object();

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimitInformation
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimitInformation
        {
            public BasicLimitInformation BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(
            IntPtr job, int informationClass,
            ref ExtendedLimitInformation information, uint length);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();

        [DllImport("kernel32.dll")]
        private static extern uint GetCurrentProcessId();

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);

        private static Exception Failure(string operation)
        {
            int code = Marshal.GetLastWin32Error();
            return new InvalidOperationException(
                "Cannot protect Cozter's supervisor process tree: " + operation +
                " failed (Windows error " + code + ": " +
                new Win32Exception(code).Message + "). " +
                "An existing scheduler or parent job may prohibit nested jobs. " +
                "Startup stopped before spawning Python to avoid orphaned processes.");
        }

        public static string Ensure()
        {
            lock (gate)
            {
                if (jobHandle != IntPtr.Zero)
                    return jobName;

                string name = "Local\\CozterLaunch-" + GetCurrentProcessId() +
                    "-" + Guid.NewGuid().ToString("N");
                // NULL security attributes make the handle noninheritable.
                IntPtr handle = CreateJobObjectW(IntPtr.Zero, name);
                if (handle == IntPtr.Zero)
                    throw Failure("CreateJobObjectW");
                if (Marshal.GetLastWin32Error() == 183)
                {
                    CloseHandle(handle);
                    throw new InvalidOperationException(
                        "Cozter's unique supervisor job already exists; startup stopped.");
                }
                try
                {
                    var information = new ExtendedLimitInformation();
                    information.BasicLimitInformation.LimitFlags = 0x2000;
                    if (!SetInformationJobObject(
                        handle, 9, ref information,
                        (uint)Marshal.SizeOf(typeof(ExtendedLimitInformation))))
                        throw Failure("SetInformationJobObject");
                    if (!AssignProcessToJobObject(handle, GetCurrentProcess()))
                        throw Failure("AssignProcessToJobObject");
                }
                catch
                {
                    // Assignment did not succeed, so closing cannot kill us.
                    CloseHandle(handle);
                    throw;
                }
                jobHandle = handle;
                jobName = name;
                return name;
            }
        }
    }
}
'@

$env:COZTER_LAUNCH_JOB = [Cozter.SupervisorLaunchJob]::Ensure()
Set-Location -LiteralPath $packageParent
$env:COZTER_WINDOWS_SUPERVISED = "1"
while ($true) {
    & $python -m Cozter
    $exitCode = $LASTEXITCODE
    Write-Warning (
        "Cozter exited with code $exitCode; restarting in " +
        "$restartDelaySeconds seconds."
    )
    Start-Sleep -Seconds $restartDelaySeconds
}
