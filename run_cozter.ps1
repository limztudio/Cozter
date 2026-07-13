# Restartable Windows Task Scheduler entry point for Cozter.
#
# Configure Task Scheduler to run this script through powershell.exe.  It
# keeps the scheduled task alive while Cozter runs and restarts the venv
# process after clean updates or failures.

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$packageParent = Split-Path -Parent $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$restartDelaySeconds = 5

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Cozter virtual-environment Python was not found: $python"
}

Set-Location -LiteralPath $packageParent
while ($true) {
    & $python -m Cozter
    $exitCode = $LASTEXITCODE
    Write-Warning (
        "Cozter exited with code $exitCode; restarting in " +
        "$restartDelaySeconds seconds."
    )
    Start-Sleep -Seconds $restartDelaySeconds
}
