# Restartable Task Scheduler entry: keeps Cozter alive, restarts venv after updates/failures.

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$packageParent = Split-Path -Parent $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$restartDelaySeconds = 5

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Cozter virtual-environment Python was not found: $python"
}

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
