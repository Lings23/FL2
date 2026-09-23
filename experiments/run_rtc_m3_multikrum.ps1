param(
    [switch]$Execute,
    [switch]$Analyze,
    [string]$Python,
    [string]$Output,
    [string]$Parent,
    [string]$DataDir
)
$ErrorActionPreference = 'Stop'
if ($Execute -and $Analyze) { throw 'Choose Execute or Analyze, not both.' }
if ($DataDir -and ($Execute -or $Analyze)) { throw 'DataDir is preparation-only.' }
$workspace = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $workspace '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$arguments = @('-B', (Join-Path $PSScriptRoot 'rtc_m3_multikrum.py'))
if ($Execute) { $arguments += '--execute' }
if ($Analyze) { $arguments += '--analyze' }
if ($Output) { $arguments += @('--output', $Output) }
if ($Parent) { $arguments += @('--parent', $Parent) }
if ($DataDir) { $arguments += @('--data-dir', $DataDir) }
Push-Location -LiteralPath $workspace
try {
    & $Python @arguments
    $runExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $runExitCode
