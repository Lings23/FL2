param(
    [ValidateSet('calibration', 'validation', 'evaluation')][string]$Stage = 'calibration',
    [switch]$Execute,
    [switch]$Analyze,
    [switch]$RestartIncomplete,
    [string]$Python,
    [string]$Output,
    [string]$DataDir,
    [Nullable[double]]$ClientNumGpus
)
$ErrorActionPreference = 'Stop'
if ($Execute -and $Analyze) { throw 'Execute and Analyze are mutually exclusive.' }
$workspace = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $workspace '.venv\Scripts\python.exe' }
$arguments = @('-B', '-m', 'experiments.rtc_mnist_all_attacks', '--stage', $Stage)
if ($Execute) { $arguments += '--execute' }
if ($Analyze) { $arguments += '--analyze' }
if ($RestartIncomplete) { $arguments += '--restart-incomplete' }
if ($Output) { $arguments += @('--output', $Output) }
if ($DataDir) { $arguments += @('--data-dir', $DataDir) }
if ($null -ne $ClientNumGpus) {
    $arguments += @('--client-num-gpus', $ClientNumGpus.ToString([Globalization.CultureInfo]::InvariantCulture))
}
Push-Location -LiteralPath $workspace
try {
    & $Python @arguments
    $code = $LASTEXITCODE
} finally { Pop-Location }
exit $code
