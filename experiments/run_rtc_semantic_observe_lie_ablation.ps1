param(
    [switch]$Execute,
    [switch]$Rerun
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$freeze = Join-Path $workspace 'config\rtc_v3_semantic_observe_lie_z05.freeze.json'
$output = Join-Path $workspace 'logs\rtc_v3_semantic_observe_lie_z05_seed42_mf03'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $freeze)) {
    throw "Attack freeze not found: $freeze"
}

$arguments = @(
    '-m', 'experiments.core.run',
    '--profile', 'rtc-byzantine',
    '--attacks', 'lie',
    '--defenses', 'rtc_full,rtc_semantic_observe',
    '--byzantine-attack-freeze', $freeze,
    '--attack-start-round', '11',
    '--attack-end-round', '-1',
    '--seeds', '42',
    '--malicious-fractions', '0.3',
    '--rounds', '60',
    '--num-clients', '20',
    '--participation-rate', '0.5',
    '--batch-size', '48',
    '--partition', 'iid',
    '--ray-client-num-cpus', '1',
    '--ray-client-num-gpus', '0.25',
    '--ray-object-store-memory-mb', '3072',
    '--ray-min-available-memory-mb', '10240',
    '--ray-memory-wait-seconds', '120',
    '--max-spec-retries', '1',
    '--skip-clean',
    '--output', $output
)

if (-not $Execute) {
    $arguments += '--dry-run'
}
if ($Rerun) {
    $arguments += '--rerun'
}

Set-Location -LiteralPath $workspace
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "RTC semantic-observe LIE ablation failed with exit code $LASTEXITCODE"
}

if ($Execute) {
    Write-Output "Experiment completed: $output"
} else {
    Write-Output "Dry-run matrix generated: $output"
    Write-Output 'Run again with -Execute to start the two paired training cells.'
}
