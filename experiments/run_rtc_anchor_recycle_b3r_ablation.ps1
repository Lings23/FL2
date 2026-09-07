param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$freeze = Join-Path $workspace 'config\rtc_v3_anchor_recycle_b3r.freeze.json'
$output = Join-Path $workspace 'logs\rtc_v3_anchor_recycle_b3r_seed42_mf03'
$manifestPath = Join-Path $output 'experiment_manifest.json'
$expectedManifestSha256 = 'f7de91c35d85f3a05143639983e3e91de4e95746230b8c7d642af572c755abe8'
$expectedTrialPlanHashes = @{
    dba = '3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67'
    scaling_backdoor = 'd27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359'
}

function Assert-B3RManifest {
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "B3R manifest not found: $manifestPath. Run this script once without -Execute."
    }

    $actualManifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualManifestSha256 -ne $expectedManifestSha256) {
        throw "B3R manifest hash changed: expected=$expectedManifestSha256 actual=$actualManifestSha256"
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -Depth 100
    foreach ($attack in $expectedTrialPlanHashes.Keys) {
        $matches = @($manifest.specs | Where-Object { $_.attack -eq $attack })
        if ($matches.Count -ne 1) {
            throw "Expected exactly one B3R spec for attack=$attack, found=$($matches.Count)"
        }
        $actualTrialPlanHash = [string]$matches[0].trial_plan_hash
        $expectedTrialPlanHash = [string]$expectedTrialPlanHashes[$attack]
        if ($actualTrialPlanHash -ne $expectedTrialPlanHash) {
            throw "Trial-plan hash changed for attack=$attack`: expected=$expectedTrialPlanHash actual=$actualTrialPlanHash"
        }
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $freeze)) {
    throw "Attack freeze not found: $freeze"
}

$arguments = @(
    '-m', 'experiments.core.run',
    '--profile', 'rtc-byzantine',
    '--attacks', 'dba,scaling_backdoor',
    '--defenses', 'rtc_cumulative_q_cap_accepted_anchor',
    '--rtc-v3-anchor-recycle-fraction', '1.0',
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
if ($Execute) {
    Assert-B3RManifest
    Write-Output 'Resume mode: valid completed round caches will be reused; only incomplete cells will run.'
}

Set-Location -LiteralPath $workspace
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "RTC B3R accepted-anchor ablation failed with exit code $LASTEXITCODE"
}

Assert-B3RManifest

if ($Execute) {
    Write-Output "Experiment completed: $output"
} else {
    Write-Output "Dry-run matrix generated: $output"
    Write-Output 'Run again with -Execute to reuse valid completed cells and train only incomplete cells.'
}
