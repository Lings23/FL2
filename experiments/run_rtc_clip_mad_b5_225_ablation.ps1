param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$freeze = Join-Path $workspace 'config\rtc_v3_clip_mad_b5_225.freeze.json'
$output = Join-Path $workspace 'logs\rtc_v3_clip_mad_b5_225_seed42_mf03'
$expectedManifestSha256 = '79e88dda7cb4f00a5ae4089def6dac84b227469cf671ac046deadc05ffaeeb0a'
$expectedTrialPlanHashes = @{
    lie = 'de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c'
    dba = '3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67'
}
$expectedAttackContractHashes = @{
    lie = '6889258e8237774a2046621cd389c2c1df76b5fb3b1848f4b4863718e77bd7ca'
    dba = '696e53f8ace709478a1357561b8532c7220d8c6861a7d39932cffdada14e0e2b'
}

function Assert-B5Manifest {
    $manifestPath = Join-Path $output 'experiment_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "B5 manifest not found: $manifestPath. Run this script once without -Execute."
    }
    if ($expectedManifestSha256) {
        $actual = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expectedManifestSha256) {
            throw "B5 manifest hash changed: expected=$expectedManifestSha256 actual=$actual"
        }
    }
    # Windows PowerShell 5.1 has no -Depth parameter on ConvertFrom-Json.
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $specs = @($manifest.specs)
    if ($specs.Count -ne 2) {
        throw "Expected exactly two B5 candidate specs"
    }
    foreach ($attack in @('lie', 'dba')) {
        $matches = @($specs | Where-Object { $_.attack -eq $attack })
        if ($matches.Count -ne 1) {
            throw "Expected exactly one B5 candidate spec for attack=$attack"
        }
        $spec = $matches[0]
        if ([string]$spec.defense -ne 'rtc_b5_clip_mad_225' -or [string]$spec.defense_type -ne 'rtc_full') {
            throw "Unexpected B5 defense contract for attack=$attack"
        }
        if ([double]$spec.malicious_fraction -ne 0.3 -or [int]$spec.seed -ne 42) {
            throw "Expected seed42 and malicious_fraction=0.3 for attack=$attack"
        }
        if ([double]$spec.custom_params.semantic_intervention_risk_floor -ne 0.5) {
            throw "Expected semantic_intervention_risk_floor=0.5 for attack=$attack"
        }
        if ([double]$spec.custom_params.cumulative_q_cap_power -ne 1.0) {
            throw "Expected cumulative_q_cap_power=1 for attack=$attack"
        }
        if ([double]$spec.custom_params.anchor_recycle_fraction -ne 0.51) {
            throw "Expected anchor_recycle_fraction=0.51 for attack=$attack"
        }
        if ([string]$spec.custom_params.anchor_recycle_weighting -ne 'accepted') {
            throw "Expected anchor_recycle_weighting=accepted for attack=$attack"
        }
        if ([double]$spec.custom_params.norm_clip_mad_k -ne 2.25) {
            throw "Expected norm_clip_mad_k=2.25 for attack=$attack"
        }
        if ($null -ne $spec.custom_params.residual_rank_cap_top_k) {
            throw "B5 must not inherit rejected B4 residual rank cap"
        }
        if ([string]$spec.trial_plan_hash -ne [string]$expectedTrialPlanHashes[$attack]) {
            throw "Trial-plan hash changed for attack=$attack"
        }
        if ([string]$spec.attack_contract_hash -ne [string]$expectedAttackContractHashes[$attack]) {
            throw "Attack-contract hash changed for attack=$attack"
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
    '--rtc-v3-anchor-recycle-fraction', '0.51',
    '--byzantine-attack-freeze', $freeze,
    '--attack-start-round', '11',
    '--attack-end-round', '-1',
    '--attacks', 'lie,dba',
    '--defenses', 'rtc_b5_clip_mad_225',
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
    Assert-B5Manifest
}

Set-Location -LiteralPath $workspace
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "RTC B5 candidate run failed with exit code $LASTEXITCODE"
}

Assert-B5Manifest
if ($Execute) {
    Write-Output "B5 experiments completed: $output"
} else {
    Write-Output "B5 dry-run matrix generated: $output"
    Write-Output 'Run again with -Execute to start exactly two candidate cells.'
}
