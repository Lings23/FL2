param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$freeze = Join-Path $workspace 'config\rtc_v3_residual_rank_cap_b4.freeze.json'
$baselineOutput = Join-Path $workspace 'logs\rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03'
$candidateOutput = Join-Path $workspace 'logs\rtc_v3_residual_rank_cap_b4_seed42_mf03'
$expectedBaselineManifestSha256 = '2a7897330049d5eb7af162a2138872ca85a11599aede8f37bd79dc52aeb53dff'
$expectedCandidateManifestSha256 = 'fd0ea03b17171b5448fe2d3edb707822f9aad11094f0b9b49812d26498af85af'
$expectedTrialPlanHashes = @{
    lie = 'de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c'
    dba = '3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67'
}
$expectedAttackContractHashes = @{
    lie = '6889258e8237774a2046621cd389c2c1df76b5fb3b1848f4b4863718e77bd7ca'
    dba = '696e53f8ace709478a1357561b8532c7220d8c6861a7d39932cffdada14e0e2b'
}

function Assert-CommonB4Spec {
    param(
        [Parameter(Mandatory=$true)]$Spec,
        [Parameter(Mandatory=$true)][string]$ExpectedDefense
    )

    $attack = [string]$Spec.attack
    if ([string]$Spec.defense -ne $ExpectedDefense) {
        throw "Unexpected B4 defense for attack=$attack`: expected=$ExpectedDefense actual=$($Spec.defense)"
    }
    if ([string]$Spec.defense_type -ne 'rtc_full') {
        throw "Expected defense_type=rtc_full for attack=$attack"
    }
    if ([double]$Spec.malicious_fraction -ne 0.3 -or [int]$Spec.seed -ne 42) {
        throw "Expected seed42 and malicious_fraction=0.3 for attack=$attack"
    }
    if ([double]$Spec.custom_params.semantic_intervention_risk_floor -ne 0.5) {
        throw "Expected semantic_intervention_risk_floor=0.5 for attack=$attack"
    }
    if ([double]$Spec.custom_params.cumulative_q_cap_power -ne 1.0) {
        throw "Expected cumulative_q_cap_power=1 for attack=$attack"
    }
    if ([double]$Spec.custom_params.anchor_recycle_fraction -ne 0.51) {
        throw "Expected anchor_recycle_fraction=0.51 for attack=$attack"
    }
    if ([string]$Spec.custom_params.anchor_recycle_weighting -ne 'accepted') {
        throw "Expected anchor_recycle_weighting=accepted for attack=$attack"
    }
    if ([string]$Spec.trial_plan_hash -ne [string]$expectedTrialPlanHashes[$attack]) {
        throw "Trial-plan hash changed for attack=$attack"
    }
    if ([string]$Spec.attack_contract_hash -ne [string]$expectedAttackContractHashes[$attack]) {
        throw "Attack-contract hash changed for attack=$attack"
    }
}

function Assert-B4Manifests {
    $baselineManifestPath = Join-Path $baselineOutput 'experiment_manifest.json'
    $candidateManifestPath = Join-Path $candidateOutput 'experiment_manifest.json'
    foreach ($path in @($baselineManifestPath, $candidateManifestPath)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "B4 manifest not found: $path. Run this script once without -Execute."
        }
    }

    if ($expectedBaselineManifestSha256) {
        $actual = (Get-FileHash -LiteralPath $baselineManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expectedBaselineManifestSha256) {
            throw "B4 LIE baseline manifest hash changed: expected=$expectedBaselineManifestSha256 actual=$actual"
        }
    }
    if ($expectedCandidateManifestSha256) {
        $actual = (Get-FileHash -LiteralPath $candidateManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expectedCandidateManifestSha256) {
            throw "B4 candidate manifest hash changed: expected=$expectedCandidateManifestSha256 actual=$actual"
        }
    }

    # Windows PowerShell 5.1 has no -Depth parameter on ConvertFrom-Json.
    $baselineManifest = Get-Content -LiteralPath $baselineManifestPath -Raw | ConvertFrom-Json
    $candidateManifest = Get-Content -LiteralPath $candidateManifestPath -Raw | ConvertFrom-Json
    $baselineSpecs = @($baselineManifest.specs)
    $candidateSpecs = @($candidateManifest.specs)
    if ($baselineSpecs.Count -ne 1 -or [string]$baselineSpecs[0].attack -ne 'lie') {
        throw "Expected exactly one LIE B3R-F0.51 baseline spec"
    }
    if ($candidateSpecs.Count -ne 2) {
        throw "Expected exactly two B4 candidate specs"
    }

    Assert-CommonB4Spec -Spec $baselineSpecs[0] -ExpectedDefense 'rtc_cumulative_q_cap_accepted_anchor'
    if ($null -ne $baselineSpecs[0].custom_params.residual_rank_cap_top_k) {
        throw "B4 LIE baseline must not enable residual rank cap"
    }
    foreach ($attack in @('lie', 'dba')) {
        $matches = @($candidateSpecs | Where-Object { $_.attack -eq $attack })
        if ($matches.Count -ne 1) {
            throw "Expected exactly one B4 candidate spec for attack=$attack"
        }
        $spec = $matches[0]
        Assert-CommonB4Spec -Spec $spec -ExpectedDefense 'rtc_b4_residual_rank_cap'
        if ([int]$spec.custom_params.residual_rank_cap_top_k -ne 2) {
            throw "Expected residual_rank_cap_top_k=2 for attack=$attack"
        }
        if ([double]$spec.custom_params.residual_rank_cap_factor -ne 0.5) {
            throw "Expected residual_rank_cap_factor=0.5 for attack=$attack"
        }
        if ([double]$spec.custom_params.residual_rank_recycle_fraction -ne 1.0) {
            throw "Expected residual_rank_recycle_fraction=1 for attack=$attack"
        }
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $freeze)) {
    throw "Attack freeze not found: $freeze"
}

$commonArguments = @(
    '-m', 'experiments.core.run',
    '--profile', 'rtc-byzantine',
    '--rtc-v3-anchor-recycle-fraction', '0.51',
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
    '--skip-clean'
)
$baselineArguments = @($commonArguments) + @(
    '--attacks', 'lie',
    '--defenses', 'rtc_cumulative_q_cap_accepted_anchor',
    '--output', $baselineOutput
)
$candidateArguments = @($commonArguments) + @(
    '--attacks', 'lie,dba',
    '--defenses', 'rtc_b4_residual_rank_cap',
    '--output', $candidateOutput
)
if (-not $Execute) {
    $baselineArguments += '--dry-run'
    $candidateArguments += '--dry-run'
}
if ($Execute) {
    Assert-B4Manifests
}

Set-Location -LiteralPath $workspace
& $python @baselineArguments
if ($LASTEXITCODE -ne 0) {
    throw "RTC B4 LIE baseline failed with exit code $LASTEXITCODE"
}
& $python @candidateArguments
if ($LASTEXITCODE -ne 0) {
    throw "RTC B4 candidate run failed with exit code $LASTEXITCODE"
}

Assert-B4Manifests
if ($Execute) {
    Write-Output "B4 experiments completed: baseline=$baselineOutput candidate=$candidateOutput"
} else {
    Write-Output "B4 dry-run matrices generated: baseline=$baselineOutput candidate=$candidateOutput"
    Write-Output 'Run again with -Execute to start exactly three cells (one LIE baseline plus two B4 candidates).'
}
