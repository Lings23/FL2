param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$freeze025 = Join-Path $workspace 'config\rtc_v3_v1_lie_z025.freeze.json'
$freezeMain = Join-Path $workspace 'config\rtc_v3_v1_lie_z05_dba.freeze.json'
$outputs = @{
    clean = Join-Path $workspace 'logs\rtc_v3_v1_clean_seeds42_46_47_mf03'
    lie025 = Join-Path $workspace 'logs\rtc_v3_v1_lie_z025_seeds42_46_47_mf03'
    seed42 = Join-Path $workspace 'logs\rtc_v3_v1_lie_z05_dba_seed42_multikrum_mf03'
    remaining = Join-Path $workspace 'logs\rtc_v3_v1_lie_z05_dba_seeds46_47_mf03'
}

$expectedManifestSha256 = @{
    clean = '44dd1af59a71171570980027640c4c3722438a3bc3e30a57b15428f62efd9a80'
    lie025 = '049b063b4bfe22565d8134770a4784c0cd44892c5b792462a6cdf888b7862e98'
    seed42 = '76819b66e3a881a1c4ff15946a4cf8f56f59338a2f8afb079254085942e56f07'
    remaining = '53fe8fedf3a4b2ffb03ca096594662944a3978c32f8cf69c2cedf6eeeae87c0f'
}

function Assert-V1Manifest {
    param(
        [string]$Name,
        [string]$Path,
        [int]$ExpectedCount,
        [string[]]$ExpectedAttacks,
        [string[]]$ExpectedDefenses,
        [int[]]$ExpectedSeeds
    )
    $manifestPath = Join-Path $Path 'experiment_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "V1 manifest not found: $manifestPath"
    }
    if ($expectedManifestSha256[$Name]) {
        $actual = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expectedManifestSha256[$Name]) {
            throw "V1 manifest hash changed for ${Name}: expected=$($expectedManifestSha256[$Name]) actual=$actual"
        }
    }
    # Windows PowerShell 5.1 has no -Depth parameter on ConvertFrom-Json.
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $specs = @($manifest.specs)
    if ($specs.Count -ne $ExpectedCount) {
        throw "Expected $ExpectedCount V1 specs for $Name, got $($specs.Count)"
    }
    foreach ($spec in $specs) {
        if ([string]$spec.attack -notin $ExpectedAttacks) {
            throw "Unexpected attack in ${Name}: $($spec.attack)"
        }
        if ([string]$spec.defense -notin $ExpectedDefenses) {
            throw "Unexpected defense in ${Name}: $($spec.defense)"
        }
        if ([int]$spec.seed -notin $ExpectedSeeds) {
            throw "Unexpected seed in ${Name}: $($spec.seed)"
        }
        if ([double]$spec.malicious_fraction -ne 0.3) {
            throw "Expected malicious_fraction=0.3 in $Name"
        }
        if ([int]$spec.krum_num_malicious -ne 3) {
            throw "Expected Byzantine budget f=3 in $Name"
        }
        if ([string]$spec.defense -eq 'multi_krum') {
            if ([string]$spec.defense_type -ne 'krum' -or [int]$spec.krum_num_to_select -ne 5) {
                throw "Unexpected Multi-Krum contract in $Name"
            }
        } else {
            if ([string]$spec.defense_type -ne 'rtc_full') {
                throw "Unexpected RTC defense type in $Name"
            }
            if ([double]$spec.custom_params.semantic_intervention_risk_floor -ne 0.5 -or
                [double]$spec.custom_params.cumulative_q_cap_power -ne 1.0 -or
                [double]$spec.custom_params.anchor_recycle_fraction -ne 0.51 -or
                [string]$spec.custom_params.anchor_recycle_weighting -ne 'accepted') {
                throw "Frozen B3R-F0.51 parameters drifted in $Name"
            }
            if ($null -ne $spec.custom_params.norm_clip_mad_k -or
                $null -ne $spec.custom_params.residual_rank_cap_top_k) {
                throw "Rejected B4/B5 parameter leaked into V1 RTC in $Name"
            }
        }
        if ([string]$spec.attack -eq 'none') {
            if ([string]$spec.attack_group -ne 'clean' -or
                [string]$spec.attack_parameter_status -ne 'not_applicable') {
                throw "Explicit clean contract drifted in $Name"
            }
        } elseif ([string]$spec.attack -eq 'lie') {
            $expectedZ = if ($Name -eq 'lie025') { 0.25 } else { 0.5 }
            if ([double]$spec.lie_z -ne $expectedZ) {
                throw "Unexpected LIE z in ${Name}: $($spec.lie_z)"
            }
        } elseif ([string]$spec.attack -eq 'dba') {
            if ([double]$spec.replacement_gain -ne 1.0 -or
                [double]$spec.poison_fraction -ne 0.3 -or
                -not [bool]$spec.dba_scale_update) {
                throw "Strong DBA contract drifted in $Name"
            }
        }
    }

    foreach ($attack in $ExpectedAttacks) {
        foreach ($seed in $ExpectedSeeds) {
            $paired = @($specs | Where-Object { $_.attack -eq $attack -and [int]$_.seed -eq $seed })
            if ($paired.Count -ne $ExpectedDefenses.Count) {
                throw "Incomplete V1 defense pair for name=$Name attack=$attack seed=$seed"
            }
            if (@($paired.trial_plan_hash | Select-Object -Unique).Count -ne 1) {
                throw "Trial-plan hash mismatch for name=$Name attack=$attack seed=$seed"
            }
            if (@($paired.attack_implementation_hash | Select-Object -Unique).Count -ne 1) {
                throw "Attack implementation hash mismatch for name=$Name attack=$attack seed=$seed"
            }
        }
    }
}

function Invoke-V1Matrix {
    param(
        [string]$Output,
        [string]$Freeze,
        [string]$Attacks,
        [string]$Defenses,
        [string]$Seeds
    )
    $arguments = @(
        '-m', 'experiments.core.run',
        '--profile', 'rtc-byzantine',
        '--rtc-v3-anchor-recycle-fraction', '0.51',
        '--byzantine-attack-freeze', $Freeze,
        '--attack-start-round', '11',
        '--attack-end-round', '-1',
        '--attacks', $Attacks,
        '--defenses', $Defenses,
        '--seeds', $Seeds,
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
        '--output', $Output
    )
    if (-not $Execute) {
        $arguments += '--dry-run'
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V1 matrix failed with exit code $LASTEXITCODE for output=$Output"
    }
}

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $freeze025)) { throw "Attack freeze not found: $freeze025" }
if (-not (Test-Path -LiteralPath $freezeMain)) { throw "Attack freeze not found: $freezeMain" }

Set-Location -LiteralPath $workspace
if ($Execute) {
    Assert-V1Manifest clean $outputs.clean 6 @('none') @('rtc_cumulative_q_cap_accepted_anchor','multi_krum') @(42,46,47)
    Assert-V1Manifest lie025 $outputs.lie025 6 @('lie') @('rtc_cumulative_q_cap_accepted_anchor','multi_krum') @(42,46,47)
    Assert-V1Manifest seed42 $outputs.seed42 2 @('lie','dba') @('multi_krum') @(42)
    Assert-V1Manifest remaining $outputs.remaining 8 @('lie','dba') @('rtc_cumulative_q_cap_accepted_anchor','multi_krum') @(46,47)
}

Invoke-V1Matrix $outputs.clean $freezeMain 'none' 'rtc_cumulative_q_cap_accepted_anchor,multi_krum' '42,46,47'
Invoke-V1Matrix $outputs.lie025 $freeze025 'lie' 'rtc_cumulative_q_cap_accepted_anchor,multi_krum' '42,46,47'
Invoke-V1Matrix $outputs.seed42 $freezeMain 'lie,dba' 'multi_krum' '42'
Invoke-V1Matrix $outputs.remaining $freezeMain 'lie,dba' 'rtc_cumulative_q_cap_accepted_anchor,multi_krum' '46,47'

Assert-V1Manifest clean $outputs.clean 6 @('none') @('rtc_cumulative_q_cap_accepted_anchor','multi_krum') @(42,46,47)
Assert-V1Manifest lie025 $outputs.lie025 6 @('lie') @('rtc_cumulative_q_cap_accepted_anchor','multi_krum') @(42,46,47)
Assert-V1Manifest seed42 $outputs.seed42 2 @('lie','dba') @('multi_krum') @(42)
Assert-V1Manifest remaining $outputs.remaining 8 @('lie','dba') @('rtc_cumulative_q_cap_accepted_anchor','multi_krum') @(46,47)

if ($Execute) {
    Write-Output 'V1 new experiment cells completed. Reused RTC seed42 cells were intentionally not rerun.'
} else {
    Write-Output 'V1 dry-run manifests generated: 22 new cells across seeds 42/46/47; 2 completed RTC seed42 cells will be reused by the analyzer.'
    Write-Output 'Run again with -Execute only after the manifest hashes are frozen.'
}
