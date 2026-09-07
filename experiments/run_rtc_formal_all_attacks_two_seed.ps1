param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$freeze = Join-Path $workspace 'config\rtc_v3_formal_all_attacks_two_seed.freeze.json'
$outputRoot = Join-Path $workspace 'logs\rtc_v3_formal_all_attacks_two_seed_mf03'
$lockPath = Join-Path $outputRoot 'manifest_lock.json'
$seeds = @(42, 46)
$cleanSeeds = $seeds
$cleanDefenses = @('fedavg')
$attacks = @(
    'none',
    'gaussian_noise',
    'random_noise',
    'sign_flip',
    'lie',
    'min_max',
    'min_sum',
    'scaling_backdoor',
    'label_flip_all_reverse',
    'dba'
)
$defenses = @(
    'fedavg',
    'rtc_cumulative_q_cap_accepted_anchor',
    'krum',
    'multi_krum',
    'trimmed_mean',
    'median',
    'foolsgold',
    'rfa',
    'freqfed',
    'fltrust'
)
$defenseTypes = @{
    fedavg = 'none'
    rtc_cumulative_q_cap_accepted_anchor = 'rtc_full'
    krum = 'krum'
    multi_krum = 'krum'
    trimmed_mean = 'trimmed_mean'
    median = 'median'
    foolsgold = 'foolsgold'
    rfa = 'rfa'
    freqfed = 'freqfed'
    fltrust = 'fltrust'
}
$codePaths = @(
    'config\rtc_v3_formal_all_attacks_two_seed.freeze.json',
    'experiments\run_rtc_formal_all_attacks_two_seed.ps1',
    'experiments\rtc_v3\byzantine.py',
    'experiments\periodic_attack.py',
    'experiments\trial_plan.py',
    'analysis\rtc_formal_all_attacks_two_seed\analyze_results.py',
    'attacks\spec.py',
    'defenses\defense_base.py',
    'defenses\rtc\v3.py'
)

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-AttackOutput {
    param([string]$Attack)
    return Join-Path $outputRoot $Attack
}

function Assert-ManifestContract {
    param(
        [string]$Attack,
        [string]$Output,
        [string]$ExpectedSha256 = ''
    )
    $manifestPath = Join-Path $Output 'experiment_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Manifest not found: $manifestPath"
    }
    $actualSha256 = Get-Sha256 $manifestPath
    if ($ExpectedSha256 -and $actualSha256 -ne $ExpectedSha256) {
        throw "Manifest hash changed for ${Attack}: expected=$ExpectedSha256 actual=$actualSha256"
    }
    # Compatible with Windows PowerShell 5.1: ConvertFrom-Json has no -Depth.
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $specs = @($manifest.specs)
    $attackSeeds = if ($Attack -eq 'none') { $cleanSeeds } else { $seeds }
    $attackDefenses = if ($Attack -eq 'none') { $cleanDefenses } else { $defenses }
    $expectedSpecs = $attackDefenses.Count * $attackSeeds.Count
    if ($specs.Count -ne $expectedSpecs) {
        throw "Expected $expectedSpecs specs for ${Attack}, got $($specs.Count)"
    }
    foreach ($spec in $specs) {
        if ([string]$spec.attack -ne $Attack) {
            throw "Unexpected attack in ${Attack} manifest: $($spec.attack)"
        }
        if ([string]$spec.defense -notin $attackDefenses) {
            throw "Unexpected defense in ${Attack}: $($spec.defense)"
        }
        if ([int]$spec.seed -notin $attackSeeds) {
            throw "Unexpected seed in ${Attack}: $($spec.seed)"
        }
        if ([string]$spec.defense_type -ne $defenseTypes[[string]$spec.defense]) {
            throw "Defense type drift for attack=$Attack defense=$($spec.defense)"
        }
        if ([double]$spec.malicious_fraction -ne 0.3 -or
            [string]$spec.partition -ne 'iid' -or
            [double]$spec.participation_rate -ne 0.5) {
            throw "Data/participation contract drift for attack=$Attack defense=$($spec.defense)"
        }
        if ([int]$spec.attack_start_round -ne 11 -or [int]$spec.attack_end_round -ne -1) {
            throw "Attack window drift for attack=$Attack defense=$($spec.defense)"
        }
        if ([int]$spec.krum_num_malicious -ne 3) {
            throw "Expected Byzantine budget f=3 for attack=$Attack"
        }
        if ([string]$spec.defense -eq 'multi_krum' -and [int]$spec.krum_num_to_select -ne 5) {
            throw "Expected Multi-Krum select=5 for attack=$Attack"
        }
        if ([string]$spec.defense -eq 'rtc_cumulative_q_cap_accepted_anchor') {
            if ([double]$spec.custom_params.semantic_intervention_risk_floor -ne 0.5 -or
                [double]$spec.custom_params.cumulative_q_cap_power -ne 1.0 -or
                [double]$spec.custom_params.anchor_recycle_fraction -ne 0.51 -or
                [string]$spec.custom_params.anchor_recycle_weighting -ne 'accepted') {
                throw "Frozen RTC B3R-F0.51 parameters drifted for attack=$Attack"
            }
            if ($null -ne $spec.custom_params.norm_clip_mad_k -or
                $null -ne $spec.custom_params.residual_rank_cap_top_k) {
                throw "Rejected RTC B4/B5 parameter leaked into attack=$Attack"
            }
        }
        if ($Attack -eq 'none') {
            if ([string]$spec.attack_group -ne 'clean' -or
                [string]$spec.attack_parameter_status -ne 'not_applicable') {
                throw "Explicit clean contract drifted"
            }
        } else {
            if ([string]$spec.attack_parameter_status -ne 'frozen') {
                throw "Attack parameters are not frozen for attack=$Attack"
            }
        }
        if ($Attack -eq 'lie' -and [double]$spec.lie_z -ne 0.5) {
            throw "Formal LIE strength must be z=0.5"
        }
        if ($Attack -eq 'sign_flip' -and [double]$spec.sign_flip_scale -ne 1.0) {
            throw "Formal Sign-flip strength must be the numerically valid scale=1"
        }
    }
    foreach ($seed in $attackSeeds) {
        $paired = @($specs | Where-Object { [int]$_.seed -eq $seed })
        if ($paired.Count -ne $attackDefenses.Count) {
            throw "Incomplete defense set for attack=$Attack seed=$seed"
        }
        $observedDefenses = @($paired.defense | Sort-Object -Unique)
        if (@(Compare-Object ($attackDefenses | Sort-Object) $observedDefenses).Count -ne 0) {
            throw "Defense coverage mismatch for attack=$Attack seed=$seed"
        }
        if (@($paired.trial_plan_hash | Select-Object -Unique).Count -ne 1) {
            throw "Trial-plan hash mismatch for attack=$Attack seed=$seed"
        }
        if (@($paired.attack_implementation_hash | Select-Object -Unique).Count -ne 1) {
            throw "Attack implementation hash mismatch for attack=$Attack seed=$seed"
        }
    }
    return $actualSha256
}

function Invoke-AttackMatrix {
    param([string]$Attack, [string]$Output, [switch]$DryRun)
    $attackSeeds = if ($Attack -eq 'none') { $cleanSeeds } else { $seeds }
    $attackDefenses = if ($Attack -eq 'none') { $cleanDefenses } else { $defenses }
    $arguments = @(
        '-m', 'experiments.core.run',
        '--profile', 'rtc-byzantine',
        '--rtc-v3-anchor-recycle-fraction', '0.51',
        '--byzantine-attack-freeze', $freeze,
        '--attack-start-round', '11',
        '--attack-end-round', '-1',
        '--attacks', $Attack,
        '--defenses', ($attackDefenses -join ','),
        '--seeds', ($attackSeeds -join ','),
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
    if ($DryRun) {
        $arguments += '--dry-run'
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Matrix failed with exit code $LASTEXITCODE for attack=$Attack output=$Output"
    }
}

function Assert-CompletedAttack {
    param([string]$Attack, [string]$Output)
    $attackSeeds = if ($Attack -eq 'none') { $cleanSeeds } else { $seeds }
    $attackDefenses = if ($Attack -eq 'none') { $cleanDefenses } else { $defenses }
    $expectedSpecs = $attackDefenses.Count * $attackSeeds.Count
    $statusFiles = @(Get-ChildItem -LiteralPath (Join-Path $Output 'status') -Filter '*.json' -File)
    $roundFiles = @(Get-ChildItem -LiteralPath (Join-Path $Output 'rounds') -Filter '*.csv' -File)
    if ($statusFiles.Count -ne $expectedSpecs -or $roundFiles.Count -ne $expectedSpecs) {
        throw "Incomplete files for attack=$Attack status=$($statusFiles.Count) rounds=$($roundFiles.Count)"
    }
    foreach ($statusFile in $statusFiles) {
        $status = Get-Content -LiteralPath $statusFile.FullName -Raw | ConvertFrom-Json
        if ([string]$status.state -notin @('completed', 'completed_cached') -or
            [int]$status.exit_code -ne 0 -or [int]$status.last_round -ne 60) {
            throw "Incomplete status: $($statusFile.FullName)"
        }
    }
    foreach ($roundFile in $roundFiles) {
        $rounds = @(Import-Csv -LiteralPath $roundFile.FullName)
        if ($rounds.Count -ne 61 -or [int]$rounds[0].round -ne 0 -or [int]$rounds[-1].round -ne 60) {
            throw "Rounds are not exactly 0..60: $($roundFile.FullName)"
        }
    }
    $gatePath = Join-Path $Output 'quality_gates.csv'
    if (-not (Test-Path -LiteralPath $gatePath)) {
        throw "Missing quality gates for attack=$Attack"
    }
    $gates = @(Import-Csv -LiteralPath $gatePath)
    $failed = @($gates | Where-Object { [string]$_.passed -notin @('True', 'true', '1') })
    if (-not $gates.Count -or $failed.Count) {
        throw "Quality gates failed for attack=${Attack}: $($failed.gate -join ',')"
    }
}

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $freeze)) { throw "Attack freeze not found: $freeze" }
Set-Location -LiteralPath $workspace

if (-not $Execute) {
    $existingRounds = @(Get-ChildItem -LiteralPath $outputRoot -Filter '*.csv' -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq 'rounds' })
    if ($existingRounds.Count) {
        throw "Refusing to replace the protocol lock after training outputs exist under $outputRoot"
    }
    $entries = @()
    foreach ($attack in $attacks) {
        $output = Get-AttackOutput $attack
        Invoke-AttackMatrix $attack $output -DryRun
        $sha = Assert-ManifestContract $attack $output
        $entries += [ordered]@{
            attack = $attack
            output = $output
            manifest_sha256 = $sha
            specs = if ($attack -eq 'none') { 2 } else { 20 }
        }
    }
    $hashes = [ordered]@{}
    foreach ($relative in $codePaths) {
        $hashes[$relative] = Get-Sha256 (Join-Path $workspace $relative)
    }
    $lock = [ordered]@{
        schema_version = 'RTCFormalAllAttacksTwoSeedLockV1'
        created_at = (Get-Date).ToUniversalTime().ToString('o')
        protocol = [ordered]@{
            seeds = $seeds
            clean_seeds = $cleanSeeds
            clean_defenses = $cleanDefenses
            attacks = $attacks
            defenses = $defenses
            malicious_fraction = 0.3
            rounds = 60
            attack_start_round = 11
            partition = 'iid'
            participation_rate = 0.5
            rtc_candidate = 'B3R-F0.51'
            lie_z = 0.5
            total_cells = 182
        }
        code_sha256 = $hashes
        manifests = $entries
    }
    New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
    $lock | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $lockPath -Encoding UTF8
    Write-Output "Dry-run complete: 10 manifests / 182 cells. Protocol lock: $lockPath"
    Write-Output "After review, execute: & '$PSCommandPath' -Execute"
    exit 0
}

if (-not (Test-Path -LiteralPath $lockPath)) {
    throw "Protocol lock not found. Run this script without -Execute first: $lockPath"
}
$locked = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
if ([string]$locked.schema_version -ne 'RTCFormalAllAttacksTwoSeedLockV1') {
    throw "Unsupported protocol lock schema"
}
if ([int]$locked.protocol.total_cells -ne 182 -or
    @($locked.protocol.clean_seeds).Count -ne 2 -or
    @(Compare-Object @($locked.protocol.clean_seeds | ForEach-Object { [int]$_ }) $seeds).Count -ne 0 -or
    @($locked.protocol.clean_defenses).Count -ne 1 -or
    [string]$locked.protocol.clean_defenses[0] -ne 'fedavg' -or
    [double]$locked.protocol.lie_z -ne 0.5) {
    throw "Protocol lock drifted from the approved 182-cell/one-clean-run-per-seed/LIE-z=0.5 design"
}
foreach ($relative in $codePaths) {
    $property = $locked.code_sha256.PSObject.Properties[$relative]
    $expected = if ($null -eq $property) { '' } else { [string]$property.Value }
    $actual = Get-Sha256 (Join-Path $workspace $relative)
    if (-not $expected -or $actual -ne $expected) {
        throw "Code hash changed after dry-run: $relative expected=$expected actual=$actual"
    }
}
foreach ($attack in $attacks) {
    $entry = @($locked.manifests | Where-Object { [string]$_.attack -eq $attack })
    if ($entry.Count -ne 1) {
        throw "Protocol lock must contain exactly one manifest for attack=$Attack"
    }
    $output = Get-AttackOutput $attack
    Assert-ManifestContract $attack $output ([string]$entry[0].manifest_sha256) | Out-Null
    Invoke-AttackMatrix $attack $output
    Assert-ManifestContract $attack $output ([string]$entry[0].manifest_sha256) | Out-Null
    Assert-CompletedAttack $attack $output
    $expectedSpecs = if ($attack -eq 'none') { 2 } else { 20 }
    Write-Output "Completed and validated attack batch: $attack ($expectedSpecs/$expectedSpecs cells)"
}
Write-Output 'Formal comparison complete: 182/182 cells. Run the analyzer command from docs/RTC_FORMAL_ALL_ATTACKS_TWO_SEED.md.'
