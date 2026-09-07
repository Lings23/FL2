# RTC-v3 V1 multi-seed validation

This stage compares the frozen accepted candidate, B3R-F0.51, with Multi-Krum
under one explicit clean condition and three attack conditions. It does not tune
another parameter.

## Fixed matrix

- Seeds: `42,46,47`
- Malicious fraction / assumed Byzantine budget: `0.3` / `f=3`
- Defenses: `rtc_cumulative_q_cap_accepted_anchor` and `multi_krum`
- Conditions: clean (`attack=none`), LIE `z=0.25`, LIE `z=0.5`, strong DBA
- RTC: floor `0.5`, linear cumulative-q cap, accepted anchor fraction `0.51`
- Multi-Krum: select `5` of `10` participating clients
- CIFAR-10 IID, 60 rounds, attack rounds 11-60, 20 clients, participation `0.5`

The manual script contains 22 new cells. It deliberately reuses the current-
implementation seed42 RTC LIE z=0.5 and DBA cells, so those completed cells are
not retrained. Older Multi-Krum results are not reused because their attack
implementation hash is stale.

This user-requested three-seed design is a paired multi-seed check, but its
paired t tests have only two degrees of freedom and materially wider confidence
intervals than the superseded five-seed design. The final report must state that
limitation and must not claim strong statistical significance from non-significance.

## Manual training boundary

```powershell
& 'D:\workspace\FL2\experiments\run_rtc_v1_multiseed_validation.ps1' -Execute
```

Codex must not execute this command. After a human finishes it and resumes the
goal, verify all statuses, 61-row round files, quality gates, manifest hashes,
trial-plan hashes and attack-implementation hashes, then run:

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' 'D:\workspace\FL2\analysis\rtc_v1_multiseed\analyze_results.py'
```

The analyzer writes `runs.csv`, `paired_by_seed.csv`,
`summary_by_condition.csv`, `statistical_tests.csv`,
`first_mechanism_divergence.csv`, and `decision.json`. In addition to ACC and
DBA ASR, it reports malicious/benign aggregation weight, malicious/benign
impact, zero/effective update mass, accepted-anchor mass, benign-identity and
behavioral-nonattacker clipping/watch/restricted/quarantine rates, attacker
detection rates, and the first RTC/Multi-Krum client-level mechanism divergence
both overall and inside the metric window. A final RTC versus Multi-Krum report
is required after these outputs pass review. Both `statistical_tests.csv` and
`decision.json` carry the three-seed inference scope, two degrees of freedom,
and an explicit guard that a non-significant test does not establish equivalence.
