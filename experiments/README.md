# Experiment code map

The supported experiment entry point is:

```powershell
python.exe -m experiments.core.run --profile rtc-fedavg ...
```

`rtc_full` and `rtc_v3` both mean the engineering-promoted semantic-time-
exposure RTC-V3 and default to
`config/rtc_v3_manifest_formal_iid_semantic.json`. Formal comparisons use
strict `TrialPlanV1` pairing and principal-uniform sampling.

## Ownership

- `experiments/core/`: shared runner, schedules, and strict trial-plan API.
- `experiments/rtc_v3/`: stable calibration, evaluation, analysis, audit,
  promotion, and reporting entry points.
- flat `rtc_v3_*` modules: compatibility implementations retained for old
  commands and evidence reproduction.
- `experiments/rtc_v3/legacy/`: marker for historical development utilities;
  these are not imported by the default profile.

## Version policy

- `rtc_full`, `rtc_v3`: promoted RTC-V3.
- `rtc_v2_legacy`, `time_consistency`: RTC-V2 compatibility only.
- `rtc_v3_candidate`: compatibility alias for pre-promotion commands; it runs
  the same V3 class but does not select a candidate manifest automatically.
- periodic `--mode ablation`: explicit historical RTC-V2 reproduction only;
  it is excluded from the default `all` matrix and smoke tests.

No default matrix uses RTC-V2. Historical files are retained because deleting
them would make prior logs and experiment commands harder to reproduce.

## Generalized Byzantine benchmark

`rtc-byzantine` is the promoted RTC-V3 attack-matrix profile. Its implementation
lives in `experiments/rtc_v3/byzantine.py`; it binds versioned attack contracts,
strict TrialPlan pairing and the FedAvg/RTC/Krum/TrimmedMean/Median/FoolsGold/
FreqFed defense matrix. Use `--dry-run` before smoke or formal training. The
frozen protocol and parameter table are documented in
`docs/RTC_V3_BYZANTINE_EXPERIMENT_PROTOCOL.md` and
`docs/RTC_V3_BYZANTINE_ATTACK_MATRIX.md`.
