# RTC-v2 target-mode experiment report

Date: 2026-07-09

## Final design

RTC-v2 keys history by server-side client identity and true server round. Each update is represented by magnitude, direction, influence, and temporal risks. The first three form an instantaneous event risk. Temporal risk is now only a conditional periodic amplifier: it requires at least four high instantaneous-risk events with consistent true-round gaps and a high-risk current event. It no longer duplicates magnitude bursts, repeats state-machine evidence, or remains latched after the current anomaly disappears.

The state machine uses `normal`, `watch`, `restricted`, and `quarantined`. Final defaults are: trust beta 0.75; thresholds 0.25/0.45/0.80; state multipliers 1.0/0.50/0.05/0.0; clip multiplier 1.0; per-client cap multiplier 0.6. Quarantine cooldown is measured in server rounds. Aggregation applies trust/state weights, clipping, influence caps, and a safe unchanged-model fallback.

## Final smoke and ablations

Condition: CIFAR-10, ResNet-18, model replacement, 1/1 schedule, boost 10, IID, full participation, seed 42, six rounds. This is a pipeline/short-response gate; it is too short for four-event temporal detection.

| Defense | Active ASR-AUC | Peak ASR | Residual ASR | Final accuracy |
|---|---:|---:|---:|---:|
| RTC full | 0.068 | 0.072 | 0.058 | 0.148 |
| RTC no temporal | 0.033 | 0.063 | 0.042 | 0.160 |
| RTC no direction | 1.000 | 1.000 | 0.087 | 0.158 |
| RTC no influence | 1.000 | 1.000 | 0.000 | 0.266 |
| RTC trust only (no clipping/caps) | 0.997 | 1.000 | 0.000 | 0.256 |
| Clip only | 1.000 | 1.000 | 0.063 | 0.140 |
| FedAvg | 1.000 | 1.000 | 0.500 | 0.122 |

In the authoritative smoke rerun after synchronizing class defaults, clean final accuracy was 0.162 for FedAvg and 0.166 for RTC. Benign quarantine was zero. All ten final smoke gates passed, including RTC/FedAvg <= 50%, RTC/clip-only <= 80%, finite metrics, real ablations, matching sampling manifests, and unique files. No numeric infinity was present in the summary.

## Three-seed IID, partial participation

Condition: 1/1 schedule, boost 10, rounds 3-9 attacked, 50% participation, seeds 42/43/44. Values use a low-cost screening budget of 20 local samples and 300 test samples.

| Seed | RTC ASR-AUC | FedAvg | Clip only | RTC/FedAvg | RTC/clip | Clean RTC-FedAvg | Benign quarantine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.388 | 2.922 | 3.000 | 13.3% | 12.9% | +2.0 pp | 0% |
| 43 | 0.786 | 3.000 | 3.000 | 26.2% | 26.2% | -1.0 pp | 0% |
| 44 | 0.002 | 2.947 | 2.994 | 0.1% | 0.1% | -0.7 pp | 0% |

Mean RTC ASR-AUC was 0.392 versus 2.956 for FedAvg and 2.998 for clip-only. Every seed improved in the same direction. The clean mean RTC-FedAvg difference was +0.1 percentage points; the worst clean drop was 1.0 point. Intermittent selection/rejoining caused no temporal trigger, demonstrating that the core instantaneous/state weighting remains effective when participation history is sparse.

## Three-seed Dirichlet Non-IID, partial participation

Condition: Dirichlet alpha 0.3, 1/1 schedule, boost 10, 50% participation, seeds 42/43/44.

| Seed | RTC ASR-AUC | FedAvg | Clip only | RTC/FedAvg | RTC/clip | Clean RTC-FedAvg | Benign quarantine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.356 | 2.966 | 2.574 | 12.0% | 13.8% | +5.7 pp | 0% |
| 43 | 1.379 | 3.000 | 3.000 | 46.0% | 46.0% | -1.0 pp | 0% |
| 44 | 0.297 | 3.000 | 2.689 | 9.9% | 11.1% | +0.7 pp | 0% |

Mean RTC ASR-AUC was 0.677 versus 2.989 for FedAvg and 2.754 for clip-only. Every seed improved in the same direction. Benign restricted rate averaged 4.3%, but benign quarantine remained zero. The worst clean utility drop was 1.0 percentage point.

## Second period and attack strength

Condition: Dirichlet alpha 0.3, long 3/3 schedule, boost 5, 50% participation, seeds 42/43/44.

| Seed | RTC ASR-AUC | FedAvg | Clip only | RTC/FedAvg | RTC/clip |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.125 | 3.000 | 2.996 | 4.2% | 4.2% |
| 43 | 2.223 | 4.072 | 4.523 | 54.6% | 49.2% |
| 44 | 0.343 | 4.000 | 3.769 | 8.6% | 9.1% |

The mean RTC/FedAvg ratio was about 24%; all seeds improved, and all beat clip-only. Seed 43 did not satisfy the per-seed 50% FedAvg target. Lowering the restricted threshold from 0.45 to 0.43 only reduced its ASR-AUC to 2.129 while increasing benign restrictions, so that change was rejected. This condition is the main known security boundary.

## 60-round confirmation on the hardest condition

The retained configuration was rerun for 60 rounds on Dirichlet alpha 0.3, 50% participation, long 3/3, boost 5, and seeds 42/43/44. Attacks ran during rounds 11-47, leaving 13 post-attack recovery rounds.

| Seed | RTC ASR-AUC | FedAvg | Clip only | RTC/FedAvg | RTC/clip | RTC residual ASR | Benign quarantine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3.124 | 13.659 | 8.433 | 22.9% | 37.0% | 0.060 | 0.00% |
| 43 | 5.509 | 14.494 | 9.684 | 38.0% | 56.9% | 0.186 | 1.25% |
| 44 | 3.593 | 12.093 | 7.766 | 29.7% | 46.3% | 0.058 | 0.00% |

Mean RTC ASR-AUC was 4.075 versus 13.415 for FedAvg and 8.628 for clip-only. Every seed improved, all security ratio gates passed, mean benign quarantine was 0.42%, and safe fallback was never used. The earlier 12-round seed-43 boundary did not persist at 60 rounds.

Paired 60-round clean final accuracy was FedAvg/RTC 0.196/0.194, 0.262/0.278, and 0.232/0.218. The worst RTC clean drop was 1.4 percentage points. Mean whole-trajectory RTC-minus-FedAvg accuracy was +0.86, -0.33, and -0.79 percentage points. The clean accuracy gate passed for every seed, and fallback remained zero.

## Temporal redesign evidence

The original mixed burst/repeated/periodic temporal score was rejected: in a 16-round full-participation experiment, RTC full ASR-AUC was 0.817 versus 0.248 without temporal, while clean and attacked temporal means were nearly identical. A periodic-only version with an independent instantaneous event history improved RTC to 0.290 versus 0.388 without temporal, but revealed latching. Requiring a high current event removed latching; RTC then achieved 0.281 versus 0.542 without temporal, with clean accuracy 0.322 versus FedAvg 0.318. This evidence supports keeping temporal as a guarded amplifier, not a primary detector.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_framework.py -q
.\.venv\Scripts\python.exe experiments\periodic_attack.py --smoke --rerun --output logs\periodic_attack_final_smoke_defaults
.\.venv\Scripts\python.exe experiments\periodic_attack.py --mode main --attacks model_replacement --defenses fedavg,clip_only,rtc_full --periods short_1_1 --seeds 42,43,44 --malicious-fractions 0.2 --rounds 12 --num-clients 10 --participation-rate 0.5 --partition dirichlet --dirichlet-alpha 0.3 --boost-factor 10 --attack-start-round 3 --attack-end-round 9 --max-client-samples 20 --max-test-samples 300 --skip-clean --rerun --output logs\rtc_multiseed_dirichlet03_partial
```

Use `--clean-only` with the same condition to produce paired clean FedAvg/RTC trajectories. The screening CLI also supports explicit attack, defense, period, seed, malicious-fraction, rounds, client count, participation, partition, Dirichlet alpha, boost, attack end, and sample-budget controls. Every run ID and experiment manifest encodes the condition to prevent collisions.

## Limits

Most screening matrices use reduced local/test sample budgets and 6-16 rounds; the hardest retained condition was additionally confirmed for 60 rounds, still with a 20-sample local cap. These runs establish comparative behavior, not final CIFAR-10 convergence. CUDA/Ray runs showed residual nondeterminism even with the same seed, so conclusions rely on paired multi-seed direction and effect size rather than exact bitwise replay. A publication claim should still run the preregistered matrix without a local-sample cap. Full-dataset local training is the remaining compute-intensive validation, not evidence used to tune the reported configuration.
