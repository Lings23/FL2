"""Validate the B3R accepted-weight recycle candidate against frozen B2 baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.rtc_anchor_recycle_b3.analyze_results import (  # noqa: E402
    _metrics,
    _numeric_mean,
    _one_file,
    _spec,
    _specs,
    _validate_completion,
)


DEFAULT_BASELINE_EXPERIMENT = ROOT / "logs" / "rtc_v3_anchor_recycle_b3_seed42_mf03"
DEFAULT_CANDIDATE_EXPERIMENT = ROOT / "logs" / "rtc_v3_anchor_recycle_b3r_seed42_mf03"
DEFAULT_OUTPUT = ROOT / "analysis" / "rtc_anchor_recycle_b3r"
ATTACKS = ("dba", "scaling_backdoor")
BASELINE = "rtc_cumulative_q_cap"
FAILED_B3 = "rtc_cumulative_q_cap_anchor"
CANDIDATE = "rtc_cumulative_q_cap_accepted_anchor"


def _all_gates_pass(directory: Path) -> bool:
    path = directory / "quality_gates.csv"
    if not path.is_file():
        raise RuntimeError(f"quality_gates.csv is missing in {directory}")
    gates = pd.read_csv(path)
    passed = gates["passed"].astype(str).str.lower().eq("true")
    if gates.empty or not passed.all():
        raise RuntimeError(
            f"quality gates failed in {directory}: "
            f"{gates.loc[~passed, 'gate'].tolist()}"
        )
    return True


def _round_metrics(
    path: Path,
    attack: str,
    defense: str,
    expected_recycle_fraction: float | None = None,
) -> dict[str, object]:
    result = _metrics(path, attack, defense)
    frame = pd.read_csv(path)
    active = frame[pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1]
    if defense == CANDIDATE:
        if "fit_rtc_v3_anchor_recycle_weighting" not in active:
            raise RuntimeError(f"{attack}: recycle weighting runtime metric is missing")
        observed = set(
            active["fit_rtc_v3_anchor_recycle_weighting"].dropna().astype(str)
        )
        if observed != {"accepted"}:
            raise RuntimeError(
                f"{attack}: expected accepted recycle weighting, got {sorted(observed)}"
            )
        if expected_recycle_fraction is not None:
            column = "fit_rtc_v3_anchor_recycle_fraction"
            if column not in active:
                raise RuntimeError(f"{attack}: recycle fraction runtime metric is missing")
            fractions = pd.to_numeric(active[column], errors="coerce").dropna()
            if fractions.empty or not (
                (fractions - expected_recycle_fraction).abs() <= 1e-12
            ).all():
                raise RuntimeError(
                    f"{attack}: expected recycle fraction "
                    f"{expected_recycle_fraction}, got {sorted(fractions.unique())}"
                )
        result["anchor_recycle_source_available_rate"] = _numeric_mean(
            active, "fit_rtc_v3_anchor_recycle_source_available"
        )
    else:
        result["anchor_recycle_source_available_rate"] = 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT
    )
    parser.add_argument(
        "--candidate-experiment-dir", type=Path, default=DEFAULT_CANDIDATE_EXPERIMENT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-experiment-dir",
        type=Path,
        default=None,
        help="Optional accepted-anchor reference used to prove a fraction-only retry.",
    )
    parser.add_argument(
        "--expected-anchor-recycle-fraction", type=float, default=1.0
    )
    parser.add_argument("--stage-name", default="b3r")
    args = parser.parse_args()
    baseline_dir = args.baseline_experiment_dir.resolve()
    candidate_dir = args.candidate_experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = (
        args.reference_experiment_dir.resolve()
        if args.reference_experiment_dir is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _all_gates_pass(baseline_dir)
    _all_gates_pass(candidate_dir)
    baseline_specs = _specs(baseline_dir)
    candidate_specs = _specs(candidate_dir)
    reference_specs = None
    if reference_dir is not None:
        _all_gates_pass(reference_dir)
        reference_specs = _specs(reference_dir)
        if len(reference_specs) != 2:
            raise RuntimeError(
                f"expected exactly two accepted-anchor reference specs, "
                f"got {len(reference_specs)}"
            )
    if len(baseline_specs) != 4:
        raise RuntimeError(f"expected four frozen B3 specs, got {len(baseline_specs)}")
    if len(candidate_specs) != 2:
        raise RuntimeError(
            f"expected exactly two candidate-only B3R specs, got {len(candidate_specs)}"
        )

    metric_rows: list[dict[str, object]] = []
    trial_hashes: dict[str, str] = {}
    for attack in ATTACKS:
        baseline_spec = _spec(baseline_specs, attack, BASELINE)
        failed_spec = _spec(baseline_specs, attack, FAILED_B3)
        candidate_spec = _spec(candidate_specs, attack, CANDIDATE)
        reference_spec = (
            _spec(reference_specs, attack, CANDIDATE)
            if reference_specs is not None
            else None
        )
        hashes = {
            str(baseline_spec["trial_plan_hash"]),
            str(failed_spec["trial_plan_hash"]),
            str(candidate_spec["trial_plan_hash"]),
        }
        if reference_spec is not None:
            hashes.add(str(reference_spec["trial_plan_hash"]))
        if len(hashes) != 1:
            raise RuntimeError(f"{attack}: B2/B3/B3R trial plans differ")
        contracts = {
            str(baseline_spec["attack_contract_hash"]),
            str(failed_spec["attack_contract_hash"]),
            str(candidate_spec["attack_contract_hash"]),
        }
        if reference_spec is not None:
            contracts.add(str(reference_spec["attack_contract_hash"]))
        if len(contracts) != 1:
            raise RuntimeError(f"{attack}: B2/B3/B3R attack contracts differ")

        baseline_custom = dict(baseline_spec["custom_params"])
        failed_custom = dict(failed_spec["custom_params"])
        candidate_custom = dict(candidate_spec["custom_params"])
        if failed_custom.get("anchor_recycle_fraction") != 1.0:
            raise RuntimeError(f"{attack}: failed B3 did not use full recycle")
        if candidate_custom.get("anchor_recycle_weighting") != "accepted":
            raise RuntimeError(f"{attack}: B3R weighting is not accepted")
        observed_fraction = candidate_custom.get("anchor_recycle_fraction")
        if observed_fraction != args.expected_anchor_recycle_fraction:
            raise RuntimeError(
                f"{attack}: expected anchor recycle fraction "
                f"{args.expected_anchor_recycle_fraction}, got {observed_fraction}"
            )
        if reference_spec is None:
            candidate_without_weighting = dict(candidate_custom)
            candidate_without_weighting.pop("anchor_recycle_weighting")
            if candidate_without_weighting != failed_custom:
                raise RuntimeError(
                    f"{attack}: B3R changes more than recycle anchor weighting"
                )
        else:
            reference_custom = dict(reference_spec["custom_params"])
            if reference_custom.get("anchor_recycle_weighting") != "accepted":
                raise RuntimeError(f"{attack}: reference weighting is not accepted")
            if reference_custom.get("anchor_recycle_fraction") != 1.0:
                raise RuntimeError(f"{attack}: reference did not use full recycle")
            candidate_without_fraction = dict(candidate_custom)
            candidate_without_fraction.pop("anchor_recycle_fraction")
            reference_without_fraction = dict(reference_custom)
            reference_without_fraction.pop("anchor_recycle_fraction")
            if candidate_without_fraction != reference_without_fraction:
                raise RuntimeError(
                    f"{attack}: retry changes more than recycle fraction"
                )
        baseline_without_anchor = dict(failed_custom)
        baseline_without_anchor.pop("anchor_recycle_fraction")
        if baseline_without_anchor != baseline_custom:
            raise RuntimeError(f"{attack}: frozen B3 baseline contract drifted")
        if baseline_custom.get("semantic_intervention_risk_floor") != 0.5:
            raise RuntimeError(f"{attack}: floor=0.5 is not frozen")
        if baseline_custom.get("cumulative_q_cap_power") != 1:
            raise RuntimeError(f"{attack}: linear q cap is not frozen")

        trial_hashes[attack] = hashes.pop()
        _validate_completion(baseline_dir, attack, BASELINE)
        _validate_completion(candidate_dir, attack, CANDIDATE)
        metric_rows.append(
            _round_metrics(
                _one_file(baseline_dir, "rounds", attack, BASELINE, ".csv"),
                attack,
                BASELINE,
            )
        )
        metric_rows.append(
            _round_metrics(
                _one_file(candidate_dir, "rounds", attack, CANDIDATE, ".csv"),
                attack,
                CANDIDATE,
                args.expected_anchor_recycle_fraction,
            )
        )

    comparison = pd.DataFrame(metric_rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    indexed = comparison.set_index(["attack", "mode"])

    deltas: list[dict[str, object]] = []
    checks: dict[str, bool] = {}
    for attack in ATTACKS:
        baseline = indexed.loc[(attack, BASELINE)]
        candidate = indexed.loc[(attack, CANDIDATE)]
        row: dict[str, object] = {"attack": attack}
        for metric in (
            "active_accuracy",
            "final_accuracy",
            "active_asr",
            "malicious_weight_share",
            "benign_weight_share",
            "malicious_impact_share",
            "zero_update_mass",
            "effective_update_mass",
            "anchor_recycle_mass",
            "benign_watch_rate",
            "benign_restricted_rate",
            "benign_quarantined_rate",
        ):
            row[f"{metric}_candidate_minus_baseline"] = float(
                candidate[metric] - baseline[metric]
            )
        row["zero_mass_ratio_candidate_over_baseline"] = float(
            candidate["zero_update_mass"] / baseline["zero_update_mass"]
            if baseline["zero_update_mass"] > 1e-12
            else 0.0
        )
        deltas.append(row)
        prefix = attack.replace("_backdoor", "")
        checks[f"{prefix}_zero_mass_halved_or_at_most_2pct"] = bool(
            candidate["zero_update_mass"]
            <= max(0.02, 0.5 * baseline["zero_update_mass"]) + 1e-12
        )
        checks[f"{prefix}_active_accuracy_not_lower"] = bool(
            candidate["active_accuracy"] >= baseline["active_accuracy"] - 1e-12
        )
        checks[f"{prefix}_active_asr_increase_at_most_0_5pp"] = bool(
            candidate["active_asr"] - baseline["active_asr"] <= 0.005 + 1e-12
        )
        checks[f"{prefix}_malicious_impact_increase_at_most_0_5pp"] = bool(
            candidate["malicious_impact_share"]
            - baseline["malicious_impact_share"]
            <= 0.005 + 1e-12
        )
        checks[f"{prefix}_malicious_weight_increase_at_most_0_5pp"] = bool(
            candidate["malicious_weight_share"]
            - baseline["malicious_weight_share"]
            <= 0.005 + 1e-12
        )
        checks[f"{prefix}_anchor_recycle_is_active"] = bool(
            candidate["anchor_recycle_mass"] > 1e-12
        )
    pd.DataFrame(deltas).to_csv(output_dir / "deltas.csv", index=False)

    group_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    asr_event_rows: list[dict[str, object]] = []
    for attack in ATTACKS:
        round_frames = {
            BASELINE: pd.read_csv(
                _one_file(baseline_dir, "rounds", attack, BASELINE, ".csv")
            ),
            CANDIDATE: pd.read_csv(
                _one_file(candidate_dir, "rounds", attack, CANDIDATE, ".csv")
            ),
        }
        client_frames: dict[str, pd.DataFrame] = {}
        for defense, directory in (
            (BASELINE, baseline_dir),
            (CANDIDATE, candidate_dir),
        ):
            clients = pd.read_csv(
                _one_file(directory, "raw", attack, defense, "_clients.csv")
            )
            clients["is_malicious"] = (
                clients["is_malicious"].astype(str).str.lower().eq("true")
            )
            client_frames[defense] = clients
            active = clients[pd.to_numeric(clients["round"], errors="coerce") >= 11]
            grouped = (
                active.groupby(["round", "is_malicious"], as_index=False)
                .agg(
                    selected_clients=("cid", "size"),
                    aggregation_weight=("aggregation_weight", "sum"),
                    impact_norm=("impact_norm", "sum"),
                    semantic_risk=("semantic_risk", "mean"),
                )
                .groupby("is_malicious", as_index=False)
                .mean(numeric_only=True)
                .drop(columns="round")
            )
            for _, group in grouped.iterrows():
                group_rows.append(
                    {"attack": attack, "mode": defense, **group.to_dict()}
                )

        candidate_active = round_frames[CANDIDATE][
            pd.to_numeric(
                round_frames[CANDIDATE]["fit_rtc_v3_anchor_recycle_mass"],
                errors="coerce",
            ).fillna(0.0)
            > 1e-12
        ]
        if not candidate_active.empty:
            first = candidate_active.sort_values("round").iloc[0]
            round_number = int(first["round"])
            base = round_frames[BASELINE].loc[
                pd.to_numeric(round_frames[BASELINE]["round"], errors="coerce")
                == round_number
            ].iloc[0]
            activation_rows.append(
                {
                    "attack": attack,
                    "round": round_number,
                    "baseline_zero_mass": float(base["fit_rtc_v3_zero_update_mass"]),
                    "candidate_zero_mass": float(first["fit_rtc_v3_zero_update_mass"]),
                    "anchor_target_mass": float(
                        first["fit_rtc_v3_anchor_recycle_target_mass"]
                    ),
                    "anchor_recycle_mass": float(
                        first["fit_rtc_v3_anchor_recycle_mass"]
                    ),
                    "source_available": float(
                        first["fit_rtc_v3_anchor_recycle_source_available"]
                    ),
                    "baseline_accuracy": float(base["server_accuracy"]),
                    "candidate_accuracy": float(first["server_accuracy"]),
                    "baseline_asr": float(base["server_asr"]),
                    "candidate_asr": float(first["server_asr"]),
                }
            )

        paired_clients = client_frames[BASELINE].merge(
            client_frames[CANDIDATE],
            on=["round", "cid"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired_clients["weight_delta"] = (
            paired_clients["aggregation_weight_candidate"]
            - paired_clients["aggregation_weight_baseline"]
        )
        changed = paired_clients[paired_clients["weight_delta"].abs() > 1e-10]
        if not changed.empty:
            first_round = int(changed["round"].min())
            for _, row in changed[changed["round"] == first_round].sort_values("cid").iterrows():
                divergence_rows.append(
                    {
                        "attack": attack,
                        "round": first_round,
                        "cid": int(row["cid"]),
                        "is_malicious": bool(row["is_malicious_baseline"]),
                        "baseline_weight": float(row["aggregation_weight_baseline"]),
                        "candidate_weight": float(row["aggregation_weight_candidate"]),
                        "candidate_minus_baseline": float(row["weight_delta"]),
                        "baseline_semantic_risk": float(row["semantic_risk_baseline"]),
                        "candidate_semantic_risk": float(row["semantic_risk_candidate"]),
                    }
                )

        paired_rounds = round_frames[BASELINE].merge(
            round_frames[CANDIDATE],
            on="round",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired_rounds = paired_rounds[
            pd.to_numeric(
                paired_rounds["planned_attack_active_candidate"], errors="coerce"
            )
            == 1
        ].copy()
        paired_rounds["asr_delta"] = (
            pd.to_numeric(paired_rounds["server_asr_candidate"], errors="coerce")
            - pd.to_numeric(paired_rounds["server_asr_baseline"], errors="coerce")
        )
        material = paired_rounds[paired_rounds["asr_delta"] > 0.005 + 1e-12]
        if not material.empty:
            first = material.sort_values("round").iloc[0]
            asr_event_rows.append(
                {
                    "attack": attack,
                    "event": "first_candidate_asr_increase_above_0_5pp",
                    "round": int(first["round"]),
                    "baseline_asr": float(first["server_asr_baseline"]),
                    "candidate_asr": float(first["server_asr_candidate"]),
                    "candidate_minus_baseline": float(first["asr_delta"]),
                    "anchor_recycle_mass": float(
                        first["fit_rtc_v3_anchor_recycle_mass_candidate"]
                    ),
                }
            )

    pd.DataFrame(group_rows).to_csv(
        output_dir / "client_group_summary.csv", index=False
    )
    pd.DataFrame(activation_rows).to_csv(
        output_dir / "first_anchor_activation.csv", index=False
    )
    pd.DataFrame(divergence_rows).to_csv(
        output_dir / "first_weight_divergence.csv", index=False
    )
    pd.DataFrame(asr_event_rows).to_csv(
        output_dir / "first_asr_gate_failure.csv", index=False
    )

    checks.update(
        {
            "candidate_completion_status_all_passed": True,
            "baseline_and_candidate_quality_gates_all_passed": True,
            "strict_pairing_and_single_mechanism_all_passed": True,
            "runtime_weighting_is_accepted": True,
            "runtime_recycle_fraction_matches_expected": True,
        }
    )
    decision = {
        "stage": args.stage_name,
        "accepted_for_next_stage": all(checks.values()),
        "checks": checks,
        "trial_plan_hashes": trial_hashes,
        "tolerances": {
            "active_accuracy_candidate_minus_baseline_min": 0.0,
            "active_asr_candidate_minus_baseline_max": 0.005,
            "malicious_impact_candidate_minus_baseline_max": 0.005,
            "malicious_weight_candidate_minus_baseline_max": 0.005,
            "zero_mass_candidate_max": "max(0.02, 0.5 * baseline)",
            "expected_anchor_recycle_fraction": args.expected_anchor_recycle_fraction,
        },
        "deltas": deltas,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
