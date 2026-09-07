"""Validate and analyze the paired RTC-v3 B3 anchor-recycle screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "logs" / "rtc_v3_anchor_recycle_b3_seed42_mf03"
DEFAULT_OUTPUT = ROOT / "analysis" / "rtc_anchor_recycle_b3"
ATTACKS = ("dba", "scaling_backdoor")
BASELINE = "rtc_cumulative_q_cap"
CANDIDATE = "rtc_cumulative_q_cap_anchor"


def _specs(directory: Path) -> list[dict[str, object]]:
    return json.loads(
        (directory / "experiment_manifest.json").read_text(encoding="utf-8")
    )["specs"]


def _spec(
    specs: list[dict[str, object]], attack: str, defense: str
) -> dict[str, object]:
    matches = [
        row
        for row in specs
        if row["attack"] == attack and row["defense"] == defense
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {attack}/{defense} manifest spec, got {len(matches)}"
        )
    return matches[0]


def _one_file(directory: Path, subdir: str, attack: str, defense: str, suffix: str) -> Path:
    matches = [
        path
        for path in (directory / subdir).glob(f"*{suffix}")
        if path.name.startswith(f"{attack}__") and f"__{defense}__" in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {attack}/{defense} {suffix} in {subdir}, got {len(matches)}"
        )
    return matches[0]


def _validate_completion(directory: Path, attack: str, defense: str) -> None:
    status_path = _one_file(directory, "status", attack, defense, ".json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    observed = {
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "last_round": status.get("last_round"),
    }
    successful_states = {"completed", "completed_cached"}
    complete = (
        observed["state"] in successful_states
        and observed["exit_code"] == 0
        and observed["last_round"] == 60
    )
    if not complete:
        expected = {
            "state": sorted(successful_states),
            "exit_code": 0,
            "last_round": 60,
        }
        raise RuntimeError(
            f"{attack}/{defense} is incomplete: expected {expected}, got {observed}"
        )


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _numeric_max(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else 0.0


def _metrics(path: Path, attack: str, defense: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    active = frame[pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1]
    if len(frame) != 61 or len(active) != 50:
        raise RuntimeError(
            f"{attack}/{defense}: expected 61 total/50 active rows, "
            f"got {len(frame)}/{len(active)}"
        )
    return {
        "attack": attack,
        "mode": defense,
        "active_accuracy": float(active["server_accuracy"].mean()),
        "final_accuracy": float(frame.iloc[-1]["server_accuracy"]),
        "active_asr": _numeric_mean(active, "server_asr"),
        "peak_asr": _numeric_max(active, "server_asr"),
        "malicious_weight_share": _numeric_mean(
            active, "fit_malicious_aggregation_weight_share"
        ),
        "benign_weight_share": float(
            (
                pd.to_numeric(active["fit_rtc_v3_weight_sum"], errors="coerce")
                - pd.to_numeric(
                    active["fit_malicious_aggregation_weight_share"], errors="coerce"
                )
            ).mean()
        ),
        "malicious_impact_share": _numeric_mean(
            active, "fit_malicious_impact_share"
        ),
        "benign_impact_share": float(
            (1.0 - pd.to_numeric(active["fit_malicious_impact_share"], errors="coerce")).mean()
        ),
        "zero_update_mass": _numeric_mean(active, "fit_rtc_v3_zero_update_mass"),
        "effective_update_mass": _numeric_mean(
            active, "fit_rtc_v3_effective_update_mass"
        ),
        "anchor_recycle_target_mass": _numeric_mean(
            active, "fit_rtc_v3_anchor_recycle_target_mass"
        ),
        "anchor_recycle_mass": _numeric_mean(
            active, "fit_rtc_v3_anchor_recycle_mass"
        ),
        "anchor_budget_limited_rate": _numeric_mean(
            active, "fit_rtc_v3_anchor_recycle_budget_limited"
        ),
        "benign_watch_rate": _numeric_mean(active, "fit_benign_watch_rate"),
        "benign_restricted_rate": _numeric_mean(
            active, "fit_benign_restricted_rate"
        ),
        "benign_quarantined_rate": _numeric_mean(
            active, "fit_benign_quarantined_rate"
        ),
        "malicious_quarantined_rate": _numeric_mean(
            active, "fit_malicious_quarantined_rate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gates_path = experiment_dir / "quality_gates.csv"
    if not gates_path.is_file():
        raise RuntimeError("experiment is incomplete: quality_gates.csv is missing")
    gates = pd.read_csv(gates_path)
    gate_pass = gates["passed"].astype(str).str.lower().eq("true")
    if gates.empty or not gate_pass.all():
        raise RuntimeError(
            f"quality gates failed: {gates.loc[~gate_pass, 'gate'].tolist()}"
        )

    specs = _specs(experiment_dir)
    if len(specs) != 4:
        raise RuntimeError(f"expected exactly four B3 specs, got {len(specs)}")
    rows: list[dict[str, object]] = []
    trial_hashes: dict[str, str] = {}
    for attack in ATTACKS:
        baseline_spec = _spec(specs, attack, BASELINE)
        candidate_spec = _spec(specs, attack, CANDIDATE)
        if baseline_spec["trial_plan_hash"] != candidate_spec["trial_plan_hash"]:
            raise RuntimeError(f"{attack} baseline/candidate trial plans differ")
        if baseline_spec["attack_contract_hash"] != candidate_spec["attack_contract_hash"]:
            raise RuntimeError(f"{attack} baseline/candidate attack contracts differ")
        baseline_custom = dict(baseline_spec["custom_params"])
        candidate_custom = dict(candidate_spec["custom_params"])
        if candidate_custom.pop("anchor_recycle_fraction", None) != 1.0:
            raise RuntimeError(f"{attack} candidate anchor_recycle_fraction is not 1.0")
        if candidate_custom != baseline_custom:
            raise RuntimeError(f"{attack} candidate changes more than anchor recycle")
        if baseline_custom.get("semantic_intervention_risk_floor") != 0.5:
            raise RuntimeError(f"{attack} baseline did not freeze floor=0.5")
        if baseline_custom.get("cumulative_q_cap_power") != 1:
            raise RuntimeError(f"{attack} baseline did not freeze linear q cap")
        trial_hashes[attack] = str(candidate_spec["trial_plan_hash"])
        for defense in (BASELINE, CANDIDATE):
            _validate_completion(experiment_dir, attack, defense)
            rows.append(
                _metrics(
                    _one_file(
                        experiment_dir, "rounds", attack, defense, ".csv"
                    ),
                    attack,
                    defense,
                )
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    indexed = comparison.set_index(["attack", "mode"])

    deltas: list[dict[str, object]] = []
    checks: dict[str, bool] = {}
    for attack in ATTACKS:
        baseline = indexed.loc[(attack, BASELINE)]
        candidate = indexed.loc[(attack, CANDIDATE)]
        row = {"attack": attack}
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
    divergence_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    asr_event_rows: list[dict[str, object]] = []
    mass_audit_rows: list[dict[str, object]] = []
    for attack in ATTACKS:
        round_frames: dict[str, pd.DataFrame] = {}
        client_frames: dict[str, pd.DataFrame] = {}
        for defense in (BASELINE, CANDIDATE):
            round_frames[defense] = pd.read_csv(
                _one_file(experiment_dir, "rounds", attack, defense, ".csv")
            )
            clients = pd.read_csv(
                _one_file(
                    experiment_dir, "raw", attack, defense, "_clients.csv"
                )
            )
            clients["is_malicious"] = (
                clients["is_malicious"].astype(str).str.lower().eq("true")
            )
            client_frames[defense] = clients
            active_rounds = round_frames[defense][
                pd.to_numeric(
                    round_frames[defense]["planned_attack_active"], errors="coerce"
                )
                == 1
            ].copy()
            weight_sum = pd.to_numeric(
                active_rounds["fit_rtc_v3_weight_sum"], errors="coerce"
            )
            anchor_mass = pd.to_numeric(
                active_rounds["fit_rtc_v3_anchor_recycle_mass"], errors="coerce"
            )
            effective_mass = pd.to_numeric(
                active_rounds["fit_rtc_v3_effective_update_mass"], errors="coerce"
            )
            zero_mass = pd.to_numeric(
                active_rounds["fit_rtc_v3_zero_update_mass"], errors="coerce"
            )
            mass_audit_rows.append(
                {
                    "attack": attack,
                    "mode": defense,
                    "active_rounds": len(active_rounds),
                    "max_weight_plus_anchor_minus_effective_abs": float(
                        (weight_sum + anchor_mass - effective_mass).abs().max()
                    ),
                    "max_one_minus_effective_minus_zero_abs": float(
                        (1.0 - effective_mass - zero_mass).abs().max()
                    ),
                }
            )
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

        candidate_rounds = round_frames[CANDIDATE]
        active_anchor = candidate_rounds[
            pd.to_numeric(
                candidate_rounds["fit_rtc_v3_anchor_recycle_mass"], errors="coerce"
            ).fillna(0.0)
            > 1e-12
        ]
        if not active_anchor.empty:
            first = active_anchor.sort_values("round").iloc[0]
            round_number = int(first["round"])
            baseline_first = round_frames[BASELINE].loc[
                pd.to_numeric(round_frames[BASELINE]["round"], errors="coerce")
                == round_number
            ].iloc[0]
            activation_rows.append(
                {
                    "attack": attack,
                    "round": round_number,
                    "baseline_zero_mass": float(
                        baseline_first["fit_rtc_v3_zero_update_mass"]
                    ),
                    "candidate_zero_mass": float(first["fit_rtc_v3_zero_update_mass"]),
                    "anchor_target_mass": float(
                        first["fit_rtc_v3_anchor_recycle_target_mass"]
                    ),
                    "anchor_recycle_mass": float(
                        first["fit_rtc_v3_anchor_recycle_mass"]
                    ),
                    "anchor_budget_limited": float(
                        first["fit_rtc_v3_anchor_recycle_budget_limited"]
                    ),
                    "baseline_accuracy": float(baseline_first["server_accuracy"]),
                    "candidate_accuracy": float(first["server_accuracy"]),
                    "baseline_asr": float(baseline_first["server_asr"]),
                    "candidate_asr": float(first["server_asr"]),
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
        above_ten = paired_rounds[
            pd.to_numeric(
                paired_rounds["server_asr_candidate"], errors="coerce"
            )
            >= 0.10
        ]
        if not above_ten.empty:
            first = above_ten.sort_values("round").iloc[0]
            asr_event_rows.append(
                {
                    "attack": attack,
                    "event": "first_candidate_asr_at_least_10pct",
                    "round": int(first["round"]),
                    "baseline_asr": float(first["server_asr_baseline"]),
                    "candidate_asr": float(first["server_asr_candidate"]),
                    "candidate_minus_baseline": float(first["asr_delta"]),
                    "anchor_recycle_mass": float(
                        first["fit_rtc_v3_anchor_recycle_mass_candidate"]
                    ),
                }
            )
        peak = paired_rounds.loc[
            pd.to_numeric(
                paired_rounds["server_asr_candidate"], errors="coerce"
            ).idxmax()
        ]
        asr_event_rows.append(
            {
                "attack": attack,
                "event": "candidate_peak_asr",
                "round": int(peak["round"]),
                "baseline_asr": float(peak["server_asr_baseline"]),
                "candidate_asr": float(peak["server_asr_candidate"]),
                "candidate_minus_baseline": float(peak["asr_delta"]),
                "anchor_recycle_mass": float(
                    peak["fit_rtc_v3_anchor_recycle_mass_candidate"]
                ),
            }
        )

        paired = client_frames[BASELINE].merge(
            client_frames[CANDIDATE],
            on=["round", "cid"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired["weight_delta"] = (
            paired["aggregation_weight_candidate"]
            - paired["aggregation_weight_baseline"]
        )
        changed = paired[paired["weight_delta"].abs() > 1e-10]
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
        output_dir / "asr_trajectory_events.csv", index=False
    )
    pd.DataFrame(mass_audit_rows).to_csv(
        output_dir / "mass_balance_audit.csv", index=False
    )

    checks.update(
        {
            "completion_status_all_passed": True,
            "quality_gates_all_passed": True,
            "strict_pairing_all_passed": True,
        }
    )
    decision = {
        "stage": "b3",
        "accepted_for_next_stage": all(checks.values()),
        "checks": checks,
        "trial_plan_hashes": trial_hashes,
        "tolerances": {
            "active_accuracy_candidate_minus_baseline_min": 0.0,
            "active_asr_candidate_minus_baseline_max": 0.005,
            "malicious_impact_candidate_minus_baseline_max": 0.005,
            "malicious_weight_candidate_minus_baseline_max": 0.005,
            "zero_mass_candidate_max": "max(0.02, 0.5 * baseline)",
        },
        "deltas": deltas,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
