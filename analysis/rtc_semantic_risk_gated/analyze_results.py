"""Evaluate the two-cell semantic risk-gate experiment against paired baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_DIR = ROOT / "logs" / "rtc_v3_semantic_risk_gated_seed42_mf03"
BASELINE_DIRS = {
    "lie": ROOT / "logs" / "rtc_v3_semantic_observe_lie_z05_seed42_mf03",
    "dba": ROOT / "logs" / "rtc_v3_semantic_observe_safety_seed42_mf03",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def _load_specs(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["specs"]


def _spec(directory: Path, attack: str, defense: str) -> dict[str, object]:
    matches = [
        row
        for row in _load_specs(directory / "experiment_manifest.json")
        if row["attack"] == attack and row["defense"] == defense
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {attack}/{defense} spec in {directory}, got {len(matches)}"
        )
    return matches[0]


def _round_file(directory: Path, attack: str, defense: str) -> Path:
    matches = [
        path
        for path in (directory / "rounds").glob("*.csv")
        if path.name.startswith(f"{attack}__") and f"__{defense}__" in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one completed {attack}/{defense} round file, got {len(matches)}"
        )
    return matches[0]


def _client_file(directory: Path, attack: str, defense: str) -> Path:
    matches = [
        path
        for path in (directory / "raw").glob("*_clients.csv")
        if path.name.startswith(f"{attack}__") and f"__{defense}__" in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one completed {attack}/{defense} client file, got {len(matches)}"
        )
    return matches[0]


def _status_file(directory: Path, attack: str, defense: str) -> Path:
    matches = [
        path
        for path in (directory / "status").glob("*.json")
        if path.name.startswith(f"{attack}__") and f"__{defense}__" in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {attack}/{defense} status file, got {len(matches)}"
        )
    return matches[0]


def _validate_completion(directory: Path, attack: str, defense: str) -> None:
    status_path = _status_file(directory, attack, defense)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    observed = {
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "last_round": status.get("last_round"),
    }
    expected = {"state": "completed", "exit_code": 0, "last_round": 60}
    if observed != expected:
        raise RuntimeError(
            f"{attack}/{defense} is not complete: expected {expected}, got {observed}"
        )


def _metrics(path: Path, attack: str, label: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    active = frame[pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1]
    if len(frame) != 61 or len(active) != 50:
        raise RuntimeError(
            f"{label} is incomplete: expected 61 total/50 active rows, "
            f"got {len(frame)}/{len(active)}"
        )
    result: dict[str, object] = {
        "attack": attack,
        "mode": label,
        "active_accuracy": float(active["server_accuracy"].mean()),
        "final_accuracy": float(frame.iloc[-1]["server_accuracy"]),
        "malicious_weight_share": float(
            active["fit_malicious_aggregation_weight_share"].mean()
        ),
        "benign_weight_share": float(
            (
                active["fit_rtc_v3_weight_sum"]
                - active["fit_malicious_aggregation_weight_share"]
            ).mean()
        ),
        "malicious_impact_share": float(
            active["fit_malicious_impact_share"].mean()
        ),
        "benign_impact_share": float(
            (1.0 - active["fit_malicious_impact_share"]).mean()
        ),
        "benign_watch_rate": float(active["fit_benign_watch_rate"].mean()),
        "benign_restricted_rate": float(
            active["fit_benign_restricted_rate"].mean()
        ),
        "benign_quarantined_rate": float(
            active["fit_benign_quarantined_rate"].mean()
        ),
        "malicious_watch_rate": float(active["fit_malicious_watch_rate"].mean()),
        "malicious_restricted_rate": float(
            active["fit_malicious_restricted_rate"].mean()
        ),
        "malicious_quarantined_rate": float(
            active["fit_malicious_quarantined_rate"].mean()
        ),
        "weight_sum": float(active["fit_rtc_v3_weight_sum"].mean()),
        "zero_update_mass": float(active["fit_rtc_v3_zero_update_mass"].mean()),
    }
    if attack == "dba":
        result["active_asr"] = float(
            active["server_dba_full_trigger_asr"].mean()
        )
        result["peak_asr"] = float(active["server_dba_full_trigger_asr"].max())
    if "fit_rtc_v3_semantic_intervention_active_count" in active:
        result["semantic_intervention_active_count_mean"] = float(
            active["fit_rtc_v3_semantic_intervention_active_count"].mean()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument(
        "--candidate-defense", default="rtc_semantic_risk_gated"
    )
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--baseline-defense", default="rtc_full")
    parser.add_argument("--stage", choices=("b1r", "b2"), default="b1r")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    candidate_dir = args.candidate_dir.resolve()
    candidate_defense = str(args.candidate_defense)
    baseline_defense = str(args.baseline_defense)
    baseline_dirs = (
        {attack: args.baseline_dir.resolve() for attack in ("lie", "dba")}
        if args.baseline_dir is not None
        else BASELINE_DIRS
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_path = candidate_dir / "quality_gates.csv"
    if not quality_path.is_file():
        raise RuntimeError(
            "candidate experiment is not complete: quality_gates.csv is missing"
        )
    gates = pd.read_csv(quality_path)
    if gates.empty or not gates["passed"].astype(str).str.lower().eq("true").all():
        failed = gates.loc[
            ~gates["passed"].astype(str).str.lower().eq("true"), "gate"
        ].tolist()
        raise RuntimeError(f"candidate quality gates failed: {failed}")

    rows: list[dict[str, object]] = []
    for attack in ("lie", "dba"):
        _validate_completion(candidate_dir, attack, candidate_defense)
        candidate_spec = _spec(
            candidate_dir, attack, candidate_defense
        )
        baseline_spec = _spec(
            baseline_dirs[attack], attack, baseline_defense
        )
        if candidate_spec["trial_plan_hash"] != baseline_spec["trial_plan_hash"]:
            raise RuntimeError(f"{attack} candidate/baseline trial plans are not paired")
        rows.append(
            _metrics(
                _round_file(baseline_dirs[attack], attack, baseline_defense),
                attack,
                baseline_defense,
            )
        )
        rows.append(
            _metrics(
                _round_file(
                    candidate_dir, attack, candidate_defense
                ),
                attack,
                candidate_defense,
            )
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    client_group_rows: list[dict[str, object]] = []
    first_divergence_rows: list[dict[str, object]] = []
    for attack in ("lie", "dba"):
        baseline_clients = pd.read_csv(
            _client_file(baseline_dirs[attack], attack, baseline_defense)
        )
        candidate_clients = pd.read_csv(
            _client_file(
                candidate_dir, attack, candidate_defense
            )
        )
        for frame in (baseline_clients, candidate_clients):
            if "rtc_v3_client_q_cap" not in frame:
                frame["rtc_v3_client_q_cap"] = float("nan")
        for mode, frame in (
            (baseline_defense, baseline_clients),
            (candidate_defense, candidate_clients),
        ):
            active = frame[pd.to_numeric(frame["round"], errors="coerce") >= 11].copy()
            active["is_malicious"] = (
                active["is_malicious"].astype(str).str.lower().eq("true")
            )
            per_round = (
                active.groupby(["round", "is_malicious"], as_index=False)
                .agg(
                    selected_clients=("cid", "size"),
                    aggregation_weight=("aggregation_weight", "sum"),
                    impact_norm=("impact_norm", "sum"),
                    semantic_risk=("semantic_risk", "mean"),
                    semantic_q=("semantic_q", "mean"),
                    cumulative_q=("rtc_v3_cumulative_q_full", "mean"),
                    direction_q=("rtc_v3_direction_q_full", "mean"),
                    client_q_cap=("rtc_v3_client_q_cap", "mean"),
                )
            )
            summary = (
                per_round.groupby("is_malicious", as_index=False)
                .mean(numeric_only=True)
                .drop(columns="round")
            )
            for _, row in summary.iterrows():
                client_group_rows.append(
                    {
                        "attack": attack,
                        "mode": mode,
                        **row.to_dict(),
                    }
                )

        paired = baseline_clients.merge(
            candidate_clients,
            on=["round", "cid"],
            suffixes=("_full", "_candidate"),
            validate="one_to_one",
        )
        paired["weight_delta_candidate_minus_full"] = (
            paired["aggregation_weight_candidate"]
            - paired["aggregation_weight_full"]
        )
        changed = paired[
            paired["weight_delta_candidate_minus_full"].abs() > 1e-10
        ]
        if changed.empty:
            continue
        first_round = int(changed["round"].min())
        first = changed[changed["round"] == first_round]
        for _, row in first.sort_values("cid").iterrows():
            divergence = {
                "attack": attack,
                "round": first_round,
                "cid": int(row["cid"]),
                "is_malicious": str(row["is_malicious_full"]).lower() == "true",
                "baseline_weight": float(row["aggregation_weight_full"]),
                "candidate_weight": float(row["aggregation_weight_candidate"]),
                "candidate_minus_full": float(
                    row["weight_delta_candidate_minus_full"]
                ),
                "baseline_semantic_risk": float(row["semantic_risk_full"]),
                "candidate_semantic_risk": float(row["semantic_risk_candidate"]),
                "baseline_semantic_q": float(row["semantic_q_full"]),
                "candidate_semantic_q": float(row["semantic_q_candidate"]),
                "baseline_cumulative_q": float(
                    row["rtc_v3_cumulative_q_full_full"]
                ),
                "candidate_cumulative_q": float(
                    row["rtc_v3_cumulative_q_full_candidate"]
                ),
                "baseline_client_q_cap": float(
                    row["rtc_v3_client_q_cap_full"]
                ),
                "candidate_client_q_cap": float(
                    row["rtc_v3_client_q_cap_candidate"]
                ),
            }
            if baseline_defense == "rtc_full":
                divergence.update(
                    {
                        "full_weight": divergence["baseline_weight"],
                        "full_semantic_risk": divergence[
                            "baseline_semantic_risk"
                        ],
                        "full_semantic_q": divergence["baseline_semantic_q"],
                        "full_cumulative_q": divergence[
                            "baseline_cumulative_q"
                        ],
                    }
                )
            first_divergence_rows.append(divergence)

    pd.DataFrame(client_group_rows).to_csv(
        output_dir / "client_group_summary.csv", index=False
    )
    pd.DataFrame(first_divergence_rows).to_csv(
        output_dir / "first_weight_divergence.csv", index=False
    )
    indexed = comparison.set_index(["attack", "mode"])
    lie_full = indexed.loc[("lie", baseline_defense)]
    lie_candidate = indexed.loc[("lie", candidate_defense)]
    dba_full = indexed.loc[("dba", baseline_defense)]
    dba_candidate = indexed.loc[("dba", candidate_defense)]

    if args.stage == "b1r":
        checks = {
            "lie_active_acc_gain_at_least_0_3pp": bool(
                lie_candidate["active_accuracy"] - lie_full["active_accuracy"]
                >= 0.003
            ),
            "dba_active_asr_at_most_10pct": bool(
                dba_candidate["active_asr"] <= 0.10
            ),
            "dba_malicious_weight_at_most_5pct": bool(
                dba_candidate["malicious_weight_share"] <= 0.05
            ),
            "dba_active_acc_drop_at_most_0_2pp": bool(
                dba_candidate["active_accuracy"] - dba_full["active_accuracy"]
                >= -0.002
            ),
        }
    else:
        checks = {
            "lie_malicious_weight_reduction_at_least_3pp": bool(
                lie_full["malicious_weight_share"]
                - lie_candidate["malicious_weight_share"]
                >= 0.03
            ),
            "lie_active_acc_drop_at_most_0_2pp": bool(
                lie_candidate["active_accuracy"] - lie_full["active_accuracy"]
                >= -0.002
            ),
            "dba_active_asr_at_most_10pct": bool(
                dba_candidate["active_asr"] <= 0.10
            ),
            "dba_malicious_weight_at_most_5pct": bool(
                dba_candidate["malicious_weight_share"] <= 0.05
            ),
            "dba_active_acc_drop_at_most_0_2pp": bool(
                dba_candidate["active_accuracy"] - dba_full["active_accuracy"]
                >= -0.002
            ),
        }
    checks.update(
        {
            "completion_status_all_passed": True,
            "quality_gates_all_passed": True,
        }
    )
    decision = {
        "stage": args.stage,
        "accepted_for_next_stage": all(checks.values()),
        "checks": checks,
        "deltas": {
            "lie_active_accuracy_candidate_minus_baseline": float(
                lie_candidate["active_accuracy"] - lie_full["active_accuracy"]
            ),
            "lie_final_accuracy_candidate_minus_baseline": float(
                lie_candidate["final_accuracy"] - lie_full["final_accuracy"]
            ),
            "dba_active_accuracy_candidate_minus_baseline": float(
                dba_candidate["active_accuracy"] - dba_full["active_accuracy"]
            ),
            "dba_active_asr_candidate_minus_baseline": float(
                dba_candidate["active_asr"] - dba_full["active_asr"]
            ),
            "dba_malicious_weight_candidate_minus_baseline": float(
                dba_candidate["malicious_weight_share"]
                - dba_full["malicious_weight_share"]
            ),
        },
        "trial_plan_hashes": {
            attack: _spec(
                candidate_dir, attack, candidate_defense
            )["trial_plan_hash"]
            for attack in ("lie", "dba")
        },
    }
    if args.stage == "b1r":
        decision["accepted_for_multiseed_validation"] = all(checks.values())
        decision["deltas"].update(
            {
                key.replace("_baseline", "_full"): value
                for key, value in list(decision["deltas"].items())
            }
        )
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
