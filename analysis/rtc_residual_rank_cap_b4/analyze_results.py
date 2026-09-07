"""Validate the RTC-v3 B4 residual-rank-cap candidate."""

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
from analysis.rtc_anchor_recycle_b3r.analyze_results import (  # noqa: E402
    _all_gates_pass,
)


ATTACKS = ("lie", "dba")
BASELINE = "rtc_cumulative_q_cap_accepted_anchor"
CANDIDATE = "rtc_b4_residual_rank_cap"
DEFAULT_LIE_BASELINE = (
    ROOT / "logs" / "rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03"
)
DEFAULT_DBA_BASELINE = (
    ROOT / "logs" / "rtc_v3_anchor_recycle_b3r_f051_seed42_mf03"
)
DEFAULT_CANDIDATE = ROOT / "logs" / "rtc_v3_residual_rank_cap_b4_seed42_mf03"
DEFAULT_OUTPUT = ROOT / "analysis" / "rtc_residual_rank_cap_b4"
RANK_PARAMS = {
    "residual_rank_cap_top_k": 2,
    "residual_rank_cap_factor": 0.5,
    "residual_rank_recycle_fraction": 1.0,
}


def _active(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1
    ].copy()


def _constant_numeric(
    frame: pd.DataFrame, column: str, expected: float, *, tolerance: float = 1e-12
) -> bool:
    if column not in frame:
        raise RuntimeError(f"runtime metric is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and ((values - expected).abs() <= tolerance).all())


def _candidate_metrics(path: Path, attack: str) -> dict[str, object]:
    result = _metrics(path, attack, CANDIDATE)
    frame = pd.read_csv(path)
    active = _active(frame)
    expected = {
        "fit_rtc_v3_anchor_recycle_fraction": 0.51,
        "fit_rtc_v3_residual_rank_cap_top_k": 2.0,
        "fit_rtc_v3_residual_rank_cap_factor": 0.5,
        "fit_rtc_v3_residual_rank_recycle_fraction": 1.0,
        "fit_rtc_v3_residual_rank_cap_active_count": 2.0,
        "fit_rtc_v3_residual_rank_reference_valid": 1.0,
    }
    runtime_exact = all(
        _constant_numeric(active, column, value)
        for column, value in expected.items()
    )
    weighting = set(
        active["fit_rtc_v3_anchor_recycle_weighting"].dropna().astype(str)
    )
    if weighting != {"accepted"}:
        raise RuntimeError(
            f"{attack}: expected accepted anchor weighting, got {sorted(weighting)}"
        )
    removed = pd.to_numeric(
        active["fit_rtc_v3_residual_rank_removed_mass"], errors="coerce"
    )
    rank_target = pd.to_numeric(
        active["fit_rtc_v3_residual_rank_recycle_target_mass"], errors="coerce"
    )
    base_target = pd.to_numeric(
        active["fit_rtc_v3_anchor_recycle_base_target_mass"], errors="coerce"
    )
    total_target = pd.to_numeric(
        active["fit_rtc_v3_anchor_recycle_target_mass"], errors="coerce"
    )
    result.update(
        {
            "rank_runtime_exact": runtime_exact,
            "rank_reference_valid_rate": _numeric_mean(
                active, "fit_rtc_v3_residual_rank_reference_valid"
            ),
            "rank_removed_mass": float(removed.mean()),
            "rank_recycle_target_mass": float(rank_target.mean()),
            "anchor_base_target_mass": float(base_target.mean()),
            "rank_recycle_matches_removed": bool(
                removed.notna().all()
                and rank_target.notna().all()
                and ((rank_target - removed).abs() <= 1e-10).all()
            ),
            "anchor_target_decomposition_matches": bool(
                total_target.notna().all()
                and (
                    (total_target - base_target - rank_target).abs() <= 1e-10
                ).all()
            ),
            "rank_removed_mass_within_nominal_top2": bool(
                removed.notna().all()
                and (removed >= -1e-12).all()
                and (removed <= 0.1 + 1e-8).all()
            ),
        }
    )
    return result


def _rank_cap_audit(
    round_frame: pd.DataFrame, client_frame: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame]:
    active_rounds = _active(round_frame)
    clients = client_frame.copy()
    clients["round"] = pd.to_numeric(clients["round"], errors="raise").astype(int)
    clients["is_malicious"] = (
        clients["is_malicious"].astype(str).str.lower().eq("true")
    )
    clients["rtc_v3_residual_norm"] = pd.to_numeric(
        clients["rtc_v3_residual_norm"], errors="raise"
    )
    clients["rtc_v3_client_q_cap"] = pd.to_numeric(
        clients["rtc_v3_client_q_cap"], errors="raise"
    )
    audit_rows: list[dict[str, object]] = []
    all_rank_matches = True
    all_cap_guards = True
    for _, round_row in active_rounds.sort_values("round").iterrows():
        round_number = int(round_row["round"])
        group = clients[clients["round"] == round_number].copy()
        if group.empty:
            raise RuntimeError(f"missing client diagnostics for round {round_number}")
        ranked = sorted(
            group.to_dict("records"),
            key=lambda row: (
                -float(row["rtc_v3_residual_norm"]),
                str(row["principal_id"]),
                str(row["cid"]),
            ),
        )
        expected_ids = {str(row["cid"]) for row in ranked[:2]}
        observed_ids = {
            value
            for value in str(
                round_row["fit_rtc_v3_residual_rank_cap_client_ids"]
            ).split("|")
            if value
        }
        rank_matches = observed_ids == expected_ids
        selected = group[group["cid"].astype(str).isin(observed_ids)]
        cap_guard = bool(
            len(selected) == 2
            and (selected["rtc_v3_client_q_cap"] <= 0.5 + 1e-12).all()
        )
        all_rank_matches &= rank_matches
        all_cap_guards &= cap_guard
        for _, row in selected.iterrows():
            audit_rows.append(
                {
                    "round": round_number,
                    "cid": str(row["cid"]),
                    "principal_id": str(row["principal_id"]),
                    "is_malicious": bool(row["is_malicious"]),
                    "residual_norm": float(row["rtc_v3_residual_norm"]),
                    "client_q_cap": float(row["rtc_v3_client_q_cap"]),
                    "rank_matches": rank_matches,
                    "cap_guard_passed": cap_guard,
                }
            )
    hits = pd.DataFrame(audit_rows)
    active_clients = clients[clients["round"].isin(active_rounds["round"].astype(int))]
    benign_total = int((~active_clients["is_malicious"]).sum())
    malicious_total = int(active_clients["is_malicious"].sum())
    benign_hits = int((~hits["is_malicious"]).sum()) if not hits.empty else 0
    malicious_hits = int(hits["is_malicious"].sum()) if not hits.empty else 0
    audit = {
        "rank_identity_matches_all_rounds": all_rank_matches,
        "rank_cap_guard_passes_all_rounds": all_cap_guards,
        "rank_cap_hits": len(hits),
        "malicious_rank_cap_hits": malicious_hits,
        "benign_rank_cap_hits": benign_hits,
        "rank_cap_malicious_precision": (
            malicious_hits / len(hits) if len(hits) else 0.0
        ),
        "malicious_rank_cap_rate": (
            malicious_hits / malicious_total if malicious_total else 0.0
        ),
        "benign_rank_cap_rate": (
            benign_hits / benign_total if benign_total else 0.0
        ),
    }
    return audit, hits


def _client_group_rows(
    clients: pd.DataFrame, attack: str, mode: str
) -> list[dict[str, object]]:
    frame = clients.copy()
    frame["is_malicious"] = (
        frame["is_malicious"].astype(str).str.lower().eq("true")
    )
    active = frame[pd.to_numeric(frame["round"], errors="coerce") >= 11]
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
    return [
        {"attack": attack, "mode": mode, **row.to_dict()}
        for _, row in grouped.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lie-baseline-dir", type=Path, default=DEFAULT_LIE_BASELINE)
    parser.add_argument("--dba-baseline-dir", type=Path, default=DEFAULT_DBA_BASELINE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    baseline_dirs = {
        "lie": args.lie_baseline_dir.resolve(),
        "dba": args.dba_baseline_dir.resolve(),
    }
    candidate_dir = args.candidate_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for directory in {*baseline_dirs.values(), candidate_dir}:
        _all_gates_pass(directory)
    candidate_specs = _specs(candidate_dir)
    if len(candidate_specs) != 2:
        raise RuntimeError(f"expected two B4 candidate specs, got {len(candidate_specs)}")

    comparisons: list[dict[str, object]] = []
    trial_hashes: dict[str, str] = {}
    audits: list[dict[str, object]] = []
    hit_frames: list[pd.DataFrame] = []
    group_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    for attack in ATTACKS:
        baseline_dir = baseline_dirs[attack]
        baseline_spec = _spec(_specs(baseline_dir), attack, BASELINE)
        candidate_spec = _spec(candidate_specs, attack, CANDIDATE)
        if baseline_spec["trial_plan_hash"] != candidate_spec["trial_plan_hash"]:
            raise RuntimeError(f"{attack}: baseline/candidate trial plans differ")
        if baseline_spec["attack_contract_hash"] != candidate_spec["attack_contract_hash"]:
            raise RuntimeError(f"{attack}: baseline/candidate attack contracts differ")
        baseline_custom = dict(baseline_spec["custom_params"])
        candidate_custom = dict(candidate_spec["custom_params"])
        for key, expected in RANK_PARAMS.items():
            if candidate_custom.pop(key, None) != expected:
                raise RuntimeError(f"{attack}: unexpected B4 {key}")
        if candidate_custom != baseline_custom:
            raise RuntimeError(f"{attack}: B4 changes more than residual rank cap")
        required_frozen = {
            "semantic_intervention_risk_floor": 0.5,
            "cumulative_q_cap_power": 1,
            "anchor_recycle_fraction": 0.51,
            "anchor_recycle_weighting": "accepted",
        }
        for key, expected in required_frozen.items():
            if baseline_custom.get(key) != expected:
                raise RuntimeError(f"{attack}: frozen B3R-F0.51 {key} drifted")
        trial_hashes[attack] = str(candidate_spec["trial_plan_hash"])
        _validate_completion(baseline_dir, attack, BASELINE)
        _validate_completion(candidate_dir, attack, CANDIDATE)

        baseline_round_path = _one_file(
            baseline_dir, "rounds", attack, BASELINE, ".csv"
        )
        candidate_round_path = _one_file(
            candidate_dir, "rounds", attack, CANDIDATE, ".csv"
        )
        baseline_client_path = _one_file(
            baseline_dir, "raw", attack, BASELINE, "_clients.csv"
        )
        candidate_client_path = _one_file(
            candidate_dir, "raw", attack, CANDIDATE, "_clients.csv"
        )
        comparisons.append(_metrics(baseline_round_path, attack, BASELINE))
        comparisons.append(_candidate_metrics(candidate_round_path, attack))

        baseline_clients = pd.read_csv(baseline_client_path)
        candidate_clients = pd.read_csv(candidate_client_path)
        audit, hits = _rank_cap_audit(
            pd.read_csv(candidate_round_path), candidate_clients
        )
        audits.append({"attack": attack, **audit})
        if not hits.empty:
            hits.insert(0, "attack", attack)
            hit_frames.append(hits)
        group_rows.extend(_client_group_rows(baseline_clients, attack, BASELINE))
        group_rows.extend(_client_group_rows(candidate_clients, attack, CANDIDATE))

        paired = baseline_clients.merge(
            candidate_clients,
            on=["round", "cid"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired["weight_delta"] = (
            pd.to_numeric(paired["aggregation_weight_candidate"], errors="raise")
            - pd.to_numeric(paired["aggregation_weight_baseline"], errors="raise")
        )
        changed = paired[paired["weight_delta"].abs() > 1e-10]
        if not changed.empty:
            first_round = int(changed["round"].min())
            for _, row in changed[changed["round"] == first_round].sort_values("cid").iterrows():
                divergence_rows.append(
                    {
                        "attack": attack,
                        "round": first_round,
                        "cid": str(row["cid"]),
                        "is_malicious": str(row["is_malicious_candidate"]).lower()
                        == "true",
                        "baseline_weight": float(row["aggregation_weight_baseline"]),
                        "candidate_weight": float(row["aggregation_weight_candidate"]),
                        "candidate_minus_baseline": float(row["weight_delta"]),
                        "candidate_residual_norm": float(
                            row["rtc_v3_residual_norm_candidate"]
                        ),
                    }
                )

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(output_dir / "rank_cap_audit.csv", index=False)
    hit_output = (
        pd.concat(hit_frames, ignore_index=True)
        if hit_frames
        else pd.DataFrame()
    )
    hit_output.to_csv(output_dir / "rank_cap_hits.csv", index=False)
    pd.DataFrame(group_rows).to_csv(
        output_dir / "client_group_summary.csv", index=False
    )
    pd.DataFrame(divergence_rows).to_csv(
        output_dir / "first_weight_divergence.csv", index=False
    )

    indexed = comparison.set_index(["attack", "mode"])
    audit_indexed = audit_frame.set_index("attack")
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
        deltas.append(row)
        checks[f"{attack}_active_accuracy_not_lower"] = bool(
            candidate["active_accuracy"] >= baseline["active_accuracy"] - 1e-12
        )
        checks[f"{attack}_zero_mass_increase_at_most_0_5pp"] = bool(
            candidate["zero_update_mass"] - baseline["zero_update_mass"]
            <= 0.005 + 1e-12
        )
        checks[f"{attack}_benign_rank_cap_rate_at_most_10pct"] = bool(
            audit_indexed.loc[attack, "benign_rank_cap_rate"] <= 0.10 + 1e-12
        )
        checks[f"{attack}_benign_restricted_increase_at_most_0_5pp"] = bool(
            candidate["benign_restricted_rate"]
            - baseline["benign_restricted_rate"]
            <= 0.005 + 1e-12
        )
        checks[f"{attack}_benign_quarantine_increase_at_most_0_5pp"] = bool(
            candidate["benign_quarantined_rate"]
            - baseline["benign_quarantined_rate"]
            <= 0.005 + 1e-12
        )
        checks[f"{attack}_runtime_rank_contract_passed"] = bool(
            candidate["rank_runtime_exact"]
            and candidate["rank_recycle_matches_removed"]
            and candidate["anchor_target_decomposition_matches"]
            and candidate["rank_removed_mass_within_nominal_top2"]
            and audit_indexed.loc[attack, "rank_identity_matches_all_rounds"]
            and audit_indexed.loc[attack, "rank_cap_guard_passes_all_rounds"]
        )
    lie_base = indexed.loc[("lie", BASELINE)]
    lie_candidate = indexed.loc[("lie", CANDIDATE)]
    checks.update(
        {
            "lie_malicious_weight_at_most_15_5pct": bool(
                lie_candidate["malicious_weight_share"] <= 0.155 + 1e-12
            ),
            "lie_malicious_weight_decreases": bool(
                lie_candidate["malicious_weight_share"]
                < lie_base["malicious_weight_share"] - 1e-12
            ),
            "lie_malicious_impact_increase_at_most_0_5pp": bool(
                lie_candidate["malicious_impact_share"]
                - lie_base["malicious_impact_share"]
                <= 0.005 + 1e-12
            ),
            "lie_rank_removed_mass_is_positive": bool(
                lie_candidate["rank_removed_mass"] > 1e-12
            ),
        }
    )
    dba_base = indexed.loc[("dba", BASELINE)]
    dba_candidate = indexed.loc[("dba", CANDIDATE)]
    checks.update(
        {
            "dba_active_asr_increase_at_most_0_5pp": bool(
                dba_candidate["active_asr"] - dba_base["active_asr"]
                <= 0.005 + 1e-12
            ),
            "dba_malicious_weight_increase_at_most_0_5pp": bool(
                dba_candidate["malicious_weight_share"]
                - dba_base["malicious_weight_share"]
                <= 0.005 + 1e-12
            ),
            "dba_malicious_impact_increase_at_most_0_5pp": bool(
                dba_candidate["malicious_impact_share"]
                - dba_base["malicious_impact_share"]
                <= 0.005 + 1e-12
            ),
        }
    )
    checks.update(
        {
            "candidate_completion_status_all_passed": True,
            "baseline_and_candidate_quality_gates_all_passed": True,
            "strict_pairing_and_single_mechanism_all_passed": True,
        }
    )
    pd.DataFrame(deltas).to_csv(output_dir / "deltas.csv", index=False)
    decision = {
        "stage": "b4",
        "accepted_for_next_stage": all(checks.values()),
        "checks": checks,
        "trial_plan_hashes": trial_hashes,
        "tolerances": {
            "lie_malicious_weight_share_max": 0.155,
            "active_accuracy_candidate_minus_baseline_min": 0.0,
            "dba_active_asr_candidate_minus_baseline_max": 0.005,
            "malicious_weight_or_impact_candidate_minus_baseline_max": 0.005,
            "zero_mass_candidate_minus_baseline_max": 0.005,
            "benign_rank_cap_rate_max": 0.10,
            "benign_restricted_or_quarantine_increase_max": 0.005,
        },
        "deltas": deltas,
        "rank_cap_audit": audits,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
