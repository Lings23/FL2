"""Validate RTC-v3 B5 norm clipping MAD-k=2.25 against frozen B3R-F0.51."""

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
CANDIDATE = "rtc_b5_clip_mad_225"
DEFAULT_LIE_BASELINE = (
    ROOT / "logs" / "rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03"
)
DEFAULT_DBA_BASELINE = (
    ROOT / "logs" / "rtc_v3_anchor_recycle_b3r_f051_seed42_mf03"
)
DEFAULT_CANDIDATE = ROOT / "logs" / "rtc_v3_clip_mad_b5_225_seed42_mf03"
DEFAULT_OUTPUT = ROOT / "analysis" / "rtc_clip_mad_b5_225"


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _clip_summary(
    clients: pd.DataFrame, attack: str, mode: str
) -> dict[str, object]:
    frame = clients.copy()
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    frame["is_malicious"] = _bool_series(frame["is_malicious"])
    frame["clipped"] = _bool_series(frame["clipped"])
    clean = frame[frame["round"].between(1, 10)]
    active = frame[frame["round"].between(11, 60)]
    benign = active[~active["is_malicious"]]
    malicious = active[active["is_malicious"]]
    if clean.empty or benign.empty or malicious.empty:
        raise RuntimeError(f"{attack}/{mode}: incomplete client clipping windows")
    return {
        "attack": attack,
        "mode": mode,
        "preattack_clean_client_rows": len(clean),
        "preattack_clean_clipping_rate": float(clean["clipped"].mean()),
        "active_benign_client_rows": len(benign),
        "active_benign_clipping_rate": float(benign["clipped"].mean()),
        "active_malicious_client_rows": len(malicious),
        "active_malicious_clipping_rate": float(malicious["clipped"].mean()),
    }


def _training_rounds(frame: pd.DataFrame, attack: str) -> pd.DataFrame:
    """Return the 60 fit rounds, excluding the evaluation-only round 0 row."""
    training = frame[pd.to_numeric(frame["round"], errors="coerce") >= 1].copy()
    if len(training) != 60:
        raise RuntimeError(f"{attack}: expected 60 training rounds, got {len(training)}")
    return training


def _candidate_metrics(path: Path, attack: str) -> dict[str, object]:
    result = _metrics(path, attack, CANDIDATE)
    frame = pd.read_csv(path)
    training = _training_rounds(frame, attack)
    runtime = {
        "fit_rtc_v3_norm_clip_mad_k": 2.25,
        "fit_rtc_v3_anchor_recycle_fraction": 0.51,
    }
    for column, expected in runtime.items():
        if column not in training:
            raise RuntimeError(f"{attack}: runtime metric missing: {column}")
        values = pd.to_numeric(training[column], errors="coerce")
        if values.isna().any() or not ((values - expected).abs() <= 1e-12).all():
            raise RuntimeError(f"{attack}: runtime {column} is not {expected}")
    weighting = set(
        training["fit_rtc_v3_anchor_recycle_weighting"].dropna().astype(str)
    )
    if weighting != {"accepted"}:
        raise RuntimeError(f"{attack}: accepted-anchor weighting drifted")
    result["runtime_norm_clip_mad_k"] = 2.25
    result["active_clipping_rate"] = float(
        frame.loc[
            pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1,
            "fit_rtc_v3_clipping_rate",
        ].mean()
    )
    return result


def _first_divergence(
    baseline: pd.DataFrame, candidate: pd.DataFrame, attack: str
) -> list[dict[str, object]]:
    paired = baseline.merge(
        candidate,
        on=["round", "cid"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["baseline_clipped"] = _bool_series(paired["clipped_baseline"])
    paired["candidate_clipped"] = _bool_series(paired["clipped_candidate"])
    paired["clipped_norm_delta"] = (
        pd.to_numeric(paired["clipped_delta_norm_candidate"], errors="raise")
        - pd.to_numeric(paired["clipped_delta_norm_baseline"], errors="raise")
    )
    changed = paired[
        (paired["baseline_clipped"] != paired["candidate_clipped"])
        | (paired["clipped_norm_delta"].abs() > 1e-10)
    ]
    if changed.empty:
        return []
    first_round = int(pd.to_numeric(changed["round"], errors="raise").min())
    rows = changed[pd.to_numeric(changed["round"], errors="raise") == first_round]
    return [
        {
            "attack": attack,
            "round": first_round,
            "cid": str(row["cid"]),
            "is_malicious": str(row["is_malicious_candidate"]).lower() == "true",
            "baseline_clipped": bool(row["baseline_clipped"]),
            "candidate_clipped": bool(row["candidate_clipped"]),
            "baseline_clipped_delta_norm": float(row["clipped_delta_norm_baseline"]),
            "candidate_clipped_delta_norm": float(row["clipped_delta_norm_candidate"]),
            "candidate_minus_baseline_clipped_norm": float(row["clipped_norm_delta"]),
        }
        for _, row in rows.sort_values("cid").iterrows()
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
        raise RuntimeError(f"expected exactly two B5 specs, got {len(candidate_specs)}")

    comparisons: list[dict[str, object]] = []
    clip_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    trial_hashes: dict[str, str] = {}
    for attack in ATTACKS:
        baseline_dir = baseline_dirs[attack]
        baseline_spec = _spec(_specs(baseline_dir), attack, BASELINE)
        candidate_spec = _spec(candidate_specs, attack, CANDIDATE)
        if baseline_spec["trial_plan_hash"] != candidate_spec["trial_plan_hash"]:
            raise RuntimeError(f"{attack}: trial plans differ")
        if baseline_spec["attack_contract_hash"] != candidate_spec["attack_contract_hash"]:
            raise RuntimeError(f"{attack}: attack contracts differ")
        baseline_custom = dict(baseline_spec["custom_params"])
        candidate_custom = dict(candidate_spec["custom_params"])
        if candidate_custom.pop("norm_clip_mad_k", None) != 2.25:
            raise RuntimeError(f"{attack}: expected only norm_clip_mad_k=2.25")
        if candidate_custom != baseline_custom:
            raise RuntimeError(f"{attack}: B5 changes more than norm_clip_mad_k")
        required = {
            "semantic_intervention_risk_floor": 0.5,
            "cumulative_q_cap_power": 1,
            "anchor_recycle_fraction": 0.51,
            "anchor_recycle_weighting": "accepted",
        }
        for key, expected in required.items():
            if baseline_custom.get(key) != expected:
                raise RuntimeError(f"{attack}: frozen {key} drifted")
        if "residual_rank_cap_top_k" in candidate_custom:
            raise RuntimeError(f"{attack}: rejected B4 rank cap leaked into B5")
        trial_hashes[attack] = str(candidate_spec["trial_plan_hash"])
        _validate_completion(baseline_dir, attack, BASELINE)
        _validate_completion(candidate_dir, attack, CANDIDATE)

        baseline_round = _one_file(baseline_dir, "rounds", attack, BASELINE, ".csv")
        candidate_round = _one_file(candidate_dir, "rounds", attack, CANDIDATE, ".csv")
        baseline_clients_path = _one_file(
            baseline_dir, "raw", attack, BASELINE, "_clients.csv"
        )
        candidate_clients_path = _one_file(
            candidate_dir, "raw", attack, CANDIDATE, "_clients.csv"
        )
        comparisons.append(_metrics(baseline_round, attack, BASELINE))
        comparisons.append(_candidate_metrics(candidate_round, attack))
        baseline_clients = pd.read_csv(baseline_clients_path)
        candidate_clients = pd.read_csv(candidate_clients_path)
        clip_rows.append(_clip_summary(baseline_clients, attack, BASELINE))
        clip_rows.append(_clip_summary(candidate_clients, attack, CANDIDATE))
        divergence_rows.extend(
            _first_divergence(baseline_clients, candidate_clients, attack)
        )

    comparison = pd.DataFrame(comparisons)
    clipping = pd.DataFrame(clip_rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    clipping.to_csv(output_dir / "clipping_summary.csv", index=False)
    pd.DataFrame(divergence_rows).to_csv(
        output_dir / "first_clipping_divergence.csv", index=False
    )
    indexed = comparison.set_index(["attack", "mode"])
    clip_indexed = clipping.set_index(["attack", "mode"])
    deltas: list[dict[str, object]] = []
    checks: dict[str, bool] = {}
    for attack in ATTACKS:
        baseline = indexed.loc[(attack, BASELINE)]
        candidate = indexed.loc[(attack, CANDIDATE)]
        baseline_clip = clip_indexed.loc[(attack, BASELINE)]
        candidate_clip = clip_indexed.loc[(attack, CANDIDATE)]
        row: dict[str, object] = {"attack": attack}
        for metric in (
            "active_accuracy",
            "final_accuracy",
            "active_asr",
            "malicious_weight_share",
            "benign_weight_share",
            "malicious_impact_share",
            "zero_update_mass",
            "anchor_recycle_mass",
            "benign_watch_rate",
            "benign_restricted_rate",
            "benign_quarantined_rate",
        ):
            row[f"{metric}_candidate_minus_baseline"] = float(
                candidate[metric] - baseline[metric]
            )
        for metric in (
            "preattack_clean_clipping_rate",
            "active_benign_clipping_rate",
            "active_malicious_clipping_rate",
        ):
            row[f"{metric}_candidate_minus_baseline"] = float(
                candidate_clip[metric] - baseline_clip[metric]
            )
        deltas.append(row)
        checks[f"{attack}_active_accuracy_not_lower"] = bool(
            candidate["active_accuracy"] >= baseline["active_accuracy"] - 1e-12
        )
        checks[f"{attack}_preattack_clean_clipping_below_8pct"] = bool(
            candidate_clip["preattack_clean_clipping_rate"] < 0.08
        )
        checks[f"{attack}_active_benign_clipping_below_8pct"] = bool(
            candidate_clip["active_benign_clipping_rate"] < 0.08
        )
        checks[f"{attack}_malicious_weight_increase_at_most_0_5pp"] = bool(
            candidate["malicious_weight_share"] - baseline["malicious_weight_share"]
            <= 0.005 + 1e-12
        )
        checks[f"{attack}_malicious_impact_increase_at_most_0_5pp"] = bool(
            candidate["malicious_impact_share"] - baseline["malicious_impact_share"]
            <= 0.005 + 1e-12
        )
        checks[f"{attack}_zero_mass_increase_at_most_0_5pp"] = bool(
            candidate["zero_update_mass"] - baseline["zero_update_mass"]
            <= 0.005 + 1e-12
        )
        checks[f"{attack}_benign_state_increase_at_most_0_5pp"] = all(
            candidate[key] - baseline[key] <= 0.005 + 1e-12
            for key in ("benign_restricted_rate", "benign_quarantined_rate")
        )
    checks["lie_active_accuracy_strictly_improves"] = bool(
        indexed.loc[("lie", CANDIDATE), "active_accuracy"]
        > indexed.loc[("lie", BASELINE), "active_accuracy"] + 1e-12
    )
    checks["dba_active_asr_increase_at_most_0_5pp"] = bool(
        indexed.loc[("dba", CANDIDATE), "active_asr"]
        - indexed.loc[("dba", BASELINE), "active_asr"]
        <= 0.005 + 1e-12
    )
    checks.update(
        {
            "runtime_norm_clip_mad_k_exact": True,
            "candidate_completion_status_all_passed": True,
            "baseline_and_candidate_quality_gates_all_passed": True,
            "strict_pairing_and_single_parameter_all_passed": True,
        }
    )
    pd.DataFrame(deltas).to_csv(output_dir / "deltas.csv", index=False)
    decision = {
        "stage": "b5_clip_mad_2_25",
        "accepted_for_v1": all(checks.values()),
        "checks": checks,
        "trial_plan_hashes": trial_hashes,
        "tolerances": {
            "preattack_clean_and_active_benign_clipping_rate_max_exclusive": 0.08,
            "active_accuracy_candidate_minus_baseline_min": 0.0,
            "lie_active_accuracy_candidate_minus_baseline_strict_min": 0.0,
            "dba_active_asr_candidate_minus_baseline_max": 0.005,
            "malicious_weight_or_impact_candidate_minus_baseline_max": 0.005,
            "zero_mass_or_benign_state_candidate_minus_baseline_max": 0.005,
        },
        "deltas": deltas,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
