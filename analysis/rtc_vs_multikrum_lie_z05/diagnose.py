"""Reproducible diagnosis of RTC versus Multi-Krum on paired LIE z=0.5 logs.

The script is intentionally read-only with respect to the source logs.  It
writes compact, report-ready tables next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


WORKSPACE = Path(__file__).resolve().parents[2]
SCREEN_ROOT = WORKSPACE / "logs" / "rtc_v3_byzantine_screen_agr"
RUN_ROOT = WORKSPACE / "logs" / "rtc_v3_lie_z05_seed42_mf03"
OUTPUT = Path(__file__).resolve().parent
ATTACK_START = 11
EXPECTED_ROUNDS = 60


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _run_spec(runs: pd.DataFrame, *, defense: str, attack: str = "lie") -> pd.Series:
    rows = runs[
        (runs["attack"].astype(str) == attack)
        & (runs["defense"].astype(str) == defense)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one run for attack={attack}, defense={defense}; got {len(rows)}")
    return rows.iloc[0]


def _rounds(run_id: str) -> pd.DataFrame:
    path = RUN_ROOT / "rounds" / f"{run_id}.csv"
    frame = pd.read_csv(path)
    observed = _finite_numeric(frame["round"]).astype(int).tolist()
    expected = list(range(EXPECTED_ROUNDS + 1))
    if observed != expected:
        raise ValueError(f"round sequence mismatch for {run_id}")
    if frame["round"].duplicated().any():
        raise ValueError(f"duplicate rounds for {run_id}")
    accuracy = _finite_numeric(frame["server_accuracy"])
    if accuracy.isna().any() or not accuracy.between(0.0, 1.0).all():
        raise ValueError(f"invalid accuracy for {run_id}")
    return frame


def _clients(run_id: str) -> pd.DataFrame:
    path = RUN_ROOT / "raw" / f"{run_id}_clients.csv"
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_ROUNDS * 10:
        raise ValueError(f"expected 600 client observations for {run_id}; got {len(frame)}")
    if frame.duplicated(["round", "cid"]).any():
        raise ValueError(f"duplicate round/client rows for {run_id}")
    return frame


def _status(run_id: str) -> dict:
    return json.loads((RUN_ROOT / "status" / f"{run_id}.json").read_text(encoding="utf-8"))


def _binary_auc(scores: pd.Series, positive: pd.Series) -> float:
    valid = scores.notna() & positive.notna()
    values = scores[valid].to_numpy(dtype=float)
    labels = positive[valid].astype(bool).to_numpy()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(values, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _block_bootstrap_mean_ci(values: np.ndarray, *, block: int = 5, draws: int = 20_000) -> tuple[float, float]:
    """Circular block bootstrap CI for within-run temporal uncertainty only."""

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260831)
    n = values.size
    blocks_needed = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[..., None] + offsets) % n
    sampled = values[indices.reshape(draws, -1)[:, :n]]
    means = sampled.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RUN_ROOT / "periodic_attack_runs.csv")
    attacked = runs[runs["attack"].astype(str) == "lie"].copy()
    if set(attacked["defense"].astype(str)) != {
        "rtc_full", "fedavg", "krum", "multi_krum", "median"
    }:
        raise ValueError("attacked defense matrix is incomplete")
    if attacked["trial_plan_hash"].astype(str).nunique() != 1:
        raise ValueError("attacked defenses are not strictly paired")
    if set(_finite_numeric(attacked["lie_z"]).dropna()) != {0.5}:
        raise ValueError("the run is not the expected LIE z=0.5 condition")
    if set(_finite_numeric(attacked["malicious_fraction"]).dropna()) != {0.2}:
        raise ValueError("the executed run uses malicious_fraction other than 0.2")

    run_specs = {
        defense: _run_spec(runs, defense=defense)
        for defense in ("rtc_full", "multi_krum", "fedavg", "krum", "median")
    }
    clean_rows = runs[
        (runs["attack"].astype(str) == "none")
        & (runs["defense"].astype(str) == "fedavg")
        & (runs["trial_plan_hash"].astype(str) == attacked.iloc[0]["trial_plan_hash"])
    ]
    if len(clean_rows) != 1:
        raise ValueError("missing plan-matched clean FedAvg run")
    clean_spec = clean_rows.iloc[0]

    all_specs = [*run_specs.values(), clean_spec]
    for spec in all_specs:
        status = _status(str(spec["run_id"]))
        if status.get("state") != "completed" or int(status.get("last_round", -1)) != EXPECTED_ROUNDS:
            raise ValueError(f"incomplete status for {spec['run_id']}")

    round_frames = {
        key: _rounds(str(spec["run_id"])) for key, spec in run_specs.items()
    }
    clean_rounds = _rounds(str(clean_spec["run_id"]))
    rtc_rounds = round_frames["rtc_full"]
    mk_rounds = round_frames["multi_krum"]

    round_comparison = pd.DataFrame(
        {
            "round": _finite_numeric(rtc_rounds["round"]).astype(int),
            "phase": np.select(
                [
                    _finite_numeric(rtc_rounds["round"]).between(1, 10),
                    _finite_numeric(rtc_rounds["round"]).between(11, 30),
                    _finite_numeric(rtc_rounds["round"]).between(31, 60),
                ],
                ["warmup", "steady", "late"],
                default="initial",
            ),
            "attack_active": _finite_numeric(rtc_rounds["planned_attack_active"]).fillna(0).astype(int),
            "rtc_accuracy": _finite_numeric(rtc_rounds["server_accuracy"]),
            "multi_krum_accuracy": _finite_numeric(mk_rounds["server_accuracy"]),
            "clean_fedavg_accuracy": _finite_numeric(clean_rounds["server_accuracy"]),
            "rtc_attacker_weight": _finite_numeric(rtc_rounds["fit_active_attacker_weight_share"]),
            "multi_krum_attacker_weight": _finite_numeric(mk_rounds["fit_active_attacker_weight_share"]),
            "selected_attackers": _finite_numeric(rtc_rounds["fit_selected_active_attackers"]),
            "rtc_zero_update_mass": _finite_numeric(rtc_rounds["fit_rtc_v3_zero_update_mass"]),
            "rtc_clip_rate": _finite_numeric(rtc_rounds["fit_rtc_v3_clipping_rate"]),
            "rtc_semantic_risk_mean": _finite_numeric(rtc_rounds["fit_rtc_v3_semantic_risk_mean"]),
            "rtc_semantic_risk_max": _finite_numeric(rtc_rounds["fit_rtc_v3_semantic_risk_max"]),
            "rtc_semantic_sync_pairs": _finite_numeric(rtc_rounds["fit_rtc_v3_semantic_synchronized_pair_count"]),
            "rtc_cumulative_active": _finite_numeric(rtc_rounds["fit_rtc_v3_cumulative_active_count"]),
            "rtc_direction_active": _finite_numeric(rtc_rounds["fit_rtc_v3_direction_active_count"]),
            "rtc_principal_budget_active": _finite_numeric(rtc_rounds["fit_rtc_v3_principal_budget_active_count"]),
        }
    )
    round_comparison["rtc_minus_multi_krum_pp"] = (
        round_comparison["rtc_accuracy"] - round_comparison["multi_krum_accuracy"]
    ) * 100.0
    round_comparison["rtc_excess_attacker_weight_pp"] = (
        round_comparison["rtc_attacker_weight"] - round_comparison["multi_krum_attacker_weight"]
    ) * 100.0
    round_comparison.to_csv(OUTPUT / "round_comparison.csv", index=False)

    windows = [
        ("clean warmup", 1, 10),
        ("attack 11-20", 11, 20),
        ("attack 21-30", 21, 30),
        ("attack 31-40", 31, 40),
        ("attack 41-50", 41, 50),
        ("attack 51-60", 51, 60),
        ("all active", 11, 60),
    ]
    window_rows = []
    for label, start, end in windows:
        part = round_comparison[round_comparison["round"].between(start, end)]
        window_rows.append(
            {
                "window": label,
                "start_round": start,
                "end_round": end,
                "rounds": len(part),
                "rtc_accuracy": part["rtc_accuracy"].mean(),
                "multi_krum_accuracy": part["multi_krum_accuracy"].mean(),
                "rtc_minus_multi_krum_pp": part["rtc_minus_multi_krum_pp"].mean(),
                "rtc_below_multi_krum_round_share": (part["rtc_minus_multi_krum_pp"] < 0).mean(),
                "rtc_attacker_weight": part["rtc_attacker_weight"].mean(),
                "multi_krum_attacker_weight": part["multi_krum_attacker_weight"].mean(),
            }
        )
    window_summary = pd.DataFrame(window_rows)
    window_summary.to_csv(OUTPUT / "window_summary.csv", index=False)

    active_rounds = round_comparison[round_comparison["round"].between(ATTACK_START, EXPECTED_ROUNDS)].copy()
    gap_ci = _block_bootstrap_mean_ci(active_rounds["rtc_minus_multi_krum_pp"].to_numpy())
    corr = spearmanr(
        active_rounds["rtc_excess_attacker_weight_pp"],
        active_rounds["rtc_minus_multi_krum_pp"],
        nan_policy="omit",
    )

    rtc_clients = _clients(str(run_specs["rtc_full"]["run_id"]))
    mk_clients = _clients(str(run_specs["multi_krum"]["run_id"]))
    rtc_active_clients = rtc_clients[_finite_numeric(rtc_clients["round"]) >= ATTACK_START].copy()
    mk_active_clients = mk_clients[_finite_numeric(mk_clients["round"]) >= ATTACK_START].copy()
    for frame in (rtc_active_clients, mk_active_clients):
        frame["is_malicious"] = frame["is_malicious"].astype(str).str.lower().eq("true")

    signal_definitions = {
        "semantic total risk": _finite_numeric(rtc_active_clients["total_risk"]),
        "semantic magnitude risk": _finite_numeric(rtc_active_clients["magnitude_risk"]),
        "semantic direction risk": _finite_numeric(rtc_active_clients["direction_risk"]),
        "semantic influence risk": _finite_numeric(rtc_active_clients["influence_risk"]),
        "semantic max z": _finite_numeric(rtc_active_clients["semantic_z"]),
        "one minus trust": 1.0 - _finite_numeric(rtc_active_clients["trust"]),
        "raw update norm": _finite_numeric(rtc_active_clients["raw_delta_norm"]),
        "residual norm": _finite_numeric(rtc_active_clients["rtc_v3_residual_norm"]),
        "one minus cone similarity": 1.0 - _finite_numeric(rtc_active_clients["rtc_v3_cone_similarity_full"]),
        "one minus cumulative q": 1.0 - _finite_numeric(rtc_active_clients["rtc_v3_cumulative_q_full"]),
        "one minus direction q": 1.0 - _finite_numeric(rtc_active_clients["rtc_v3_direction_q_full"]),
    }
    separation_rows = []
    malicious_mask = rtc_active_clients["is_malicious"]
    for signal, values in signal_definitions.items():
        valid = values.notna()
        mal = values[valid & malicious_mask]
        benign = values[valid & ~malicious_mask]
        pooled_sd = float(values[valid].std(ddof=1))
        delta = float(mal.mean() - benign.mean()) if len(mal) and len(benign) else float("nan")
        separation_rows.append(
            {
                "signal": signal,
                "higher_means_more_suspicious": True,
                "malicious_observations": len(mal),
                "benign_observations": len(benign),
                "malicious_mean": mal.mean(),
                "benign_mean": benign.mean(),
                "malicious_median": mal.median(),
                "benign_median": benign.median(),
                "malicious_minus_benign": delta,
                "standardized_mean_difference": delta / pooled_sd if pooled_sd > 0 else np.nan,
                "malicious_detection_auc": _binary_auc(values, malicious_mask),
            }
        )
    signal_separation = pd.DataFrame(separation_rows)
    signal_separation.to_csv(OUTPUT / "rtc_signal_separation.csv", index=False)

    state_rows = []
    for is_malicious, group in rtc_active_clients.groupby("is_malicious", observed=True):
        denom = len(group)
        states = group["state"].fillna("missing").astype(str)
        raw_norm = _finite_numeric(group["raw_delta_norm"])
        clipped_norm = _finite_numeric(group["clipped_delta_norm"])
        state_rows.append(
            {
                "group": "malicious" if is_malicious else "benign",
                "observations": denom,
                "positive_weight_rate": (_finite_numeric(group["aggregation_weight"]) > 1e-12).mean(),
                "mean_aggregation_weight": _finite_numeric(group["aggregation_weight"]).mean(),
                "clip_rate": group["clipped"].astype(str).str.lower().eq("true").mean(),
                "mean_clip_factor": (clipped_norm / raw_norm.replace(0.0, np.nan)).mean(),
                "cap_rate": group["capped"].astype(str).str.lower().eq("true").mean(),
                "mean_raw_update_norm": raw_norm.mean(),
                "mean_residual_norm": _finite_numeric(group["rtc_v3_residual_norm"]).mean(),
                "mean_semantic_risk": _finite_numeric(group["total_risk"]).mean(),
                "mean_cumulative_q": _finite_numeric(group["rtc_v3_cumulative_q_full"]).mean(),
                "normal_rate": states.eq("normal").mean(),
                "watch_rate": states.eq("watch").mean(),
                "restricted_rate": states.eq("restricted").mean(),
                "quarantined_rate": states.eq("quarantined").mean(),
            }
        )
    rtc_state_summary = pd.DataFrame(state_rows)
    rtc_state_summary.to_csv(OUTPUT / "rtc_state_summary.csv", index=False)

    mechanism_rows = []
    for defense, frame, client_frame in (
        ("RTC", rtc_rounds, rtc_active_clients),
        ("Multi-Krum", mk_rounds, mk_active_clients),
    ):
        active = frame[_finite_numeric(frame["round"]) >= ATTACK_START]
        nominal_share = (
            _finite_numeric(active["fit_selected_active_attackers"])
            / (
                _finite_numeric(active["fit_selected_active_attackers"])
                + _finite_numeric(active["fit_selected_benign_clients"])
            )
        )
        actual_share = _finite_numeric(active["fit_active_attacker_weight_share"])
        malicious_obs = client_frame[client_frame["is_malicious"]]
        benign_obs = client_frame[~client_frame["is_malicious"]]
        mechanism_rows.append(
            {
                "defense": defense,
                "active_rounds": len(active),
                "mean_attackers_selected": _finite_numeric(active["fit_selected_active_attackers"]).mean(),
                "nominal_attacker_share": nominal_share.mean(),
                "actual_attacker_weight_share": actual_share.mean(),
                "attacker_weight_retention": actual_share.mean() / nominal_share.mean(),
                "malicious_positive_weight_rate": (_finite_numeric(malicious_obs["aggregation_weight"]) > 1e-12).mean(),
                "benign_positive_weight_rate": (_finite_numeric(benign_obs["aggregation_weight"]) > 1e-12).mean(),
                "malicious_mean_weight": _finite_numeric(malicious_obs["aggregation_weight"]).mean(),
                "benign_mean_weight": _finite_numeric(benign_obs["aggregation_weight"]).mean(),
            }
        )
    weight_mechanism = pd.DataFrame(mechanism_rows)
    weight_mechanism.to_csv(OUTPUT / "weight_mechanism.csv", index=False)

    # These are weight-path diagnostics, not ACC counterfactuals.  Proposed
    # variants route unused client mass to RTC's already-computed robust anchor.
    weight_scenarios = []
    for round_value, group in rtc_active_clients.groupby("round", sort=True):
        nominal = _finite_numeric(group["nominal_mass"]).to_numpy(dtype=float)
        current = _finite_numeric(group["aggregation_weight"]).to_numpy(dtype=float)
        cumulative_q = _finite_numeric(group["rtc_v3_cumulative_q_full"]).fillna(1.0).to_numpy(dtype=float)
        residual = _finite_numeric(group["rtc_v3_residual_norm"]).to_numpy(dtype=float)
        malicious = group["is_malicious"].to_numpy(dtype=bool)
        variants: dict[str, np.ndarray] = {
            "current RTC": current,
            "semantic-observe proxy (nominal weights)": nominal,
            "direct cumulative-q cap": nominal * cumulative_q,
            "direct cumulative-q squared cap": nominal * np.square(cumulative_q),
        }
        order = np.argsort(-residual, kind="stable")
        for top_k, factor in ((2, 0.5), (3, 0.5), (2, 0.0)):
            candidate = nominal.copy()
            candidate[order[:top_k]] *= factor
            variants[f"top-{top_k} residual cap x{factor:g}"] = candidate
        for scenario, weights in variants.items():
            client_mass = float(weights.sum())
            weight_scenarios.append(
                {
                    "round": int(round_value),
                    "scenario": scenario,
                    "attacker_client_mass": float(weights[malicious].sum()),
                    "benign_client_mass": float(weights[~malicious].sum()),
                    "robust_anchor_or_zero_mass": max(0.0, 1.0 - client_mass),
                    "client_weight_sum": client_mass,
                }
            )
    weight_scenarios_frame = pd.DataFrame(weight_scenarios)
    weight_scenarios_frame.to_csv(OUTPUT / "weight_scenarios_by_round.csv", index=False)
    weight_scenario_summary = (
        weight_scenarios_frame.groupby("scenario", sort=False, as_index=False)
        .agg(
            attacker_client_mass=("attacker_client_mass", "mean"),
            benign_client_mass=("benign_client_mass", "mean"),
            robust_anchor_or_zero_mass=("robust_anchor_or_zero_mass", "mean"),
            client_weight_sum=("client_weight_sum", "mean"),
        )
    )
    weight_scenario_summary["attacker_mass_reduction_vs_current"] = 1.0 - (
        weight_scenario_summary["attacker_client_mass"]
        / float(weight_scenario_summary.iloc[0]["attacker_client_mass"])
    )
    weight_scenario_summary.to_csv(OUTPUT / "weight_scenario_summary.csv", index=False)

    defense_accuracy_rows = []
    for defense, frame in round_frames.items():
        active = frame[_finite_numeric(frame["round"]) >= ATTACK_START]
        attacker_weight = (
            _finite_numeric(active["fit_active_attacker_weight_share"]).mean()
            if "fit_active_attacker_weight_share" in active.columns
            else np.nan
        )
        defense_accuracy_rows.append(
            {
                "defense": defense,
                "active_accuracy": _finite_numeric(active["server_accuracy"]).mean(),
                "final_accuracy": _finite_numeric(frame["server_accuracy"]).iloc[-1],
                "active_attacker_weight": attacker_weight,
                "active_min_accuracy": _finite_numeric(active["server_accuracy"]).min(),
            }
        )
    defense_accuracy = pd.DataFrame(defense_accuracy_rows)
    defense_accuracy.to_csv(OUTPUT / "defense_accuracy_summary.csv", index=False)

    mk_active = mk_rounds[_finite_numeric(mk_rounds["round"]) >= ATTACK_START].copy()
    mk_actual_attackers = _finite_numeric(mk_active["fit_selected_active_attackers"])
    configured_f = int(run_specs["multi_krum"]["krum_num_malicious"])

    class_rows = []
    for class_index in range(10):
        rtc_col = _finite_numeric(rtc_rounds[f"server_class_recall_{class_index}"])
        mk_col = _finite_numeric(mk_rounds[f"server_class_recall_{class_index}"])
        mask = _finite_numeric(rtc_rounds["round"]) >= ATTACK_START
        class_rows.append(
            {
                "class": class_index,
                "rtc_active_recall": rtc_col[mask].mean(),
                "multi_krum_active_recall": mk_col[mask].mean(),
                "rtc_minus_multi_krum_pp": (rtc_col[mask].mean() - mk_col[mask].mean()) * 100.0,
                "rtc_final_recall": rtc_col.iloc[-1],
                "multi_krum_final_recall": mk_col.iloc[-1],
                "final_gap_pp": (rtc_col.iloc[-1] - mk_col.iloc[-1]) * 100.0,
            }
        )
    pd.DataFrame(class_rows).to_csv(OUTPUT / "class_recall_comparison.csv", index=False)

    screening = pd.read_csv(SCREEN_ROOT / "attack_strength_screening.csv")
    lie_screen = screening[screening["attack"].astype(str) == "lie"]
    if len(lie_screen) != 1 or str(lie_screen.iloc[0]["monotonic_with_tolerance"]).lower() != "true":
        raise ValueError("LIE screening evidence is incomplete or non-monotonic")

    statistical_tests_path = RUN_ROOT / "statistical_tests.csv"
    try:
        statistical_tests_rows = len(pd.read_csv(statistical_tests_path))
    except pd.errors.EmptyDataError:
        statistical_tests_rows = 0

    summary = {
        "scope": {
            "attack": "lie",
            "lie_z": 0.5,
            "screening_strength_label": "medium",
            "malicious_fraction_requested_label": "mf03 in directory name",
            "malicious_fraction_executed": 0.2,
            "seed": 42,
            "partition": "iid",
            "active_rounds": "11-60",
            "trial_plan_hash": str(attacked.iloc[0]["trial_plan_hash"]),
        },
        "accuracy": {
            "rtc_active_mean": float(active_rounds["rtc_accuracy"].mean()),
            "multi_krum_active_mean": float(active_rounds["multi_krum_accuracy"].mean()),
            "rtc_minus_multi_krum_active_pp": float(active_rounds["rtc_minus_multi_krum_pp"].mean()),
            "rtc_minus_multi_krum_block_bootstrap_95pct_pp": list(gap_ci),
            "rtc_below_multi_krum_active_round_share": float((active_rounds["rtc_minus_multi_krum_pp"] < 0).mean()),
            "rtc_final": float(round_comparison.iloc[-1]["rtc_accuracy"]),
            "multi_krum_final": float(round_comparison.iloc[-1]["multi_krum_accuracy"]),
            "final_gap_pp": float(round_comparison.iloc[-1]["rtc_minus_multi_krum_pp"]),
            "round10_gap_pp": float(round_comparison.loc[round_comparison["round"] == 10, "rtc_minus_multi_krum_pp"].iloc[0]),
            "excess_weight_gap_spearman_r": float(corr.statistic),
            "excess_weight_gap_spearman_p_descriptive_only": float(corr.pvalue),
            "fedavg_active_mean": float(defense_accuracy.loc[defense_accuracy["defense"] == "fedavg", "active_accuracy"].iloc[0]),
            "rtc_minus_fedavg_active_pp": float(
                100.0
                * (
                    defense_accuracy.loc[defense_accuracy["defense"] == "rtc_full", "active_accuracy"].iloc[0]
                    - defense_accuracy.loc[defense_accuracy["defense"] == "fedavg", "active_accuracy"].iloc[0]
                )
            ),
            "multi_krum_minus_fedavg_active_pp": float(
                100.0
                * (
                    defense_accuracy.loc[defense_accuracy["defense"] == "multi_krum", "active_accuracy"].iloc[0]
                    - defense_accuracy.loc[defense_accuracy["defense"] == "fedavg", "active_accuracy"].iloc[0]
                )
            ),
        },
        "mechanism": {
            "rtc_attacker_weight_share": float(weight_mechanism.loc[weight_mechanism["defense"] == "RTC", "actual_attacker_weight_share"].iloc[0]),
            "multi_krum_attacker_weight_share": float(weight_mechanism.loc[weight_mechanism["defense"] == "Multi-Krum", "actual_attacker_weight_share"].iloc[0]),
            "multi_krum_weight_reduction_vs_rtc": float(
                1.0
                - weight_mechanism.loc[weight_mechanism["defense"] == "Multi-Krum", "actual_attacker_weight_share"].iloc[0]
                / weight_mechanism.loc[weight_mechanism["defense"] == "RTC", "actual_attacker_weight_share"].iloc[0]
            ),
            "rtc_mean_zero_update_mass": float(active_rounds["rtc_zero_update_mass"].mean()),
            "rtc_zero_mass_positive_round_share": float((active_rounds["rtc_zero_update_mass"] > 1e-10).mean()),
            "rtc_mean_clip_rate": float(active_rounds["rtc_clip_rate"].mean()),
            "rtc_semantic_sync_active_round_share": float((active_rounds["rtc_semantic_sync_pairs"] > 0).mean()),
            "rtc_semantic_risk_positive_round_share": float((active_rounds["rtc_semantic_risk_max"] > 0).mean()),
            "rtc_cumulative_active_round_share": float((active_rounds["rtc_cumulative_active"] > 0).mean()),
            "rtc_direction_active_round_share": float((active_rounds["rtc_direction_active"] > 0).mean()),
            "rtc_principal_budget_active_round_share": float((active_rounds["rtc_principal_budget_active"] > 0).mean()),
            "multi_krum_actual_attackers_exceed_configured_f_round_share": float((mk_actual_attackers > configured_f).mean()),
            "multi_krum_configured_f": configured_f,
        },
        "data_quality": {
            "quality_gates_passed": bool(pd.read_csv(RUN_ROOT / "quality_gates.csv")["passed"].astype(str).str.lower().eq("true").all()),
            "completed_attacked_cells": int(len(attacked)),
            "round_rows_per_cell": EXPECTED_ROUNDS + 1,
            "client_rows_per_cell": EXPECTED_ROUNDS * 10,
            "statistical_tests_rows": int(statistical_tests_rows),
            "inference_limit": "single seed; round-level observations are path-dependent and not independent replicates",
            "counterfactual_limit": "only a clean FedAvg run exists; there is no clean RTC or clean Multi-Krum counterfactual",
            "directory_label_mismatch": "directory says mf03, executed configs and manifests say malicious_fraction=0.2",
        },
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
