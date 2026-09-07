"""Summarize the paired RTC semantic-observe safety ablation.

This analysis is deliberately read-only with respect to experiment logs.  It
materializes compact CSV/JSON evidence under this analysis directory so the
mechanism conclusions can be reproduced without rerunning training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "rtc_v3_semantic_observe_safety_seed42_mf03"
ROUND_DIR = LOG_DIR / "rounds"
RAW_DIR = LOG_DIR / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent


def _mode(path: Path) -> str:
    return "observe" if "rtc_semantic_observe" in path.name else "full"


def _attack(path: Path) -> str:
    return "gaussian_noise" if path.name.startswith("gaussian_noise") else "dba"


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    run_rows: list[dict[str, object]] = []
    round_frames: list[pd.DataFrame] = []
    client_frames: list[pd.DataFrame] = []

    for path in sorted(ROUND_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        attack = _attack(path)
        mode = _mode(path)
        frame["attack"] = attack
        frame["mode"] = mode
        round_frames.append(frame)
        active = frame[pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1]
        row: dict[str, object] = {
            "attack": attack,
            "mode": mode,
            "active_rounds": int(len(active)),
            "final_accuracy": float(frame.iloc[-1]["server_accuracy"]),
            "active_accuracy_mean": float(active["server_accuracy"].mean()),
            "malicious_weight_share_mean": float(
                active["fit_malicious_aggregation_weight_share"].mean()
            ),
            "malicious_impact_share_mean": float(
                active["fit_malicious_impact_share"].mean()
            ),
            "benign_watch_rate_mean": float(active["fit_benign_watch_rate"].mean()),
            "benign_restricted_rate_mean": float(
                active["fit_benign_restricted_rate"].mean()
            ),
            "malicious_quarantined_rate_mean": float(
                active["fit_malicious_quarantined_rate"].mean()
            ),
            "weight_sum_mean": float(active["fit_rtc_v3_weight_sum"].mean()),
            "zero_update_mass_mean": float(
                active["fit_rtc_v3_zero_update_mass"].mean()
            ),
        }
        if attack == "dba":
            row.update(
                {
                    "active_asr_mean": float(
                        active["server_dba_full_trigger_asr"].mean()
                    ),
                    "peak_asr": float(active["server_dba_full_trigger_asr"].max()),
                }
            )
        run_rows.append(row)

    for path in sorted(RAW_DIR.glob("*_clients.csv")):
        frame = pd.read_csv(path)
        frame["attack"] = _attack(path)
        frame["mode"] = _mode(path)
        frame["is_malicious"] = _bool(frame["is_malicious"])
        frame["attack_active"] = _bool(frame["attack_active"])
        client_frames.append(frame)

    rounds = pd.concat(round_frames, ignore_index=True)
    clients = pd.concat(client_frames, ignore_index=True)
    run_summary = pd.DataFrame(run_rows).sort_values(["attack", "mode"])
    run_summary.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)

    active_clients = clients[clients["attack_active"]].copy()
    per_round_group = (
        active_clients.groupby(["attack", "mode", "round", "is_malicious"], as_index=False)
        .agg(
            selected_clients=("cid", "size"),
            aggregation_weight_sum=("aggregation_weight", "sum"),
            effective_weight_sum=("effective_weight", "sum"),
            impact_norm_sum=("impact_norm", "sum"),
            semantic_risk_mean=("semantic_risk", "mean"),
            semantic_q_mean=("semantic_q", "mean"),
            cumulative_q_mean=("rtc_v3_cumulative_q_full", "mean"),
            direction_q_mean=("rtc_v3_direction_q_full", "mean"),
            quarantine_rate=("quarantined", lambda values: float(_bool(values).mean())),
        )
    )
    group_summary = (
        per_round_group.groupby(["attack", "mode", "is_malicious"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["round"])
    )
    group_summary.to_csv(OUTPUT_DIR / "client_group_summary.csv", index=False)

    first_rows: list[dict[str, object]] = []
    paired_rows: list[pd.DataFrame] = []
    for attack in sorted(clients["attack"].unique()):
        subset = clients[clients["attack"] == attack]
        full = subset[subset["mode"] == "full"].copy()
        observe = subset[subset["mode"] == "observe"].copy()
        paired = full.merge(
            observe,
            on=["round", "cid"],
            suffixes=("_full", "_observe"),
            validate="one_to_one",
        )
        paired["attack"] = attack
        paired["aggregation_weight_delta_observe_minus_full"] = (
            paired["aggregation_weight_observe"] - paired["aggregation_weight_full"]
        )
        paired["effective_weight_delta_observe_minus_full"] = (
            paired["effective_weight_observe"] - paired["effective_weight_full"]
        )
        paired_rows.append(
            paired[
                [
                    "attack",
                    "round",
                    "cid",
                    "is_malicious_full",
                    "attack_active_full",
                    "aggregation_weight_full",
                    "aggregation_weight_observe",
                    "aggregation_weight_delta_observe_minus_full",
                    "semantic_risk_full",
                    "semantic_q_full",
                    "rtc_v3_cumulative_q_full_full",
                    "rtc_v3_direction_q_full_full",
                    "state_full",
                ]
            ]
        )
        changed = paired[
            paired["aggregation_weight_delta_observe_minus_full"].abs() > 1e-10
        ]
        if changed.empty:
            continue
        first_round = int(changed["round"].min())
        first = changed[changed["round"] == first_round]
        for _, row in first.sort_values("cid").iterrows():
            first_rows.append(
                {
                    "attack": attack,
                    "first_divergence_round": first_round,
                    "cid": int(row["cid"]),
                    "is_malicious": bool(row["is_malicious_full"]),
                    "attack_active": bool(row["attack_active_full"]),
                    "full_weight": float(row["aggregation_weight_full"]),
                    "observe_weight": float(row["aggregation_weight_observe"]),
                    "observe_minus_full": float(
                        row["aggregation_weight_delta_observe_minus_full"]
                    ),
                    "full_semantic_risk": float(row["semantic_risk_full"]),
                    "full_semantic_q": float(row["semantic_q_full"]),
                    "full_cumulative_q": float(row["rtc_v3_cumulative_q_full_full"]),
                    "full_direction_q": float(row["rtc_v3_direction_q_full_full"]),
                    "full_state": str(row["state_full"]),
                }
            )

    pd.concat(paired_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "paired_client_weights.csv", index=False
    )
    first_divergence = pd.DataFrame(first_rows)
    first_divergence.to_csv(OUTPUT_DIR / "first_weight_divergence.csv", index=False)

    dba_rounds = rounds[rounds["attack"] == "dba"].copy()
    dba_timeline = dba_rounds[
        [
            "round",
            "mode",
            "server_accuracy",
            "server_dba_full_trigger_asr",
            "fit_malicious_aggregation_weight_share",
            "fit_malicious_impact_share",
            "fit_rtc_v3_semantic_exposure",
            "fit_rtc_v3_weight_sum",
        ]
    ].sort_values(["round", "mode"])
    dba_timeline.to_csv(OUTPUT_DIR / "dba_timeline.csv", index=False)

    summary = {
        "source": str(LOG_DIR),
        "quality_gates_passed": bool(
            pd.read_csv(LOG_DIR / "quality_gates.csv")["passed"].astype(bool).all()
        ),
        "run_summary": run_summary.replace({np.nan: None}).to_dict("records"),
        "first_weight_divergence": first_divergence.replace({np.nan: None}).to_dict(
            "records"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
