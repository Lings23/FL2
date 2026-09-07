"""Cross-attack diagnosis for RTC V3 strong seed-42 Byzantine runs.

All cross-attack comparisons, clean counterfactuals, and client-level mechanism
evidence come from the corrected strong formal run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
FORMAL = ROOT / "logs" / "rtc_v3_byzantine_strong_seed42_mf03"


def auc_from_scores(labels: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame({"label": labels.astype(bool), "score": scores}).dropna()
    positives = int(frame["label"].sum())
    negatives = int((~frame["label"]).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = frame["score"].rank(method="average")
    rank_sum = float(ranks[frame["label"]].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    if series.dtype == object:
        series = series.astype(str).str.lower().map({"true": True, "false": False})
    return float(pd.to_numeric(series, errors="coerce").mean())


def client_metrics(run: pd.Series) -> tuple[dict, list[dict]]:
    path = FORMAL / "raw" / f"{run['run_id']}_clients.csv"
    clients = pd.read_csv(path)
    active = clients[clients["round"] >= 11].copy()
    if len(active) != 500 or active[["round", "cid"]].duplicated().any():
        raise ValueError(f"unexpected client grain for {run['run_id']}: {len(active)}")

    by_round = active.groupby("round", sort=True)
    nominal_attacker = by_round["is_malicious"].mean()
    attacker_mass = by_round.apply(
        lambda frame: frame.loc[frame["is_malicious"].astype(bool), "aggregation_weight"].sum(),
        include_groups=False,
    )
    benign_mass = by_round.apply(
        lambda frame: frame.loc[~frame["is_malicious"].astype(bool), "aggregation_weight"].sum(),
        include_groups=False,
    )
    total_mass = by_round["aggregation_weight"].sum()
    benign = active[~active["is_malicious"].astype(bool)]
    malicious = active[active["is_malicious"].astype(bool)]

    result = {
        "attack": run["attack"],
        "defense": run["defense"],
        "active_client_rows": len(active),
        "malicious_observations": len(malicious),
        "benign_observations": len(benign),
        "nominal_attacker_share": float(nominal_attacker.mean()),
        "actual_attacker_mass": float(attacker_mass.mean()),
        "attacker_mass_retention": float(attacker_mass.mean() / nominal_attacker.mean()),
        "actual_benign_mass": float(benign_mass.mean()),
        "benign_mass_loss_vs_nominal": float((1.0 - nominal_attacker.mean()) - benign_mass.mean()),
        "zero_update_mass": float((1.0 - total_mass).mean()),
        "rounds_with_zero_mass": float(((1.0 - total_mass) > 1e-9).mean()),
        "benign_clip_rate": rate(benign["clipped"]),
        "malicious_clip_rate": rate(malicious["clipped"]),
        "benign_cap_rate": rate(benign["capped"]),
        "malicious_cap_rate": rate(malicious["capped"]),
        "benign_non_normal_rate": float((benign["state"].astype(str) != "normal").mean()),
        "malicious_non_normal_rate": float((malicious["state"].astype(str) != "normal").mean()),
        "benign_quarantine_rate_client": float((benign["state"].astype(str) == "quarantined").mean()),
        "malicious_quarantine_rate_client": float((malicious["state"].astype(str) == "quarantined").mean()),
        "benign_mean_weight": float(benign["aggregation_weight"].mean()),
        "malicious_mean_weight": float(malicious["aggregation_weight"].mean()),
    }

    signals = {
        "semantic_total_risk": ("total_risk", True),
        "semantic_risk": ("semantic_risk", True),
        "semantic_z": ("semantic_z", True),
        "raw_update_norm": ("raw_delta_norm", True),
        "residual_norm": ("rtc_v3_residual_norm", True),
        "one_minus_cumulative_q": ("rtc_v3_cumulative_q_full", False),
        "one_minus_trust": ("trust", False),
    }
    signal_rows: list[dict] = []
    for signal, (column, direct) in signals.items():
        if column not in active.columns:
            continue
        values = pd.to_numeric(active[column], errors="coerce")
        suspiciousness = values if direct else 1.0 - values
        signal_rows.append(
            {
                "attack": run["attack"],
                "signal": signal,
                "malicious_detection_auc": auc_from_scores(active["is_malicious"], suspiciousness),
                "malicious_mean": float(values[active["is_malicious"].astype(bool)].mean()),
                "benign_mean": float(values[~active["is_malicious"].astype(bool)].mean()),
            }
        )
    return result, signal_rows


def active_accuracy_from_rounds(run_id: str) -> float:
    rounds = pd.read_csv(FORMAL / "rounds" / f"{run_id}.csv")
    active_window = rounds[(rounds["round"] >= 11) & (rounds["round"] <= 60)]
    if len(active_window) != 50:
        raise ValueError(f"unexpected active-window row count for {run_id}: {len(active_window)}")
    return float(active_window["server_accuracy"].mean())


def classify(row: pd.Series) -> str:
    if row["attack"] in {"dba", "scaling_backdoor"}:
        return "有效压制恶意，但以零质量/轻微良性动作换取安全"
    if row["rtc_acc_gap_to_best_pp"] >= -0.3:
        return "RTC 已接近或达到最佳；不存在系统性误伤证据"
    if row["attacker_mass_retention"] >= 0.85 and row["benign_non_normal_rate"] >= 0.05:
        return "双重失效：恶意权重未压低，同时损失良性质量"
    if row["attacker_mass_retention"] >= 0.85:
        return "主要是恶意权重保留，不是良性误伤"
    return "混合原因，需结合安全指标与零质量判断"


def main() -> None:
    formal = pd.read_csv(FORMAL / "periodic_attack_runs.csv")

    if formal["run_id"].duplicated().any():
        raise ValueError("formal comparison contains duplicate run_id values")
    if not formal["valid"].astype(bool).all():
        raise ValueError("formal comparison contains invalid runs")
    if set(formal["seed"].unique()) != {42}:
        raise ValueError("formal comparison is expected to contain seed 42 only")
    attacked = formal[formal["attack"] != "none"]
    if set(attacked["malicious_fraction"].unique()) != {0.3}:
        raise ValueError("formal attacked runs are expected to use malicious_fraction=0.3")
    if set(attacked["attack_start_round"].unique()) != {11}:
        raise ValueError("formal attacked runs are expected to start at round 11")

    for run_id in formal["run_id"]:
        round_path = FORMAL / "rounds" / f"{run_id}.csv"
        if not round_path.exists():
            raise FileNotFoundError(round_path)
        rounds = pd.read_csv(round_path, usecols=["round"])
        if rounds["round"].tolist() != list(range(61)):
            raise ValueError(f"unexpected round coverage for {run_id}")

    complete_attacks = []
    expected_defenses = {"rtc_full", "fedavg", "krum", "multi_krum", "median"}
    for attack, frame in formal[formal["attack"] != "none"].groupby("attack"):
        if set(frame["defense"]) == expected_defenses:
            complete_attacks.append(attack)
    complete_attacks = sorted(complete_attacks)

    attack_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    signal_rows: list[dict] = []
    for attack in complete_attacks:
        frame = formal[formal["attack"] == attack].copy()
        rtc = frame[frame["defense"] == "rtc_full"].iloc[0]
        mk = frame[frame["defense"] == "multi_krum"].iloc[0]
        fedavg = frame[frame["defense"] == "fedavg"].iloc[0]
        clean_match = formal[
            (formal["attack"] == "none")
            & (formal["counterfactual_for"] == rtc["pairing_group_id"])
        ]
        if len(clean_match) != 1:
            raise ValueError(f"expected one clean counterfactual for {attack}, found {len(clean_match)}")
        clean_active_accuracy = active_accuracy_from_rounds(str(clean_match.iloc[0]["run_id"]))
        best = frame.loc[frame["active_accuracy"].idxmax()]
        targeted = attack in {"dba", "scaling_backdoor"}
        robust = frame[frame["active_asr"].fillna(np.inf) <= 0.10] if targeted else frame
        best_balanced = robust.loc[robust["active_accuracy"].idxmax()] if not robust.empty else None

        rtc_mechanism, rtc_signals = client_metrics(rtc)
        mk_mechanism, _ = client_metrics(mk)
        mechanism_rows.extend([rtc_mechanism, mk_mechanism])
        signal_rows.extend(rtc_signals)

        attack_rows.append(
            {
                "attack": attack,
                "targeted": targeted,
                "rtc_active_accuracy": float(rtc["active_accuracy"]),
                "multi_krum_active_accuracy": float(mk["active_accuracy"]),
                "fedavg_active_accuracy": float(fedavg["active_accuracy"]),
                "clean_fedavg_active_accuracy": clean_active_accuracy,
                "formal_fedavg_attack_drop_pp": 100.0 * float(clean_active_accuracy - fedavg["active_accuracy"]),
                "best_acc_defense": str(best["defense"]),
                "best_active_accuracy": float(best["active_accuracy"]),
                "rtc_acc_gap_to_best_pp": 100.0 * float(rtc["active_accuracy"] - best["active_accuracy"]),
                "rtc_acc_gap_to_multi_krum_pp": 100.0 * float(rtc["active_accuracy"] - mk["active_accuracy"]),
                "rtc_acc_gap_to_fedavg_pp": 100.0 * float(rtc["active_accuracy"] - fedavg["active_accuracy"]),
                "rtc_active_asr": None if pd.isna(rtc["active_asr"]) else float(rtc["active_asr"]),
                "multi_krum_active_asr": None if pd.isna(mk["active_asr"]) else float(mk["active_asr"]),
                "best_balanced_defense": None if best_balanced is None else str(best_balanced["defense"]),
                "best_balanced_accuracy": None if best_balanced is None else float(best_balanced["active_accuracy"]),
                "rtc_balanced_gap_pp": None if best_balanced is None else 100.0 * float(rtc["active_accuracy"] - best_balanced["active_accuracy"]),
                "nominal_attacker_share": rtc_mechanism["nominal_attacker_share"],
                "rtc_attacker_mass": rtc_mechanism["actual_attacker_mass"],
                "multi_krum_attacker_mass": mk_mechanism["actual_attacker_mass"],
                "attacker_mass_retention": rtc_mechanism["attacker_mass_retention"],
                "benign_mass_loss_vs_nominal": rtc_mechanism["benign_mass_loss_vs_nominal"],
                "zero_update_mass": rtc_mechanism["zero_update_mass"],
                "benign_non_normal_rate": rtc_mechanism["benign_non_normal_rate"],
                "malicious_non_normal_rate": rtc_mechanism["malicious_non_normal_rate"],
                "benign_cap_rate": rtc_mechanism["benign_cap_rate"],
                "malicious_cap_rate": rtc_mechanism["malicious_cap_rate"],
                "benign_clip_rate": rtc_mechanism["benign_clip_rate"],
                "malicious_clip_rate": rtc_mechanism["malicious_clip_rate"],
            }
        )

    attacks = pd.DataFrame(attack_rows)
    attacks["diagnosis"] = attacks.apply(classify, axis=1)
    mechanisms = pd.DataFrame(mechanism_rows)
    signals = pd.DataFrame(signal_rows)

    sign_flip = formal[(formal["attack"] == "sign_flip") & (formal["defense"] == "rtc_full")]
    sign_flip_note = None
    if not sign_flip.empty:
        sign_mechanism, sign_signals = client_metrics(sign_flip.iloc[0])
        sign_flip_note = sign_mechanism
        mechanisms = pd.concat([mechanisms, pd.DataFrame([sign_mechanism])], ignore_index=True)
        signals = pd.concat([signals, pd.DataFrame(sign_signals)], ignore_index=True)

    attacks.to_csv(OUTPUT / "cross_attack_summary.csv", index=False)
    mechanisms.to_csv(OUTPUT / "cross_attack_rtc_mechanism.csv", index=False)
    signals.to_csv(OUTPUT / "cross_attack_signal_auc.csv", index=False)

    result = {
        "data_quality": {
            "formal_rows": int(len(formal)),
            "formal_valid_rows": int(formal["valid"].astype(bool).sum()),
            "unique_run_ids": int(formal["run_id"].nunique()),
            "clean_counterfactual_rows": int((formal["attack"] == "none").sum()),
            "complete_attacks": complete_attacks,
            "complete_defenses_per_attack": sorted(expected_defenses),
            "seed": int(formal["seed"].iloc[0]),
            "formal_malicious_fraction": float(formal.loc[formal["attack"] != "none", "malicious_fraction"].iloc[0]),
            "attack_start_round": int(formal.loc[formal["attack"] != "none", "attack_start_round"].iloc[0]),
            "round_rows_per_run": 61,
        },
        "primary_answer": "No: benign-quality loss is a cross-attack utility tax, but it is not the universal primary cause.",
        "attack_diagnoses": attacks.set_index("attack")["diagnosis"].to_dict(),
        "sign_flip_comparator_available": False,
        "sign_flip_rtc_mechanism": sign_flip_note,
    }
    (OUTPUT / "cross_attack_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(attacks.to_string(index=False))
    print(json.dumps(result["data_quality"], ensure_ascii=False))


if __name__ == "__main__":
    main()
