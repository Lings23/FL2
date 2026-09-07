"""Plot the completed LIE z=0.5 rerun with its paired clean baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


DEFENSES = ("fedavg", "rtc_full", "krum", "multi_krum", "median")
LABELS = {
    "clean": "Paired clean",
    "fedavg": "No defense",
    "rtc_full": "RTC",
    "krum": "Krum",
    "multi_krum": "Multi-Krum",
    "median": "Median",
}
COLORS = {
    "clean": "#374151",
    "fedavg": "#6B7280",
    "rtc_full": "#2563A6",
    "krum": "#D08A2E",
    "multi_krum": "#8A7A36",
    "median": "#B25D79",
}
STYLES = {
    "clean": (0, (7, 2)),
    "fedavg": "--",
    "rtc_full": "-",
    "krum": "-.",
    "multi_krum": ":",
    "median": (0, (5, 2)),
}


def status(root: Path, run_id: str) -> dict[str, Any]:
    path = root / "status" / f"{run_id}.json"
    if not path.is_file():
        raise ValueError(f"missing status: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def rounds(root: Path, run_id: str, expected_rounds: int) -> pd.DataFrame:
    path = root / "rounds" / f"{run_id}.csv"
    if not path.is_file():
        raise ValueError(f"missing round file: {run_id}")
    frame = pd.read_csv(path)
    observed = pd.to_numeric(frame["round"], errors="raise").astype(int).tolist()
    expected = list(range(expected_rounds + 1))
    if observed != expected:
        raise ValueError(f"round sequence mismatch: {run_id}")
    accuracy = pd.to_numeric(frame["server_accuracy"], errors="coerce")
    if not np.isfinite(accuracy).all() or not accuracy.between(0.0, 1.0).all():
        raise ValueError(f"invalid server accuracy: {run_id}")
    return frame


def load(root: Path, expected_rounds: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    runs = pd.read_csv(root / "periodic_attack_runs.csv")
    attacked = runs[(runs["attack"].astype(str) == "lie") & runs["defense"].astype(str).isin(DEFENSES)].copy()
    if len(attacked) != len(DEFENSES) or set(attacked["defense"].astype(str)) != set(DEFENSES):
        raise ValueError("the five LIE defense cells are not complete")
    if attacked["trial_plan_hash"].astype(str).nunique() != 1:
        raise ValueError("LIE defenses do not share one TrialPlan")
    if set(pd.to_numeric(attacked["lie_z"], errors="raise")) != {0.5}:
        raise ValueError("expected lie_z=0.5")
    if set(pd.to_numeric(attacked["malicious_fraction"], errors="raise")) != {0.2}:
        raise ValueError("expected the executed malicious_fraction=0.2")

    plan = str(attacked.iloc[0]["trial_plan_hash"])
    clean = runs[
        (runs["attack"].astype(str) == "none")
        & (runs["defense"].astype(str) == "fedavg")
        & (runs["trial_plan_hash"].astype(str) == plan)
    ]
    if len(clean) != 1:
        raise ValueError("expected exactly one plan-matched clean baseline")

    frames: dict[str, pd.DataFrame] = {}
    for _, spec in pd.concat([attacked, clean]).iterrows():
        run_id = str(spec["run_id"])
        run_status = status(root, run_id)
        if run_status.get("state") != "completed" or int(run_status.get("last_round", -1)) != expected_rounds:
            raise ValueError(f"incomplete run: {run_id}")
        frame = rounds(root, run_id, expected_rounds)
        if str(spec["attack"]) == "lie":
            active = pd.to_numeric(frame["planned_attack_active"], errors="raise").astype(int)
            expected_active = (pd.to_numeric(frame["round"]) >= 11).astype(int)
            if not active.equals(expected_active):
                raise ValueError(f"attack schedule mismatch: {run_id}")
        frames[run_id] = frame
    return attacked, frames, clean.iloc[0]


def summarize(attacked: pd.DataFrame, frames: dict[str, pd.DataFrame], clean_spec: pd.Series) -> pd.DataFrame:
    clean = frames[str(clean_spec["run_id"])].copy()
    clean_active = clean[pd.to_numeric(clean["round"]) >= 11]
    clean_accuracy = pd.to_numeric(clean_active["server_accuracy"], errors="raise")
    rows: list[dict[str, Any]] = []
    for defense in DEFENSES:
        spec = attacked[attacked["defense"].astype(str) == defense].iloc[0]
        frame = frames[str(spec["run_id"])]
        active = frame[pd.to_numeric(frame["round"]) >= 11]
        attacked_accuracy = pd.to_numeric(active["server_accuracy"], errors="raise").reset_index(drop=True)
        clean_aligned = clean_accuracy.reset_index(drop=True)
        result: dict[str, Any] = {
            "defense": defense,
            "defense_label": LABELS[defense],
            "active_accuracy": float(attacked_accuracy.mean()),
            "clean_active_accuracy": float(clean_aligned.mean()),
            "active_accuracy_drop": float((clean_aligned - attacked_accuracy).mean()),
            "final_accuracy": float(pd.to_numeric(frame["server_accuracy"]).iloc[-1]),
            "active_min_accuracy": float(attacked_accuracy.min()),
            "attacker_weight_share": np.nan,
            "clip_recall_active_attackers": np.nan,
            "nominal_f_violation_rate": np.nan,
        }
        if "fit_active_attacker_weight_share" in active:
            result["attacker_weight_share"] = float(
                pd.to_numeric(active["fit_active_attacker_weight_share"], errors="coerce").mean()
            )
        if "fit_clip_recall_active_attackers" in active:
            result["clip_recall_active_attackers"] = float(
                pd.to_numeric(active["fit_clip_recall_active_attackers"], errors="coerce").mean()
            )
        if defense in {"krum", "multi_krum"}:
            actual = pd.to_numeric(active["fit_selected_malicious_clients"], errors="coerce")
            result["nominal_f_violation_rate"] = float((actual > int(spec["krum_num_malicious"])).mean())
        rows.append(result)
    return pd.DataFrame(rows)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4B5563",
            "axes.facecolor": "#FCFCFD",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finish(fig: plt.Figure, path: Path, top: float = 0.89) -> None:
    fig.tight_layout(rect=(0.0, 0.02, 1.0, top))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(attacked: pd.DataFrame, frames: dict[str, pd.DataFrame], clean_spec: pd.Series, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    clean = frames[str(clean_spec["run_id"])]
    ax.plot(
        pd.to_numeric(clean["round"]), pd.to_numeric(clean["server_accuracy"]),
        color=COLORS["clean"], linestyle=STYLES["clean"], linewidth=2.0, label=LABELS["clean"],
    )
    for defense in DEFENSES:
        spec = attacked[attacked["defense"].astype(str) == defense].iloc[0]
        frame = frames[str(spec["run_id"])]
        ax.plot(
            pd.to_numeric(frame["round"]), pd.to_numeric(frame["server_accuracy"]),
            color=COLORS[defense], linestyle=STYLES[defense], linewidth=1.8, label=LABELS[defense],
        )
    ax.axvline(11, color="#111827", linestyle=":", linewidth=1.2)
    ax.text(11.6, 0.04, "Attack starts", color="#374151", fontsize=9)
    ax.set(xlim=(0, 60), ylim=(0, 1), xlabel="Server round", ylabel="Main-task accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=6, frameon=False)
    fig.suptitle("LIE z=0.5: main-task accuracy trajectories", x=0.065, y=0.995, ha="left", fontsize=16)
    fig.text(0.065, 0.945, "Seed 42; malicious fraction 0.2; IID; attack active in rounds 11–60.", color="#5F6368")
    path = output / "accuracy_timelines.png"
    finish(fig, path, top=0.89)
    return path


def plot_comparison(summary: pd.DataFrame, output: Path) -> Path:
    ordered = summary.sort_values("active_accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.barh(
        ordered["defense_label"], ordered["active_accuracy"],
        color=[COLORS[value] for value in ordered["defense"]], edgecolor="#374151", linewidth=0.7,
    )
    clean = float(ordered["clean_active_accuracy"].iloc[0])
    ax.axvline(clean, color=COLORS["clean"], linestyle="--", linewidth=1.5, label=f"Paired clean: {clean:.1%}")
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in ordered["active_accuracy"]], padding=4)
    ax.set(xlim=(0, 1), xlabel="Mean accuracy, rounds 11–60", ylabel="Defense")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Active-round main-task accuracy", x=0.08, y=0.995, ha="left", fontsize=16)
    fig.text(0.08, 0.945, "Higher is better; dashed reference is the plan-matched clean FedAvg trajectory.", color="#5F6368")
    path = output / "active_accuracy_comparison.png"
    finish(fig, path)
    return path


def plot_drop(summary: pd.DataFrame, output: Path) -> Path:
    ordered = summary.sort_values("active_accuracy_drop", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.barh(
        ordered["defense_label"], ordered["active_accuracy_drop"],
        color=[COLORS[value] for value in ordered["defense"]], edgecolor="#374151", linewidth=0.7,
    )
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in ordered["active_accuracy_drop"]], padding=4)
    upper = max(0.05, float(ordered["active_accuracy_drop"].max()) * 1.18)
    ax.set(xlim=(0, upper), xlabel="Paired clean accuracy minus attacked accuracy", ylabel="Defense")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Active-round accuracy drop", x=0.08, y=0.995, ha="left", fontsize=16)
    fig.text(0.08, 0.945, "Mean paired difference over rounds 11–60; lower is better.", color="#5F6368")
    path = output / "active_accuracy_drop.png"
    finish(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "lie_z05_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    attacked, frames, clean_spec = load(root, args.rounds)
    summary = summarize(attacked, frames, clean_spec)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "source_root": str(root),
        "attack": "lie",
        "lie_z": 0.5,
        "malicious_fraction": 0.2,
        "seed": 42,
        "attack_start_round": 11,
        "rounds": args.rounds,
        "completed_attacked_cells": len(attacked),
        "paired_clean_complete": True,
        "trial_plan_hash": str(attacked.iloc[0]["trial_plan_hash"]),
        "interpretation": "single-seed descriptive result; not comparable to m=0.3 as a z-only ablation",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    style()
    paths = [
        plot_timeline(attacked, frames, clean_spec, output),
        plot_comparison(summary, output),
        plot_drop(summary, output),
    ]
    pd.DataFrame(
        [
            {"artifact": paths[0].name, "question": "How do clean and attacked accuracy trajectories evolve over 60 rounds?"},
            {"artifact": paths[1].name, "question": "Which defense preserves the highest mean active-round accuracy?"},
            {"artifact": paths[2].name, "question": "How large is each defense's paired active-round accuracy loss?"},
        ]
    ).to_csv(output / "chart_map.csv", index=False)
    print(json.dumps({**metadata, "output": str(output), "plots": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
