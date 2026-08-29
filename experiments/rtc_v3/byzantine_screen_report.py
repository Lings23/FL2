"""Render the FedAvg-only Byzantine strength screen, including failed gates."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib-cache").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from experiments.rtc_v3.byzantine import CANONICAL_ATTACKS
from experiments.rtc_v3.byzantine_analysis import ATTACK_LABELS, TARGETED, _truth


LEVELS = ("weak", "medium", "strong")
COLORS = {"weak": "#B8C7D9", "medium": "#4F759B", "strong": "#D89000"}


def _preferred_input(root: Path, reconstructed: str, original: str) -> Path:
    reconstructed_path = root / reconstructed
    return reconstructed_path if reconstructed_path.exists() else root / original


def load_screen(
    root: Path,
    runs_path: Path | None = None,
    recommendations_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_path = runs_path or _preferred_input(
        root, "byzantine_attack_runs_reconstructed.csv", "byzantine_attack_runs.csv"
    )
    recommendations_path = recommendations_path or _preferred_input(
        root,
        "attack_strength_screening_reconstructed.csv",
        "attack_strength_screening.csv",
    )
    runs = pd.read_csv(runs_path)
    gates = pd.read_csv(_preferred_input(
        root, "quality_gates_reconstructed.csv", "quality_gates.csv"
    ))
    recommendations = pd.read_csv(recommendations_path)
    invalid = gates.loc[
        ~gates["passed"].map(_truth)
        & ~gates["gate"].astype(str).str.startswith("attack_strength:"),
        "gate",
    ].astype(str).tolist()
    if invalid:
        raise ValueError(f"non-strength screening gates failed: {invalid}")
    attacked = runs.loc[runs["attack"].astype(str) != "none"].copy()
    if set(attacked["defense"].astype(str)) != {"fedavg"}:
        raise ValueError("strength screening must be FedAvg-only")
    per_attack_plans = attacked.groupby("attack")["trial_plan_hash"].nunique()
    if not bool((per_attack_plans == 1).all()):
        raise ValueError("strength levels do not share one TrialPlan per attack")
    return attacked, recommendations


def plot_screen(attacked: pd.DataFrame, recommendations: pd.DataFrame, output: Path) -> Path:
    attacks = [
        attack
        for attack in CANONICAL_ATTACKS
        if attack in set(attacked["attack"].astype(str))
    ]
    if not attacks:
        raise ValueError("strength screening contains no canonical attacks")
    columns = min(5, len(attacks))
    rows_count = int(math.ceil(len(attacks) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(3.1 * columns, 3.5 * rows_count),
        sharey=True,
        squeeze=False,
    )
    lookup = recommendations.set_index("attack")
    plotted_values: list[float] = []
    for attack in attacks:
        metric = str(lookup.loc[attack, "metric"]) if (
            attack in lookup.index and "metric" in lookup.columns
        ) else (
            "participating_asr" if attack in TARGETED else "accuracy_drop"
        )
        values = pd.to_numeric(
            attacked.loc[attacked["attack"].astype(str) == attack, metric],
            errors="coerce",
        ).dropna()
        plotted_values.extend(values.astype(float).tolist())
    y_min = min(-0.05, min(plotted_values, default=0.0) - 0.03)
    y_max = max(0.5, max(plotted_values, default=0.0) + 0.08)
    for ax, attack in zip(axes.flat, attacks):
        rows = attacked.loc[attacked["attack"].astype(str) == attack].set_index("strength_level")
        metric = str(lookup.loc[attack, "metric"]) if (
            attack in lookup.index and "metric" in lookup.columns
        ) else (
            "participating_asr" if attack in TARGETED else "accuracy_drop"
        )
        values = [float(pd.to_numeric(rows.loc[level, metric], errors="coerce")) for level in LEVELS]
        bars = ax.bar(range(3), values, color=[COLORS[level] for level in LEVELS],
                      edgecolor="#343A40", linewidth=0.6)
        statuses = {
            level: str(lookup.loc[attack, f"{level}_status"])
            if attack in lookup.index and f"{level}_status" in lookup.columns
            else "VALID"
            for level in LEVELS
        }
        for level, bar in zip(LEVELS, bars):
            if statuses[level] == "COLLAPSED/INVALID":
                bar.set_hatch("///")
                bar.set_alpha(0.35)
        threshold = float(pd.to_numeric(
            lookup.loc[attack, "threshold"]
            if attack in lookup.index and "threshold" in lookup.columns
            else math.nan,
            errors="coerce",
        ))
        if np.isfinite(threshold):
            ax.axhline(threshold, color="#343A40", linestyle=":", linewidth=1.1)
        selected = str(lookup.loc[attack, "selected_strength"]) if attack in lookup.index else ""
        if selected in LEVELS:
            index = LEVELS.index(selected)
            bars[index].set_linewidth(2.2)
            bars[index].set_edgecolor("#111111")
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.1%}",
                        ha="center", va="bottom", fontsize=7)
        for index, level in enumerate(LEVELS):
            if statuses[level] == "COLLAPSED/INVALID":
                ax.text(index, y_min + 0.01, "INVALID", rotation=90,
                        ha="center", va="bottom", fontsize=6.5, color="#A33A2B")
        ax.set_xticks(range(3), [level.title() for level in LEVELS], rotation=20)
        ax.set_title(ATTACK_LABELS.get(attack, attack), fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        metric_label = {
            "participating_asr": "participating ASR",
            "accuracy_drop": "accuracy drop",
            "source_recall_drop": "source recall drop",
        }.get(metric, metric.replace("_", " "))
        ax.text(0.02, 0.98, metric_label,
                transform=ax.transAxes, va="top", fontsize=7.5, color="#5F6368")
    for ax in list(axes.flat)[len(attacks):]:
        ax.set_visible(False)
    fig.suptitle("FedAvg-only attack strength screening — seed 42, 25 rounds", fontsize=15)
    fig.text(0.5, 0.015,
             "Bars are discrete preregistered strengths; dotted lines are attack-specific gates; hatched/faded bars are collapsed or invalid; bold outline is frozen.",
             ha="center", fontsize=8, color="#5F6368")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = output / "attack_strength_screening.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def run(
    root: Path,
    output: Path | None = None,
    runs_path: Path | None = None,
    recommendations_path: Path | None = None,
) -> Path:
    root = root.resolve()
    target = (output or root / "plots").resolve()
    target.mkdir(parents=True, exist_ok=True)
    attacked, recommendations = load_screen(
        root, runs_path=runs_path, recommendations_path=recommendations_path
    )
    recommendations.to_csv(target / "attack_strength_screening_table.csv", index=False)
    return plot_screen(attacked, recommendations, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/rtc_v3_byzantine_screen")
    parser.add_argument("--output", default="")
    parser.add_argument("--runs", default="", help="Optional result-matrix CSV")
    parser.add_argument(
        "--recommendations", default="", help="Optional strength-screening CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(
        Path(args.root),
        Path(args.output) if args.output else None,
        Path(args.runs) if args.runs else None,
        Path(args.recommendations) if args.recommendations else None,
    ))
