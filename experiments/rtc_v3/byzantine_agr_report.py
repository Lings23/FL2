"""Validate and summarize robust-AGR reproduction runs separately from RTC."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rtc_v3.byzantine import (
    ROBUST_AGR_REPRODUCTION_ATTACKS,
    ROBUST_AGR_REPRODUCTION_DEFENSES,
)


def summarize(root: Path, *, allow_partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = root / "byzantine_attack_runs.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    runs = pd.read_csv(source)
    selected = runs[
        runs["attack"].astype(str).isin(ROBUST_AGR_REPRODUCTION_ATTACKS)
        & runs["defense"].astype(str).isin(ROBUST_AGR_REPRODUCTION_DEFENSES)
    ].copy()
    if selected.empty:
        raise ValueError("no robust-AGR reproduction rows found")

    expected = {
        (attack, defense)
        for attack in ROBUST_AGR_REPRODUCTION_ATTACKS
        for defense in ROBUST_AGR_REPRODUCTION_DEFENSES
    }
    observed = set(zip(selected["attack"].astype(str), selected["defense"].astype(str)))
    missing = sorted(expected - observed)
    if missing and not allow_partial:
        raise ValueError(f"robust-AGR reproduction matrix is incomplete: {missing}")

    divergence = selected.get(
        "numerical_divergence", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    collapsed = selected.get(
        "collapsed_invalid", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    random_guess = selected.get(
        "model_random_guess", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    selected["numerically_valid"] = ~(divergence | collapsed | random_guess)
    selected["reproduction_status"] = np.where(
        selected["numerically_valid"], "VALID", "COLLAPSED/INVALID"
    )

    numeric_metrics = [
        column
        for column in (
            "final_accuracy",
            "active_accuracy",
            "accuracy_drop",
            "active_accuracy_drop",
            "nan_rounds",
        )
        if column in selected
    ]
    aggregate = (
        selected.groupby(["attack", "defense"], as_index=False)[numeric_metrics]
        .mean(numeric_only=True)
    )
    validity = (
        selected.groupby(["attack", "defense"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            valid_runs=("numerically_valid", "sum"),
        )
    )
    aggregate = aggregate.merge(validity, on=["attack", "defense"], how="outer")
    aggregate["all_runs_valid"] = aggregate["valid_runs"] == aggregate["runs"]
    effect_metric = (
        "active_accuracy_drop"
        if "active_accuracy_drop" in aggregate
        else "accuracy_drop"
    )
    if effect_metric in aggregate:
        lie = (
            aggregate[aggregate["attack"] == "lie"]
            .set_index("defense")[effect_metric]
        )
        aggregate["effect_metric"] = effect_metric
        aggregate["effect_delta_vs_lie"] = aggregate.apply(
            lambda row: (
                float(row[effect_metric]) - float(lie.get(row["defense"], np.nan))
                if row["attack"] != "lie"
                else 0.0
            ),
            axis=1,
        )
        aggregate["more_effective_than_lie"] = (
            (aggregate["attack"] != "lie")
            & aggregate["all_runs_valid"]
            & (aggregate["effect_delta_vs_lie"] > 0.0)
        )
    return selected, aggregate


def optimization_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Report whether each optimization attack beats LIE on most robust AGRs."""
    rows = []
    for attack in ("min_max", "min_sum"):
        candidates = aggregate[aggregate["attack"] == attack]
        comparable = candidates[
            candidates["effect_delta_vs_lie"].map(np.isfinite)
            & candidates["all_runs_valid"].fillna(False).astype(bool)
        ]
        wins = int(comparable["more_effective_than_lie"].sum())
        total = int(len(comparable))
        rows.append({
            "attack": attack,
            "robust_agrs_compared": total,
            "wins_vs_lie": wins,
            "win_rate_vs_lie": float(wins / total) if total else np.nan,
            "majority_more_effective_than_lie": bool(total > 0 and wins > total / 2),
            "validation_status": (
                "SUPPORTED"
                if total > 0 and wins > total / 2
                else "NOT_SUPPORTED"
                if total > 0
                else "INSUFFICIENT_VALID_RUNS"
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/rtc_v3_byzantine_agr")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs, aggregate = summarize(root, allow_partial=bool(args.allow_partial))
    comparison = optimization_comparison(aggregate)
    runs_path = root / "robust_agr_reproduction_runs.csv"
    aggregate_path = root / "robust_agr_reproduction_summary.csv"
    comparison_path = root / "robust_agr_optimization_vs_lie.csv"
    runs.to_csv(runs_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    print(runs_path)
    print(aggregate_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
