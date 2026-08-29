"""Publication-ready plots for the completed RTC-V3 semantic experiments.

The script is intentionally fail-closed: it only plots complete round files and
requires all strict execution gates in the candidate attack root to pass.  It
does not consume incomplete clean/recovery runs or the cancelled ablations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path.cwd() / ".matplotlib-cache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ATTACK_LABELS = {
    "label_flip_targeted": "Targeted label flip (5→3)",
    "label_flip_all_reverse": "All-reverse label flip",
}
DEFENSE_LABELS = {
    "rtc_v2": "Promoted RTC-V3 V2",
    "clip_only": "Clip-only",
    "rtc_v3": "Semantic–temporal–exposure",
}
COLORS = {
    "rtc_v2": "#8A8F98",
    "clip_only": "#D89000",
    "rtc_v3": "#1769AA",
}
LINESTYLES = {"rtc_v2": ":", "clip_only": "--", "rtc_v3": "-"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _require_strict_gates(root: Path) -> None:
    gates = _read_csv(root / "execution_validation.csv")
    if gates.empty or "passed" not in gates:
        raise ValueError("execution_validation.csv has no gates")
    passed = gates["passed"].astype(str).str.lower().map({"true": True, "false": False})
    if passed.isna().any() or not bool(passed.all()):
        failed = gates.loc[~passed.fillna(False), "gate"].astype(str).tolist()
        raise ValueError(f"strict execution gates failed or malformed: {failed}")


def _round_file(root: Path, attack: str, defense: str) -> Path:
    matches = sorted((root / "rounds").glob(f"*{attack}*__{defense}__*.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one round file for {attack}/{defense} in {root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_rounds(root: Path, attack: str, defense: str) -> pd.DataFrame:
    frame = _read_csv(_round_file(root, attack, defense)).copy()
    required = {"round", "server_asr", "server_accuracy", "planned_attack_active"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"round file is missing columns: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"non-finite required round metrics in {attack}/{defense}")
    observed = set(frame.loc[frame["round"] > 0, "round"].astype(int))
    if observed != set(range(1, 61)):
        raise ValueError(f"incomplete rounds for {attack}/{defense}: {len(observed)}/60")
    return frame.sort_values("round").reset_index(drop=True)


def active_summary(frame: pd.DataFrame) -> dict[str, float]:
    active = frame.loc[
        (frame["round"] > 0) & (frame["planned_attack_active"] > 0.5)
    ]
    trained = frame.loc[frame["round"] > 0]
    if active.empty or trained.empty:
        raise ValueError("round data has no active attack or trained rounds")
    return {
        "active_mean_asr": float(active["server_asr"].mean()),
        "active_peak_asr": float(active["server_asr"].max()),
        "final_accuracy": float(trained.iloc[-1]["server_accuracy"]),
        "last10_accuracy": float(trained.tail(10)["server_accuracy"].mean()),
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#495057",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.008, text, fontsize=7.5, color="#5F6368", ha="left")


def plot_timelines(rounds: dict[tuple[str, str], pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), sharex=True)
    for row, attack in enumerate(ATTACK_LABELS):
        for defense in ("clip_only", "rtc_v3"):
            frame = rounds[(attack, defense)]
            label = DEFENSE_LABELS[defense]
            common = dict(
                color=COLORS[defense], linestyle=LINESTYLES[defense], linewidth=2.0,
                label=label,
            )
            axes[row, 0].plot(frame["round"], frame["server_asr"], **common)
            axes[row, 1].plot(frame["round"], frame["server_accuracy"], **common)
        for col in range(2):
            ax = axes[row, col]
            ax.axvspan(11, 60, color="#F4B942", alpha=0.09, linewidth=0)
            ax.axvline(11, color="#5F6368", linewidth=1.0, linestyle=":")
            ax.text(11.5, 0.96, "attack starts", transform=ax.get_xaxis_transform(),
                    fontsize=8, color="#5F6368", va="top")
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_title(f"{ATTACK_LABELS[attack]} — {'ASR' if col == 0 else 'main accuracy'}")
            ax.set_xlabel("Server round")
            ax.set_ylabel("Rate")
            ax.spines[["top", "right"]].set_visible(False)
        axes[row, 0].legend(loc="upper right", frameon=False)
    fig.suptitle("RTC-V3 attack trajectories", fontsize=15, fontweight="bold", y=0.995)
    _footer(fig, "Seed 42 · IID · 20 clients · participation 0.5 · malicious fraction 0.2 · strict paired TrialPlan")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return fig


def plot_headline(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4))
    metrics = [
        ("active_mean_asr", "Active mean ASR"),
        ("active_peak_asr", "Active peak ASR"),
        ("final_accuracy", "Final main-task accuracy"),
        ("last10_accuracy", "Last-10 main-task accuracy"),
    ]
    defense_order = ["rtc_v2", "clip_only", "rtc_v3"]
    width = 0.24
    x = np.arange(len(ATTACK_LABELS), dtype=float)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for index, defense in enumerate(defense_order):
            values = []
            for attack in ATTACK_LABELS:
                row = summary.loc[
                    (summary["attack"] == attack) & (summary["defense"] == defense)
                ]
                values.append(float(row.iloc[0][metric]) if len(row) else np.nan)
            positions = x + (index - 1) * width
            bars = ax.bar(
                positions, values, width, color=COLORS[defense],
                edgecolor="#343A40", linewidth=0.55, label=DEFENSE_LABELS[defense],
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                            f"{value:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, ["Targeted 5→3", "All-reverse"])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Security–utility comparison", fontsize=15, fontweight="bold", y=0.995)
    _footer(fig, "All ASR summaries use rounds 11–60 only; no confidence intervals are shown because held-out evidence is one seed.")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return fig


def plot_mechanisms(analysis: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    attacks = list(ATTACK_LABELS)
    names = ["Targeted 5→3", "All-reverse"]
    x = np.arange(2, dtype=float)
    width = 0.34
    for index, defense in enumerate(("clip_only", "rtc_v3")):
        rows = analysis.set_index(["attack", "defense"])
        values = [float(rows.loc[(attack, defense), "malicious_effective_weight"]) for attack in attacks]
        bars = axes[0].bar(x + (index - 0.5) * width, values, width,
                           color=COLORS[defense], edgecolor="#343A40", linewidth=0.55,
                           label=DEFENSE_LABELS[defense])
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, value + 0.006,
                         f"{value:.2%}", ha="center", fontsize=8)
    axes[0].set_title("Malicious effective aggregation weight")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 0.25)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=8)

    candidate = analysis.loc[analysis["defense"] == "rtc_v3"].set_index("attack")
    trigger_columns = [
        ("semantic_trigger_count", "Semantic"),
        ("cumulative_trigger_count", "Cumulative"),
        ("semantic_group_trigger_count_replayed", "Semantic group"),
        ("semantic_principal_trigger_count_replayed", "Semantic principal"),
    ]
    widths = 0.18
    for index, (column, label) in enumerate(trigger_columns):
        values = [float(candidate.loc[attack, column]) for attack in attacks]
        positions = x + (index - 1.5) * widths
        axes[1].bar(positions, values, widths, label=label,
                    color=["#1769AA", "#D89000", "#7A8F35", "#A65D8F"][index],
                    edgecolor="#343A40", linewidth=0.5)
    axes[1].set_title("Active constraint counts in the full closure")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("Count across 60 rounds")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Defense mechanism evidence", fontsize=15, fontweight="bold", y=0.995)
    _footer(fig, "Candidate semantic detection AUC: targeted 1.000, all-reverse 0.995; prototype contamination events: 0 in both runs.")
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    return fig


def plot_performance(performance: dict[str, object]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    means = [float(performance["baseline_defense_mean_seconds"]),
             float(performance["candidate_defense_mean_seconds"])]
    p95 = [float(performance["baseline_defense_p95_seconds"]),
           float(performance["candidate_defense_p95_seconds"])]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width/2, means, width, label="Mean", color="#1769AA",
                edgecolor="#343A40", linewidth=0.55)
    axes[0].bar(x + width/2, p95, width, label="P95", color="#D89000",
                edgecolor="#343A40", linewidth=0.55)
    axes[0].set_xticks(x, ["RTC-V2 baseline", "Semantic candidate"])
    axes[0].set_ylabel("Seconds per server round")
    axes[0].set_ylim(0, max(p95) * 1.22)
    axes[0].set_title("Total RTC defense time")
    axes[0].legend(frameon=False)
    for ix, value in enumerate(means):
        axes[0].text(ix - width/2, value + 0.08, f"{value:.3f}s", ha="center", fontsize=8)
    for ix, value in enumerate(p95):
        axes[0].text(ix + width/2, value + 0.08, f"{value:.3f}s", ha="center", fontsize=8)

    semantic_ms = [1000 * float(performance["semantic_mean_seconds"]),
                   1000 * float(performance["semantic_p95_seconds"])]
    bars = axes[1].bar(["Mean", "P95"], semantic_ms, color=["#1769AA", "#D89000"],
                       edgecolor="#343A40", linewidth=0.55)
    axes[1].axhline(150, color="#5F6368", linestyle="--", linewidth=1, label="Mean gate 150 ms")
    axes[1].axhline(250, color="#5F6368", linestyle=":", linewidth=1, label="P95 gate 250 ms")
    axes[1].set_ylim(0, 275)
    axes[1].set_ylabel("Milliseconds per server round")
    axes[1].set_title("Incremental semantic pipeline time")
    axes[1].legend(frameon=False, fontsize=8)
    for bar, value in zip(bars, semantic_ms):
        axes[1].text(bar.get_x() + bar.get_width()/2, value + 5, f"{value:.2f} ms",
                     ha="center", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Server-side performance", fontsize=15, fontweight="bold", y=0.995)
    _footer(fig, "Paired 3-round GPU performance run, seed 43. Short-window result; not evidence of guaranteed speedup.")
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    return fig


def _save(fig: plt.Figure, output: Path, stem: str, dpi: int) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def build_figures(
    attack_root: Path,
    baseline_root: Path,
    performance_path: Path,
    output: Path,
    dpi: int = 220,
) -> list[Path]:
    _style()
    _require_strict_gates(attack_root)
    analysis = _read_csv(attack_root / "semantic_analysis.csv")
    expected = {(a, d) for a in ATTACK_LABELS for d in ("clip_only", "rtc_v3")}
    observed = set(zip(analysis["attack"], analysis["defense"]))
    if not expected.issubset(observed):
        raise ValueError(f"semantic analysis is incomplete: missing {sorted(expected-observed)}")

    rounds: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for attack in ATTACK_LABELS:
        for defense in ("clip_only", "rtc_v3"):
            frame = load_rounds(attack_root, attack, defense)
            rounds[(attack, defense)] = frame
            summary_rows.append({"attack": attack, "defense": defense, **active_summary(frame)})
    baseline_summary = _read_csv(baseline_root / "periodic_attack_runs.csv")
    for attack in ATTACK_LABELS:
        row = baseline_summary.loc[
            (baseline_summary["attack"] == attack)
            & (baseline_summary["defense"] == "rtc_v3")
        ]
        if len(row) != 1:
            raise ValueError(f"expected one promoted V2 summary row for {attack}")
        record = row.iloc[0]
        baseline_rounds = _read_csv(_round_file(baseline_root, attack, "rtc_v3"))
        accuracy = pd.to_numeric(baseline_rounds["server_accuracy"], errors="coerce")
        round_number = pd.to_numeric(baseline_rounds["round"], errors="coerce")
        trained_accuracy = accuracy.loc[round_number > 0]
        if len(trained_accuracy) < 60 or trained_accuracy.isna().any():
            raise ValueError(f"incomplete promoted V2 accuracy rounds for {attack}")
        summary_rows.append(
            {
                "attack": attack,
                "defense": "rtc_v2",
                "active_mean_asr": float(record["active_asr"]),
                "active_peak_asr": float(record["peak_asr"]),
                "final_accuracy": float(record["final_accuracy"]),
                "last10_accuracy": float(trained_accuracy.tail(10).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    if not bool(performance.get("passed")):
        raise ValueError("performance gates did not pass")

    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "plotted_summary.csv", index=False)
    figures = [
        ("attack_timelines", plot_timelines(rounds)),
        ("headline_comparison", plot_headline(summary)),
        ("mechanism_evidence", plot_mechanisms(analysis)),
        ("performance", plot_performance(performance)),
    ]
    paths: list[Path] = [output / "plotted_summary.csv"]
    with PdfPages(output / "rtc_v3_semantic_results_all_figures.pdf") as multipage:
        for stem, fig in figures:
            paths.extend(_save(fig, output, stem, dpi))
            multipage.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    paths.append(output / "rtc_v3_semantic_results_all_figures.pdf")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-root", type=Path,
                        default=Path("logs/rtc_v3_semantic_heldout_s42_attacks"))
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("logs/rtc_v3_cone_v2_heldout_attacks"))
    parser.add_argument("--performance-gates", type=Path,
                        default=Path("logs/rtc_v3_semantic_performance_formal/performance_gates.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/figures/rtc_v3_semantic_results"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = build_figures(
        args.attack_root, args.baseline_root, args.performance_gates, args.output, args.dpi
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
