"""Plot the completed LIE z=0.5 comparison for RFA, FLTrust, and FoolsGold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


DEFENSES = ("rfa", "fltrust", "foolsgold")
LABELS = {"rfa": "RFA", "fltrust": "FLTrust", "foolsgold": "FoolsGold"}
COLORS = {"rfa": "#2563A6", "fltrust": "#D08A2E", "foolsgold": "#8A7A36"}
STYLES = {"rfa": "-", "fltrust": "--", "foolsgold": "-."}
MARKERS = {"rfa": "o", "fltrust": "s", "foolsgold": "^"}


def load(root: Path, expected_rounds: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runs = pd.read_csv(root / "periodic_attack_runs.csv")
    selected = runs[
        (runs["attack"].astype(str) == "lie")
        & runs["defense"].astype(str).isin(DEFENSES)
    ].copy()
    if len(selected) != len(DEFENSES) or set(selected["defense"].astype(str)) != set(DEFENSES):
        raise ValueError("expected one completed run for each of RFA, FLTrust, and FoolsGold")
    if selected["trial_plan_hash"].astype(str).nunique() != 1:
        raise ValueError("the three defenses do not share one TrialPlan")
    if set(pd.to_numeric(selected["lie_z"], errors="raise")) != {0.5}:
        raise ValueError("expected lie_z=0.5")
    if set(pd.to_numeric(selected["malicious_fraction"], errors="raise")) != {0.2}:
        raise ValueError("expected malicious_fraction=0.2")

    frames: dict[str, pd.DataFrame] = {}
    for _, spec in selected.iterrows():
        run_id = str(spec["run_id"])
        status_path = root / "status" / f"{run_id}.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed" or int(status.get("last_round", -1)) != expected_rounds:
            raise ValueError(f"incomplete run: {run_id}")
        frame = pd.read_csv(root / "rounds" / f"{run_id}.csv")
        rounds = pd.to_numeric(frame["round"], errors="raise").astype(int)
        if rounds.tolist() != list(range(expected_rounds + 1)):
            raise ValueError(f"round sequence mismatch: {run_id}")
        for column in ("server_accuracy", "server_loss"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {column}: {run_id}")
            frame[column] = values
        active = pd.to_numeric(frame["planned_attack_active"], errors="raise").astype(int)
        if active.tolist() != [int(round_no >= 11) for round_no in rounds]:
            raise ValueError(f"attack schedule mismatch: {run_id}")
        frames[str(spec["defense"])] = frame
    return selected, frames


def summarize(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for defense in DEFENSES:
        frame = frames[defense]
        active = frame[frame["round"] >= 11]
        best_index = active["server_accuracy"].idxmax()
        rows.append(
            {
                "defense": defense,
                "label": LABELS[defense],
                "active_mean_accuracy": float(active["server_accuracy"].mean()),
                "active_std_accuracy": float(active["server_accuracy"].std(ddof=1)),
                "active_min_accuracy": float(active["server_accuracy"].min()),
                "best_accuracy": float(frame.loc[best_index, "server_accuracy"]),
                "best_round": int(frame.loc[best_index, "round"]),
                "final_accuracy": float(frame["server_accuracy"].iloc[-1]),
                "round_10_accuracy": float(frame.loc[frame["round"] == 10, "server_accuracy"].iloc[0]),
                "round_11_accuracy": float(frame.loc[frame["round"] == 11, "server_accuracy"].iloc[0]),
            }
        )
    result = pd.DataFrame(rows)
    result["attack_onset_change"] = result["round_11_accuracy"] - result["round_10_accuracy"]
    return result


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4B5563",
            "axes.facecolor": "#FCFCFD",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def decorate_round_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.axvspan(11, 60, color="#F3F4F6", alpha=0.65, zorder=0)
    ax.axvline(11, color="#111827", linestyle=":", linewidth=1.2)
    ax.text(11.6, 0.96, "LIE starts (round 11)", transform=ax.get_xaxis_transform(), va="top", color="#374151")
    ax.set(xlim=(0, 60), xlabel="Federated round", ylabel=ylabel)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_timelines(frames: dict[str, pd.DataFrame], output: Path) -> Path:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 9.0),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    for defense in DEFENSES:
        frame = frames[defense]
        common = {
            "color": COLORS[defense],
            "linestyle": STYLES[defense],
            "linewidth": 2.0,
            "marker": MARKERS[defense],
            "markevery": 5,
            "markersize": 4.2,
            "label": LABELS[defense],
        }
        axes[0].plot(frame["round"], frame["server_accuracy"], **common)
        axes[1].plot(frame["round"], frame["server_loss"], **common)
    decorate_round_axis(axes[0], "Server test accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(loc="lower right", ncol=3, frameon=False)
    decorate_round_axis(axes[1], "Server test loss")
    axes[1].set_ylim(bottom=0.0)
    fig.suptitle("LIE z=0.5: defense trajectories", x=0.075, y=0.985, ha="left", fontsize=16)
    fig.text(
        0.075,
        0.952,
        "CIFAR-10 / ResNet18; seed 42; IID; 20 clients; 10 selected per round; malicious fraction 0.2.",
        color="#5F6368",
    )
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.98, hspace=0.28)
    path = output / "lie_z05_defense_timelines.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(summary: pd.DataFrame, output: Path) -> Path:
    ordered = summary.sort_values("active_mean_accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.barh(
        ordered["label"],
        ordered["active_mean_accuracy"],
        color=[COLORS[value] for value in ordered["defense"]],
        edgecolor="#374151",
        linewidth=0.7,
    )
    for index, row in ordered.iterrows():
        ax.text(
            row["active_mean_accuracy"] - 0.012,
            index,
            f"mean {row['active_mean_accuracy']:.1%}",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
        )
        ax.plot(
            row["final_accuracy"],
            index,
            marker="D",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#111827",
            linestyle="none",
        )
        ax.text(
            row["final_accuracy"] + 0.012,
            index,
            f"{row['final_accuracy']:.1%}",
            ha="left",
            va="center",
            color="#111827",
        )
    ax.plot([], [], marker="D", markerfacecolor="white", markeredgecolor="#111827", linestyle="none", label="Final round")
    ax.set(xlim=(0.0, 1.0), xlabel="Accuracy", ylabel="Defense")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("LIE z=0.5: active-round accuracy", x=0.10, y=0.985, ha="left", fontsize=16)
    fig.text(0.10, 0.935, "Bars show the mean over rounds 11–60; diamonds show round 60. Higher is better.", color="#5F6368")
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.14, right=0.98)
    path = output / "lie_z05_active_accuracy.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "plots" / "lie_z05_three_defenses").resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, frames = load(root, args.rounds)
    summary = summarize(frames)
    summary.to_csv(output / "summary.csv", index=False)
    configure_style()
    paths = [plot_timelines(frames, output), plot_summary(summary, output)]
    metadata = {
        "source_root": str(root),
        "attack": "lie",
        "lie_z": 0.5,
        "malicious_fraction": 0.2,
        "seed": 42,
        "attack_start_round": 11,
        "rounds": args.rounds,
        "defenses": list(DEFENSES),
        "trial_plan_hash": str(runs.iloc[0]["trial_plan_hash"]),
        "note": "Single-seed descriptive comparison; no clean baseline was run.",
        "plots": [str(path) for path in paths],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
