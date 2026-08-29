"""Plot completed cells from an interrupted rtc-byzantine run.

By default, missing or failed cells are left blank instead of being imputed.
Every plotted cell must be completed for all 60 rounds and have its plan-matched
clean FedAvg trajectory. The output is intentionally descriptive: a one-seed
experiment cannot support confidence intervals or formal ranking claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


DEFAULT_ATTACKS = (
    "dba",
    "scaling_backdoor",
    "lie",
    "min_max",
    "min_sum",
    "gaussian_noise",
    "sign_flip",
)
DEFAULT_DEFENSES = ("fedavg", "rtc_full", "krum", "multi_krum", "median")
ATTACK_LABELS = {
    "dba": "DBA",
    "lie": "LIE",
    "min_max": "Min-Max",
    "min_sum": "Min-Sum",
    "gaussian_noise": "Gaussian",
    "sign_flip": "Sign Flip",
    "scaling_backdoor": "Scaling Backdoor",
}
DEFENSE_LABELS = {
    "fedavg": "No defense",
    "rtc_full": "RTC",
    "krum": "Krum",
    "multi_krum": "Multi-Krum",
    "median": "Median",
}
DEFENSE_COLORS = {
    "fedavg": "#4B5563",
    "rtc_full": "#2563A6",
    "krum": "#D08A2E",
    "multi_krum": "#8A7A36",
    "median": "#B25D79",
}
DEFENSE_STYLES = {
    "fedavg": "--",
    "rtc_full": "-",
    "krum": "-.",
    "multi_krum": ":",
    "median": (0, (5, 2)),
}


def _csv_names(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _status(root: Path, run_id: str) -> dict[str, Any]:
    path = root / "status" / f"{run_id}.json"
    if not path.is_file():
        raise ValueError(f"missing status for {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _round_frame(root: Path, run_id: str, rounds: int) -> pd.DataFrame:
    path = root / "rounds" / f"{run_id}.csv"
    if not path.is_file():
        raise ValueError(f"missing round file for {run_id}")
    frame = pd.read_csv(path)
    observed = pd.to_numeric(frame["round"], errors="raise").astype(int).tolist()
    expected = list(range(0, rounds + 1))
    if observed != expected:
        raise ValueError(f"incomplete or unordered rounds for {run_id}: {observed}")
    accuracy = pd.to_numeric(frame["server_accuracy"], errors="coerce")
    if not np.isfinite(accuracy).all() or not accuracy.between(0.0, 1.0).all():
        raise ValueError(f"invalid server accuracy for {run_id}")
    return frame


def load_completed_subset(
    root: Path,
    attacks: tuple[str, ...],
    defenses: tuple[str, ...],
    rounds: int,
    require_complete: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    runs_path = root / "periodic_attack_runs.csv"
    matrix_path = root / "experiment_matrix.csv"
    if not runs_path.is_file() or not matrix_path.is_file():
        raise ValueError("experiment summary or matrix is missing")
    runs = pd.read_csv(runs_path)
    matrix = pd.read_csv(matrix_path)
    selected = runs[
        runs["attack"].astype(str).isin(attacks)
        & runs["defense"].astype(str).isin(defenses)
    ].copy()
    expected = {(attack, defense) for attack in attacks for defense in defenses}
    observed = set(zip(selected["attack"].astype(str), selected["defense"].astype(str)))
    if require_complete and (observed != expected or len(selected) != len(expected)):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"requested subset is not complete: missing={missing}, extra={extra}")

    if selected.empty:
        raise ValueError("none of the requested attack/defense cells is complete")

    attacked_frames: dict[str, pd.DataFrame] = {}
    clean_frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        run_id = str(row["run_id"])
        status = _status(root, run_id)
        if status.get("state") != "completed" or int(status.get("last_round", -1)) != rounds:
            raise ValueError(f"requested run is not complete: {run_id}")
        frame = _round_frame(root, run_id, rounds)
        active = pd.to_numeric(frame["planned_attack_active"], errors="coerce").fillna(0).astype(int)
        expected_active = (pd.to_numeric(frame["round"]) >= int(row["attack_start_round"])).astype(int)
        if not active.equals(expected_active):
            raise ValueError(f"attack schedule mismatch for {run_id}")
        attacked_frames[run_id] = frame
        coverage_rows.append(
            {
                "attack": str(row["attack"]),
                "defense": str(row["defense"]),
                "run_id": run_id,
                "state": str(status["state"]),
                "last_round": int(status["last_round"]),
                "trial_plan_hash": str(row["trial_plan_hash"]),
            }
        )

    available_attacks = tuple(dict.fromkeys(selected["attack"].astype(str)))
    for attack in available_attacks:
        plans = selected.loc[selected["attack"].astype(str) == attack, "trial_plan_hash"].astype(str).unique()
        if len(plans) != 1:
            raise ValueError(f"defenses do not share one TrialPlan for {attack}: {plans}")
        plan = plans[0]
        clean = runs[
            runs["attack"].astype(str).eq("none")
            & runs["defense"].astype(str).eq("fedavg")
            & runs["trial_plan_hash"].astype(str).eq(plan)
        ]
        if len(clean) == 0 and not require_complete:
            continue
        if len(clean) != 1:
            raise ValueError(f"expected one plan-matched clean baseline for {attack}")
        clean_id = str(clean.iloc[0]["run_id"])
        clean_status = _status(root, clean_id)
        if clean_status.get("state") != "completed" or int(clean_status.get("last_round", -1)) != rounds:
            raise ValueError(f"clean baseline is not complete for {attack}: {clean_id}")
        clean_frames[attack] = _round_frame(root, clean_id, rounds)

    planned_attacked = matrix[matrix["attack"].astype(str).ne("none")]
    completed_attacked = runs[runs["attack"].astype(str).ne("none")]
    coverage = pd.DataFrame(coverage_rows)
    coverage.attrs["planned_attacked_cells"] = int(len(planned_attacked))
    coverage.attrs["completed_attacked_cells"] = int(len(completed_attacked))
    coverage.attrs["requested_cells"] = int(len(expected))
    coverage.attrs["missing_cells"] = sorted(f"{attack}/{defense}" for attack, defense in expected - observed)
    return selected, attacked_frames, clean_frames, coverage


def summarize(
    selected: pd.DataFrame,
    attacked_frames: dict[str, pd.DataFrame],
    clean_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targeted = {"dba", "scaling_backdoor"}
    for _, spec in selected.iterrows():
        attack = str(spec["attack"])
        defense = str(spec["defense"])
        frame = attacked_frames[str(spec["run_id"])].copy()
        clean = clean_frames.get(attack)
        frame["round"] = pd.to_numeric(frame["round"])
        active = frame[pd.to_numeric(frame["planned_attack_active"], errors="coerce").eq(1)].copy()
        attacked_accuracy = pd.to_numeric(active["server_accuracy"], errors="coerce")
        clean_accuracy = pd.Series(np.nan, index=attacked_accuracy.index, dtype=float)
        if clean is not None:
            clean = clean.copy()
            clean["round"] = pd.to_numeric(clean["round"])
            merged = active[["round", "server_accuracy"]].merge(
                clean[["round", "server_accuracy"]], on="round", suffixes=("_attack", "_clean")
            )
            attacked_accuracy = pd.to_numeric(merged["server_accuracy_attack"], errors="coerce")
            clean_accuracy = pd.to_numeric(merged["server_accuracy_clean"], errors="coerce")
        result: dict[str, Any] = {
            "attack": attack,
            "attack_label": ATTACK_LABELS.get(attack, attack),
            "defense": defense,
            "defense_label": DEFENSE_LABELS.get(defense, defense),
            "seed": int(spec["seed"]),
            "active_rounds": int(len(active)),
            "active_accuracy": float(attacked_accuracy.mean()),
            "active_clean_accuracy": float(clean_accuracy.mean()),
            "active_accuracy_drop": float((clean_accuracy - attacked_accuracy).mean()),
            "active_min_accuracy": float(attacked_accuracy.min()),
            "final_accuracy": float(pd.to_numeric(frame["server_accuracy"]).iloc[-1]),
            "active_asr": np.nan,
            "peak_asr": np.nan,
            "final_asr": np.nan,
            "nominal_f_violation_rate": np.nan,
            "aggregation_seconds": np.nan,
            "attacker_weight_share": np.nan,
            "clip_recall_active_attackers": np.nan,
        }
        if "fit_aggregation_time_seconds" in active:
            result["aggregation_seconds"] = float(
                pd.to_numeric(active["fit_aggregation_time_seconds"], errors="coerce").mean()
            )
        weight_column = (
            "fit_active_attacker_weight_share"
            if "fit_active_attacker_weight_share" in active
            else "fit_malicious_aggregation_weight_share"
        )
        if weight_column in active:
            result["attacker_weight_share"] = float(
                pd.to_numeric(active[weight_column], errors="coerce").mean()
            )
        if "fit_clip_recall_active_attackers" in active:
            result["clip_recall_active_attackers"] = float(
                pd.to_numeric(active["fit_clip_recall_active_attackers"], errors="coerce").mean()
            )
        if attack in targeted:
            asr = pd.to_numeric(active["server_asr"], errors="coerce")
            if not np.isfinite(asr).all() or not asr.between(0.0, 1.0).all():
                raise ValueError(f"invalid active ASR for {attack}/{defense}")
            result.update(
                active_asr=float(asr.mean()),
                peak_asr=float(asr.max()),
                final_asr=float(pd.to_numeric(frame["server_asr"], errors="coerce").iloc[-1]),
            )
        if defense in {"krum", "multi_krum"}:
            actual = pd.to_numeric(active["fit_selected_malicious_clients"], errors="coerce")
            nominal_f = int(spec["krum_num_malicious"])
            result["nominal_f_violation_rate"] = float((actual > nominal_f).mean())
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["attack", "defense"]).reset_index(drop=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4B5563",
            "axes.facecolor": "#FCFCFD",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _finish(fig: plt.Figure, path: Path, *, top: float = 0.95) -> None:
    fig.tight_layout(rect=(0.0, 0.02, 1.0, top))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_timelines(
    selected: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    attacks: tuple[str, ...],
    defenses: tuple[str, ...],
    output: Path,
) -> Path:
    ncols = 2
    nrows = int(np.ceil(len(attacks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 3.2 * nrows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, attack in zip(axes_flat, attacks):
        for defense in defenses:
            rows = selected[
                selected["attack"].astype(str).eq(attack)
                & selected["defense"].astype(str).eq(defense)
            ]
            if rows.empty:
                continue
            row = rows.iloc[0]
            frame = frames[str(row["run_id"])]
            rounds = pd.to_numeric(frame["round"])
            accuracy = pd.to_numeric(frame["server_accuracy"])
            ax.plot(
                rounds,
                accuracy,
                color=DEFENSE_COLORS[defense],
                linestyle=DEFENSE_STYLES[defense],
                linewidth=1.7,
                label=DEFENSE_LABELS[defense],
            )
        ax.axvline(11, color="#6B7280", linestyle=":", linewidth=1.1)
        ax.set_title(ATTACK_LABELS.get(attack, attack), loc="left")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 60)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for row_index in range(nrows):
        axes_array[row_index, 0].set_ylabel("Main-task accuracy")
    for ax in axes_array[-1, :]:
        ax.set_xlabel("Server round")
    for ax in axes_flat[len(attacks):]:
        ax.set_visible(False)
    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Main-task accuracy trajectories", x=0.06, y=0.995, ha="left", fontsize=15)
    fig.text(0.06, 0.972, "Seed 42; strong attacks; attack begins at round 11. Missing/failed cells are omitted.", color="#5F6368")
    path = output / "accuracy_timelines.png"
    _finish(fig, path, top=0.93)
    return path


def plot_backdoor_asr_timelines(
    selected: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    defenses: tuple[str, ...],
    output: Path,
) -> Path:
    attacks = ("dba", "scaling_backdoor")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharex=True, sharey=True)
    for ax, attack in zip(axes, attacks):
        for defense in defenses:
            row = selected[
                selected["attack"].astype(str).eq(attack)
                & selected["defense"].astype(str).eq(defense)
            ].iloc[0]
            frame = frames[str(row["run_id"])]
            ax.plot(
                pd.to_numeric(frame["round"]),
                pd.to_numeric(frame["server_asr"], errors="coerce"),
                color=DEFENSE_COLORS[defense],
                linestyle=DEFENSE_STYLES[defense],
                linewidth=1.7,
                label=DEFENSE_LABELS[defense],
            )
        ax.axvline(11, color="#6B7280", linestyle=":", linewidth=1.1)
        ax.set_title(ATTACK_LABELS[attack], loc="left")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 60)
        ax.set_xlabel("Server round")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Attack success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.85))
    fig.suptitle("Backdoor attack success trajectories", x=0.06, y=0.995, ha="left", fontsize=15)
    fig.text(0.06, 0.920, "Seed 42; ASR is evaluated per server round; attack begins at round 11.", color="#5F6368")
    path = output / "backdoor_asr_timelines.png"
    _finish(fig, path, top=0.76)
    return path


def _annotated_heatmap(
    table: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    colorbar_label: str,
    path: Path,
    diverging: bool,
) -> Path:
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 4.8 if len(table) > 2 else 3.8))
    if diverging:
        limit = max(0.01, float(np.nanmax(np.abs(values))))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = plt.get_cmap("PuOr").copy()
        cmap.set_bad("#E5E7EB")
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    else:
        cmap = plt.get_cmap("Oranges").copy()
        cmap.set_bad("#E5E7EB")
        image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        limit = 1.0
    ax.set_xticks(range(len(table.columns)), [DEFENSE_LABELS.get(x, x) for x in table.columns])
    ax.set_yticks(range(len(table.index)), [ATTACK_LABELS.get(x, x) for x in table.index])
    ax.set_xlabel("Defense")
    ax.set_ylabel("Attack")
    ax.set_title(title, loc="left", pad=26, fontsize=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, color="#5F6368", fontsize=9)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                color = "white" if abs(value) > limit * 0.55 else "#1F2937"
                ax.text(col, row, f"{value:.1%}", ha="center", va="center", color=color, fontsize=9)
            else:
                ax.text(col, row, "N/A", ha="center", va="center", color="#6B7280", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label(colorbar_label)
    _finish(fig, path)
    return path


def plot_heatmaps(summary: pd.DataFrame, attacks: tuple[str, ...], defenses: tuple[str, ...], output: Path) -> list[Path]:
    absolute_accuracy = summary.pivot(index="attack", columns="defense", values="active_accuracy").reindex(
        index=attacks, columns=defenses
    )
    accuracy = summary.pivot(index="attack", columns="defense", values="active_accuracy_drop").reindex(
        index=attacks, columns=defenses
    )
    asr = summary[summary["attack"].isin(["dba", "scaling_backdoor"])].pivot(
        index="attack", columns="defense", values="active_asr"
    ).reindex(index=["dba", "scaling_backdoor"], columns=defenses)
    return [
        _annotated_heatmap(
            absolute_accuracy,
            title="Active-round main-task accuracy",
            subtitle="Mean server accuracy over rounds 11–60; higher is better. N/A means no completed trajectory.",
            colorbar_label="Mean active accuracy",
            path=output / "active_accuracy_heatmap.png",
            diverging=False,
        ),
        _annotated_heatmap(
            accuracy,
            title="Active-round accuracy drop",
            subtitle="Plan-matched clean FedAvg accuracy minus attacked-defense accuracy; negative favors the defense.",
            colorbar_label="Accuracy drop",
            path=output / "active_accuracy_drop_heatmap.png",
            diverging=True,
        ),
        _annotated_heatmap(
            asr,
            title="Active-round backdoor ASR",
            subtitle="Mean ASR over rounds 11–60; lower is better.",
            colorbar_label="Mean active ASR",
            path=output / "backdoor_active_asr_heatmap.png",
            diverging=False,
        ),
    ]


def write_chart_map(paths: list[Path], output: Path) -> None:
    descriptions = {
        "accuracy_timelines.png": "Seven-panel 60-round accuracy trajectories; missing or failed cells are omitted.",
        "backdoor_asr_timelines.png": "DBA and Scaling Backdoor ASR trajectories across the five defenses.",
        "active_accuracy_heatmap.png": "Mean active-round accuracy for all seven attacks; N/A is not imputed.",
        "active_accuracy_drop_heatmap.png": "Plan-matched active-round utility drop for seven attacks and five defenses; N/A is not imputed.",
        "backdoor_active_asr_heatmap.png": "Active-round mean ASR for the two targeted attacks and five defenses.",
    }
    pd.DataFrame(
        {
            "artifact": path.name,
            "question": descriptions[path.name],
            "palette_policy": "relaxed multi-category for defense trajectories; single/two-root for heatmaps",
            "evidence_scope": "completed 60-round seed-42 subset only; no confidence intervals",
        }
        for path in paths
    ).to_csv(output / "chart_map.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--defenses", default=",".join(DEFAULT_DEFENSES))
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "partial_analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    attacks = _csv_names(args.attacks)
    defenses = _csv_names(args.defenses)
    selected, frames, clean, coverage = load_completed_subset(
        root, attacks, defenses, args.rounds, args.require_complete
    )
    summary = summarize(selected, frames, clean)
    summary.to_csv(output / "completed_subset_summary.csv", index=False)
    coverage.to_csv(output / "completed_subset_coverage.csv", index=False)
    metadata = {
        "source_root": str(root),
        "planned_attacked_cells": coverage.attrs["planned_attacked_cells"],
        "completed_attacked_cells": coverage.attrs["completed_attacked_cells"],
        "requested_cells": coverage.attrs["requested_cells"],
        "requested_cells_complete": int(len(coverage)),
        "missing_cells": coverage.attrs["missing_cells"],
        "attacks": list(attacks),
        "defenses": list(defenses),
        "rounds": int(args.rounds),
        "seed_count": int(summary["seed"].nunique()),
        "interpretation": "descriptive available-cell analysis; missing cells are not imputed; single seed",
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _style()
    paths = [
        plot_accuracy_timelines(selected, frames, attacks, defenses, output),
        plot_backdoor_asr_timelines(selected, frames, defenses, output),
        *plot_heatmaps(summary, attacks, defenses, output),
    ]
    write_chart_map(paths, output)
    print(json.dumps({"output": str(output), "plots": [str(path) for path in paths], **metadata}, indent=2))


if __name__ == "__main__":
    main()
