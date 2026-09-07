"""Fail-closed aggregation and plotting for the RTC-V3 Byzantine matrix.

Formal experiments may be executed in several resumable roots.  This module
merges those roots only after every local quality gate passes, rejects
conflicting duplicate runs, verifies the frozen attack implementation hash,
and emits paired statistics and static publication figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from attacks.spec import attack_source_hash
from experiments import periodic_attack
from experiments.rtc_v3.byzantine import CANONICAL_ATTACKS, DEFAULT_DEFENSES


ATTACK_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "random_noise": "Random noise",
    "sign_flip": "Sign flip",
    "lie": "LIE / ALIE",
    "min_max": "Min-Max",
    "min_sum": "Min-Sum",
    "scaling_backdoor": "Scaling backdoor",
    "label_flip_targeted": "Targeted label flip",
    "label_flip_all_reverse": "Reverse label flip",
    "dba": "DBA",
}
DEFENSE_LABELS = {
    "fedavg": "FedAvg",
    "rtc_full": "RTC-V3",
    "krum": "Krum",
    "trimmed_mean": "Trimmed mean",
    "median": "Median",
    "foolsgold": "FoolsGold",
    "rfa": "RFA / geometric median",
    "freqfed": "FreqFed",
    "fltrust": "FLTrust (trusted root)",
}
DEFENSE_COLORS = {
    "fedavg": "#8A8F98",
    "rtc_full": "#1769AA",
    "krum": "#D89000",
    "trimmed_mean": "#7A8F35",
    "median": "#A65D8F",
    "foolsgold": "#4F759B",
    "rfa": "#6A4C93",
    "freqfed": "#B56B45",
    "fltrust": "#2A9D8F",
}
TARGETED = {"scaling_backdoor", "label_flip_targeted", "dba"}
REQUIRED_GLOBAL_GATES = {
    "selected_partition_ids_match",
    "data_manifest_hash_match",
    "trial_plan_hash_match",
    "trial_plan_sequence_match",
    "completed_sequence_equals_plan",
    "client_random_stream_match",
    "malicious_identity_match",
    "paired_data_manifest_match",
    "initial_model_match",
}


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _canonical_row_hash(row: Mapping[str, object]) -> str:
    normalized = {
        str(key): (None if pd.isna(value) else value)
        for key, value in row.items()
    }
    payload = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_root(root: Path, freeze: Mapping[str, object]) -> None:
    gates_path = root / "quality_gates.csv"
    runs_path = root / "byzantine_attack_runs.csv"
    if not gates_path.is_file() or not runs_path.is_file():
        raise FileNotFoundError(f"incomplete result root: {root}")
    gates = pd.read_csv(gates_path)
    if gates.empty or not {"gate", "passed"}.issubset(gates.columns):
        raise ValueError(f"malformed quality gates: {gates_path}")
    failed = gates.loc[~gates["passed"].map(_truth), "gate"].astype(str).tolist()
    if failed:
        raise ValueError(f"failed gates in {root}: {failed}")
    observed = set(gates["gate"].astype(str))
    missing = REQUIRED_GLOBAL_GATES.difference(observed)
    if missing:
        raise ValueError(f"missing strict gates in {root}: {sorted(missing)}")

    runs = pd.read_csv(runs_path)
    if runs.empty:
        raise ValueError(f"empty result root: {root}")
    if "attack_implementation_hash" not in runs:
        raise ValueError(f"attack implementation hash missing in {root}")
    expected = str(freeze["implementation_source_sha256"])
    attacked = runs.loc[runs["attack"].astype(str) != "none"]
    hashes = set(attacked["attack_implementation_hash"].dropna().astype(str))
    if hashes != {expected}:
        raise ValueError(
            f"attack implementation hash mismatch in {root}: {sorted(hashes)} != {expected}"
        )


def load_and_merge(
    roots: Sequence[Path], freeze_path: Path
) -> tuple[pd.DataFrame, dict[str, Path], Mapping[str, object]]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "RTCByzantineAttackFreezeV1":
        raise ValueError("formal analysis requires RTCByzantineAttackFreezeV1")
    if freeze.get("implementation_source_sha256") != attack_source_hash():
        raise ValueError("freeze is stale relative to the current attack implementation")

    frames: list[pd.DataFrame] = []
    round_files: dict[str, Path] = {}
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        validate_root(root, freeze)
        frame = pd.read_csv(root / "byzantine_attack_runs.csv")
        frame["_source_root"] = str(root)
        frames.append(frame)
        for path in (root / "rounds").glob("*.csv"):
            existing = round_files.get(path.stem)
            if existing is not None and existing.read_bytes() != path.read_bytes():
                raise ValueError(f"conflicting duplicate round file: {path.stem}")
            round_files[path.stem] = path

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "run_id" not in merged:
        raise ValueError("result summaries do not contain run_id")
    keep: list[int] = []
    for run_id, group in merged.groupby("run_id", sort=False):
        comparable = group.drop(columns=["_source_root"], errors="ignore")
        hashes = {_canonical_row_hash(row) for row in comparable.to_dict("records")}
        if len(hashes) != 1:
            raise ValueError(f"conflicting duplicate summary row: {run_id}")
        keep.append(int(group.index[0]))
    merged = merged.loc[keep].reset_index(drop=True)
    expected_files = set(merged["run_id"].astype(str))
    missing_files = expected_files.difference(round_files)
    if missing_files:
        raise ValueError(f"missing round files: {sorted(missing_files)[:5]}")
    return merged, round_files, freeze


def validate_complete_matrix(
    summary: pd.DataFrame,
    *,
    attacks: Sequence[str] = CANONICAL_ATTACKS,
    defenses: Sequence[str] = DEFAULT_DEFENSES,
    seeds: Sequence[int] = (42, 43, 44),
) -> pd.DataFrame:
    gates: list[dict[str, object]] = []
    attacked = summary.loc[summary["attack"].astype(str) != "none"].copy()
    for attack in attacks:
        for seed in seeds:
            rows = attacked.loc[
                (attacked["attack"].astype(str) == attack)
                & (pd.to_numeric(attacked["seed"], errors="coerce") == seed)
            ]
            observed_defenses = set(rows["defense"].astype(str))
            plan_hashes = set(rows["trial_plan_hash"].dropna().astype(str))
            passed = observed_defenses == set(defenses) and len(plan_hashes) == 1
            gates.append(
                {
                    "gate": f"complete_matrix:{attack}:seed{seed}",
                    "passed": passed,
                    "observed": (
                        f"defenses={sorted(observed_defenses)} plan_hashes={len(plan_hashes)}"
                    ),
                    "required": f"defenses={sorted(defenses)} plan_hashes=1",
                }
            )
    clean = summary.loc[
        (summary["attack"].astype(str) == "none")
        & (summary["defense"].astype(str) == "fedavg")
    ]
    attacked_plan_hashes = set(attacked["trial_plan_hash"].dropna().astype(str))
    clean_plan_hashes = set(clean["trial_plan_hash"].dropna().astype(str))
    gates.append(
        {
            "gate": "paired_clean_coverage",
            "passed": attacked_plan_hashes == clean_plan_hashes,
            "observed": f"attacked={len(attacked_plan_hashes)} clean={len(clean_plan_hashes)}",
            "required": "identical trial-plan hash sets",
        }
    )
    result = pd.DataFrame(gates)
    if not bool(result["passed"].all()):
        failed = result.loc[~result["passed"], "gate"].tolist()
        raise ValueError(f"formal matrix is incomplete: {failed}")
    return result


def paired_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Return defense-minus-FedAvg effects at the exact TrialPlan/seed grain."""
    attacked = summary.loc[summary["attack"].astype(str) != "none"].copy()
    rows: list[dict[str, object]] = []
    for attack, group in attacked.groupby("attack", sort=False):
        metric = "active_asr_auc" if str(attack) in TARGETED else "accuracy_drop_auc"
        if metric not in group:
            continue
        pivot = group.pivot_table(
            index=["trial_plan_hash", "seed"],
            columns="defense",
            values=metric,
            aggfunc="first",
        )
        if "fedavg" not in pivot:
            continue
        for defense in DEFAULT_DEFENSES:
            if defense not in pivot:
                continue
            if defense == "fedavg":
                delta = pd.Series(0.0, index=pivot["fedavg"].dropna().index)
            else:
                paired = pivot[["fedavg", defense]].dropna()
                # Positive values always mean improvement over FedAvg.
                delta = paired["fedavg"] - paired[defense]
            rows.append(
                {
                    "attack": attack,
                    "defense": defense,
                    "metric": metric,
                    "n": len(delta),
                    "mean_improvement": float(delta.mean()) if len(delta) else math.nan,
                    "median_improvement": float(delta.median()) if len(delta) else math.nan,
                    "min_improvement": float(delta.min()) if len(delta) else math.nan,
                    "max_improvement": float(delta.max()) if len(delta) else math.nan,
                    "all_seeds_non_worse": bool((delta >= -0.02).all()) if len(delta) else False,
                }
            )
    return pd.DataFrame(rows)


def paired_rtc_statistics(summary: pd.DataFrame, *, seed: int = 2026) -> pd.DataFrame:
    """Compare RTC-V3 with every comparator at the paired seed/plan grain.

    Three formal seeds are intentionally treated as descriptive evidence:
    paired bootstrap intervals are reported, while permutation p-values and
    Holm correction are withheld until at least eight paired observations.
    """
    attacked = summary.loc[summary["attack"].astype(str) != "none"].copy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for attack, group in attacked.groupby("attack", sort=False):
        metric = "active_asr_auc" if str(attack) in TARGETED else "accuracy_drop_auc"
        if metric not in group:
            continue
        pivot = group.pivot_table(
            index=["trial_plan_hash", "seed"], columns="defense", values=metric,
            aggfunc="first",
        )
        if "rtc_full" not in pivot:
            continue
        for comparator in DEFAULT_DEFENSES:
            if comparator == "rtc_full" or comparator not in pivot:
                continue
            paired = pivot[[comparator, "rtc_full"]].dropna()
            differences = (paired[comparator] - paired["rtc_full"]).to_numpy(float)
            if not len(differences):
                continue
            boot = np.asarray([
                float(np.mean(rng.choice(differences, size=len(differences), replace=True)))
                for _ in range(5000)
            ])
            descriptive_only = len(differences) < 8
            p_value = (
                math.nan
                if descriptive_only
                else periodic_attack.paired_permutation_pvalue(differences, seed)
            )
            rows.append(
                {
                    "attack": attack,
                    "metric": metric,
                    "left": "rtc_full",
                    "right": comparator,
                    "n": len(differences),
                    "descriptive_only": descriptive_only,
                    "mean_improvement": float(np.mean(differences)),
                    "median_improvement": float(np.median(differences)),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "seeds_rtc_non_worse": int(np.count_nonzero(differences >= -0.02)),
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = math.nan
        valid = result["p_value"].notna()
        if valid.any():
            result.loc[valid, "p_holm"] = periodic_attack.holm_adjust(
                result.loc[valid, "p_value"].to_numpy()
            )
    return result


def defense_assumption_audit(
    summary: pd.DataFrame, round_files: Mapping[str, Path]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = summary.loc[
        (summary["attack"].astype(str) != "none")
        & summary["defense"].astype(str).isin(["krum", "trimmed_mean", "rfa"])
    ]
    for _, spec in selected.iterrows():
        run_id = str(spec["run_id"])
        frame = pd.read_csv(round_files[run_id])
        frame = frame.loc[pd.to_numeric(frame["round"], errors="coerce") > 0]
        malicious = pd.to_numeric(frame["fit_selected_malicious_clients"], errors="coerce")
        benign = pd.to_numeric(frame["fit_selected_benign_clients"], errors="coerce")
        if malicious.isna().any() or benign.isna().any():
            raise ValueError(f"missing selected-client evidence for {run_id}")
        defense = str(spec["defense"])
        max_malicious_input_weight_share = math.nan
        if defense == "krum":
            bound = float(spec["krum_num_malicious"])
            violation = malicious > bound
            assumption = f"selected malicious <= nominal f={int(bound)}"
        elif defense == "trimmed_mean":
            fraction = malicious / (malicious + benign).clip(lower=1)
            bound = float(spec["trim_fraction"])
            violation = fraction > bound + 1e-12
            assumption = f"selected malicious fraction <= trim={bound:.3f}"
        else:
            column = "fit_selected_malicious_example_share"
            if column not in frame:
                raise ValueError(f"missing RFA input-weight evidence for {run_id}")
            fraction = pd.to_numeric(frame[column], errors="coerce")
            if fraction.isna().any():
                raise ValueError(f"missing RFA input-weight evidence for {run_id}")
            bound = 0.5
            violation = fraction >= bound - 1e-12
            assumption = "malicious input sample weight share < 0.500"
            max_malicious_input_weight_share = float(fraction.max())
        rows.append(
            {
                "run_id": run_id,
                "attack": spec["attack"],
                "seed": int(spec["seed"]),
                "defense": spec["defense"],
                "assumption": assumption,
                "rounds": len(frame),
                "violating_rounds": int(violation.sum()),
                "violation_rate": float(violation.mean()),
                "max_selected_malicious": int(malicious.max()),
                "max_malicious_input_weight_share": max_malicious_input_weight_share,
            }
        )
    return pd.DataFrame(rows)


def rtc_mechanism_summary(
    summary: pd.DataFrame, round_files: Mapping[str, Path]
) -> pd.DataFrame:
    rtc_runs = summary.loc[
        (summary["attack"].astype(str) != "none")
        & (summary["defense"].astype(str) == "rtc_full")
    ]
    frames: list[pd.DataFrame] = []
    required = {
        "round",
        "attack",
        "planned_attack_active",
        "fit_selected_active_attackers",
        "fit_selected_benign_clients",
        "fit_malicious_clipped_clients",
        "fit_active_attacker_weight_share",
        "fit_rtc_v3_cone_overflow_assignments",
        "fit_rtc_v3_cumulative_active_count",
        "fit_rtc_v3_direction_active_count",
        "fit_total_defense_seconds",
        "fit_rtc_v3_max_constraint_violation",
        "fit_rtc_v3_semantic_q_min_round",
        "fit_rtc_v3_cumulative_q_min",
        "fit_rtc_v3_direction_q_min",
        "fit_rtc_v3_principal_budget_active_count",
    }
    for run_id in rtc_runs["run_id"].astype(str):
        frame = pd.read_csv(round_files[run_id])
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"RTC mechanism columns missing in {run_id}: {sorted(missing)}")
        frame = frame.loc[
            (pd.to_numeric(frame["round"], errors="coerce") > 0)
            & (pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1)
        ].copy()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    semantic_columns = [
        name for name in (
            "fit_rtc_v3_semantic_active_w1",
            "fit_rtc_v3_semantic_active_w4",
            "fit_rtc_v3_semantic_active_w8",
        ) if name in data
    ]
    if not semantic_columns:
        raise ValueError("RTC semantic active-window metrics are missing")
    for attack, group in data.groupby("attack", sort=False):
        numeric = lambda name: pd.to_numeric(group[name], errors="coerce").fillna(0.0)
        attackers = numeric("fit_selected_active_attackers")
        benign = numeric("fit_selected_benign_clients")
        total = (attackers + benign).clip(lower=1.0)
        clipped = numeric("fit_malicious_clipped_clients")
        semantic_active = pd.concat(
            [numeric(name) > 0 for name in semantic_columns], axis=1
        ).any(axis=1)
        rows.append(
            {
                "attack": attack,
                "active_round_observations": len(group),
                "clip_recall_active_attackers": float(clipped.sum() / max(1.0, attackers.sum())),
                "active_attacker_weight_share": float(numeric("fit_active_attacker_weight_share").mean()),
                "cone_overflow_share": float(numeric("fit_rtc_v3_cone_overflow_assignments").sum() / total.sum()),
                "cone_matched_share": float(1.0 - numeric("fit_rtc_v3_cone_overflow_assignments").sum() / total.sum()),
                "semantic_active_round_rate": float(semantic_active.mean()),
                "cumulative_active_round_rate": float((numeric("fit_rtc_v3_cumulative_active_count") > 0).mean()),
                "direction_active_round_rate": float((numeric("fit_rtc_v3_direction_active_count") > 0).mean()),
                "principal_budget_active_round_rate": float((numeric("fit_rtc_v3_principal_budget_active_count") > 0).mean()),
                "semantic_q_min": float(numeric("fit_rtc_v3_semantic_q_min_round").min()),
                "cumulative_q_min": float(numeric("fit_rtc_v3_cumulative_q_min").min()),
                "direction_q_min": float(numeric("fit_rtc_v3_direction_q_min").min()),
                "max_constraint_violation": float(numeric("fit_rtc_v3_max_constraint_violation").max()),
                "mean_total_defense_seconds": float(numeric("fit_total_defense_seconds").mean()),
            }
        )
    return pd.DataFrame(rows)


def runtime_summary(
    summary: pd.DataFrame, round_files: Mapping[str, Path]
) -> pd.DataFrame:
    observations: list[pd.DataFrame] = []
    for _, spec in summary.loc[summary["attack"].astype(str) != "none"].iterrows():
        frame = pd.read_csv(round_files[str(spec["run_id"])])
        frame = frame.loc[pd.to_numeric(frame["round"], errors="coerce") > 0].copy()
        frame["defense"] = str(spec["defense"])
        frame["attack"] = str(spec["attack"])
        frame["seed"] = int(spec["seed"])
        observations.append(frame)
    data = pd.concat(observations, ignore_index=True)
    rows: list[dict[str, object]] = []
    for defense, group in data.groupby("defense", sort=False):
        aggregation = pd.to_numeric(group["fit_aggregation_time_seconds"], errors="coerce").dropna()
        total_column = "fit_total_defense_seconds"
        total = (
            pd.to_numeric(group[total_column], errors="coerce").dropna()
            if total_column in group
            else aggregation
        )
        rows.append(
            {
                "defense": defense,
                "round_observations": len(aggregation),
                "aggregation_mean_seconds": float(aggregation.mean()),
                "aggregation_p95_seconds": float(aggregation.quantile(0.95)),
                "defense_total_mean_seconds": float(total.mean()),
                "defense_total_p95_seconds": float(total.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_update_reference_distances(
    summary: pd.DataFrame, round_files: Mapping[str, Path]
) -> pd.DataFrame:
    """Compare each attacked server update with its plan-matched clean update."""
    clean = summary.loc[
        (summary["attack"].astype(str) == "none")
        & (summary["defense"].astype(str) == "fedavg")
    ]
    clean_by_plan = {
        str(row["trial_plan_hash"]): str(row["run_id"])
        for _, row in clean.iterrows()
    }
    rows: list[dict[str, object]] = []
    attacked = summary.loc[summary["attack"].astype(str) != "none"]
    required = {
        "round",
        "planned_attack_active",
        "fit_aggregate_update_norm",
        "fit_aggregate_update_sketch_json",
        "fit_aggregate_update_sketch_version",
        "fit_aggregate_update_sketch_dimension",
        "fit_aggregate_update_sketch_seed",
    }
    for _, spec in attacked.iterrows():
        plan = str(spec["trial_plan_hash"])
        if plan not in clean_by_plan:
            raise ValueError(f"missing clean aggregate-update reference for plan {plan}")
        attacked_frame = pd.read_csv(round_files[str(spec["run_id"])])
        clean_frame = pd.read_csv(round_files[clean_by_plan[plan]])
        for name, frame in (("attacked", attacked_frame), ("clean", clean_frame)):
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{name} aggregate-update evidence missing for plan {plan}: {sorted(missing)}"
                )
        joined = attacked_frame[list(required)].merge(
            clean_frame[list(required)], on="round", suffixes=("_attack", "_clean")
        )
        joined = joined.loc[
            (pd.to_numeric(joined["round"], errors="coerce") > 0)
            & (pd.to_numeric(joined["planned_attack_active_attack"], errors="coerce") == 1)
        ]
        cosine: list[float] = []
        approximate_l2: list[float] = []
        norm_ratio: list[float] = []
        for _, pair in joined.iterrows():
            for field in (
                "fit_aggregate_update_sketch_version",
                "fit_aggregate_update_sketch_dimension",
                "fit_aggregate_update_sketch_seed",
            ):
                if str(pair[f"{field}_attack"]) != str(pair[f"{field}_clean"]):
                    raise ValueError(f"aggregate-update sketch contract mismatch: {field}")
            left = np.asarray(
                json.loads(str(pair["fit_aggregate_update_sketch_json_attack"])),
                dtype=np.float64,
            )
            right = np.asarray(
                json.loads(str(pair["fit_aggregate_update_sketch_json_clean"])),
                dtype=np.float64,
            )
            if left.shape != right.shape or left.ndim != 1:
                raise ValueError("aggregate-update sketch shape mismatch")
            left_norm = float(pair["fit_aggregate_update_norm_attack"])
            right_norm = float(pair["fit_aggregate_update_norm_clean"])
            cosine.append(float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0)))
            approximate_l2.append(float(np.linalg.norm(left * left_norm - right * right_norm)))
            norm_ratio.append(left_norm / max(right_norm, 1e-12))
        rows.append(
            {
                "run_id": spec["run_id"],
                "trial_plan_hash": plan,
                "attack": spec["attack"],
                "defense": spec["defense"],
                "seed": int(spec["seed"]),
                "active_rounds": len(cosine),
                "mean_cosine_distance_to_clean": float(np.mean(cosine)),
                "p95_cosine_distance_to_clean": float(np.quantile(cosine, 0.95)),
                "mean_approx_l2_to_clean": float(np.mean(approximate_l2)),
                "p95_approx_l2_to_clean": float(np.quantile(approximate_l2, 0.95)),
                "mean_update_norm_ratio_to_clean": float(np.mean(norm_ratio)),
            }
        )
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#495057",
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_effect_heatmap(effects: pd.DataFrame, output: Path) -> Path:
    table = effects.pivot(index="attack", columns="defense", values="mean_improvement")
    table = table.reindex(index=CANONICAL_ATTACKS, columns=DEFAULT_DEFENSES)
    values = table.to_numpy(dtype=float)
    limit = max(0.05, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    image = ax.imshow(values, cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [DEFENSE_LABELS.get(x, x) for x in table.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), [ATTACK_LABELS.get(x, x) for x in table.index])
    ax.set_title("Paired security improvement over FedAvg")
    ax.set_xlabel("Defense")
    ax.set_ylabel("Attack")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:+.1%}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > limit * 0.55 else "#212529")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("FedAvg metric − defense metric; positive is better")
    path = output / "paired_security_improvement_heatmap.png"
    _save(fig, path)
    return path


def plot_accuracy_heatmap(summary: pd.DataFrame, output: Path) -> Path:
    attacked = summary.loc[summary["attack"].astype(str) != "none"]
    table = attacked.pivot_table(index="attack", columns="defense", values="active_accuracy_drop", aggfunc="mean")
    table = table.reindex(index=CANONICAL_ATTACKS, columns=DEFAULT_DEFENSES)
    values = table.to_numpy(dtype=float)
    limit = max(0.05, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    image = ax.imshow(values, cmap="Oranges", vmin=0, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [DEFENSE_LABELS.get(x, x) for x in table.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), [ATTACK_LABELS.get(x, x) for x in table.index])
    ax.set_title("Active-round main-task accuracy drop")
    ax.set_xlabel("Defense")
    ax.set_ylabel("Attack")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1%}", ha="center", va="center", fontsize=7.5,
                        color="white" if value > limit * 0.55 else "#212529")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Clean FedAvg accuracy − attacked defense accuracy")
    path = output / "active_accuracy_drop_heatmap.png"
    _save(fig, path)
    return path


def plot_runtime(runtime: pd.DataFrame, output: Path) -> Path:
    runtime = runtime.sort_values("aggregation_mean_seconds")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = [DEFENSE_COLORS.get(name, "#8A8F98") for name in runtime["defense"]]
    labels = runtime["defense"].map(lambda x: DEFENSE_LABELS.get(x, x))
    bars = ax.barh(labels, runtime["aggregation_mean_seconds"],
                   color=colors, edgecolor="#343A40", linewidth=0.6, label="Mean")
    ax.scatter(runtime["aggregation_p95_seconds"], labels, marker="|", s=130,
               linewidths=2, color="#212529", label="P95")
    ax.set_title("Server aggregation time by defense")
    ax.set_xlabel("Seconds per round")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    for bar, value in zip(bars, runtime["aggregation_mean_seconds"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}s", va="center", fontsize=8)
    ax.legend(frameon=False)
    path = output / "aggregation_runtime.png"
    _save(fig, path)
    return path


def plot_assumption_violations(audit: pd.DataFrame, output: Path) -> Path:
    table = audit.pivot_table(
        index="attack", columns="defense", values="violation_rate", aggfunc="mean"
    ).reindex(index=CANONICAL_ATTACKS, columns=["krum", "trimmed_mean"])
    values = table.to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    image = ax.imshow(values, cmap="Oranges", vmin=0, vmax=max(0.5, float(np.nanmax(values))), aspect="auto")
    ax.set_xticks(range(2), [DEFENSE_LABELS[name] for name in table.columns])
    ax.set_yticks(range(len(table.index)), [ATTACK_LABELS.get(name, name) for name in table.index])
    ax.set_title("Rounds exceeding nominal robust-aggregation assumptions")
    ax.set_xlabel("Defense")
    ax.set_ylabel("Attack")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.3 else "#212529")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Fraction of 60 rounds")
    path = output / "defense_assumption_violations.png"
    _save(fig, path)
    return path


def plot_rtc_mechanisms(mechanisms: pd.DataFrame, output: Path) -> Path:
    metrics = [
        "clip_recall_active_attackers",
        "active_attacker_weight_share",
        "cone_overflow_share",
        "semantic_active_round_rate",
        "cumulative_active_round_rate",
        "direction_active_round_rate",
    ]
    labels = [
        "Clip recall",
        "Attacker weight",
        "Cone overflow",
        "Semantic active",
        "Cumulative active",
        "Direction active",
    ]
    table = mechanisms.set_index("attack").reindex(CANONICAL_ATTACKS)[metrics]
    values = table.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9.3, 6.2))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), [ATTACK_LABELS.get(name, name) for name in table.index])
    ax.set_title("RTC-V3 mechanism activation during attack-active rounds")
    ax.set_xlabel("Mechanism metric")
    ax.set_ylabel("Attack")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1%}", ha="center", va="center", fontsize=7.5,
                        color="white" if value > 0.55 else "#212529")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Rate/share")
    path = output / "rtc_mechanism_activation.png"
    _save(fig, path)
    return path


def plot_update_reference_distances(distances: pd.DataFrame, output: Path) -> Path:
    table = distances.pivot_table(
        index="attack",
        columns="defense",
        values="mean_cosine_distance_to_clean",
        aggfunc="mean",
    ).reindex(index=CANONICAL_ATTACKS, columns=DEFAULT_DEFENSES)
    values = table.to_numpy(float)
    limit = max(0.1, float(np.nanmax(values)))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    image = ax.imshow(values, cmap="Oranges", vmin=0, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [DEFENSE_LABELS.get(x, x) for x in table.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), [ATTACK_LABELS.get(x, x) for x in table.index])
    ax.set_title("Aggregate-update cosine distance to plan-matched clean FedAvg")
    ax.set_xlabel("Defense")
    ax.set_ylabel("Attack")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if value > limit * 0.55 else "#212529")
    fig.colorbar(image, ax=ax, shrink=0.82).set_label("Cosine distance (0=same direction)")
    path = output / "aggregate_update_reference_distance.png"
    _save(fig, path)
    return path


def plot_timelines(summary: pd.DataFrame, round_files: Mapping[str, Path], output: Path) -> Path:
    attacked = summary.loc[summary["attack"].astype(str) != "none"]
    frames: list[pd.DataFrame] = []
    wanted = set(attacked.loc[attacked["defense"].isin(["fedavg", "rtc_full"]), "run_id"].astype(str))
    for run_id in wanted:
        frame = pd.read_csv(round_files[run_id])
        frame = frame.loc[pd.to_numeric(frame["round"], errors="coerce") > 0].copy()
        frames.append(frame)
    timeline = pd.concat(frames, ignore_index=True)
    timeline["round"] = pd.to_numeric(timeline["round"], errors="coerce")
    timeline["server_accuracy"] = pd.to_numeric(timeline["server_accuracy"], errors="coerce")
    timeline["server_asr"] = pd.to_numeric(timeline.get("server_asr"), errors="coerce")
    _style()
    path = output / "fedavg_vs_rtc_timelines.pdf"
    with PdfPages(path) as pdf:
        for attack in CANONICAL_ATTACKS:
            data = timeline.loc[timeline["attack"].astype(str) == attack]
            fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
            for defense in ("fedavg", "rtc_full"):
                subset = data.loc[data["defense"].astype(str) == defense]
                grouped = subset.groupby("round")
                color = DEFENSE_COLORS[defense]
                for ax, metric, label in (
                    (axes[0], "server_accuracy", "Main-task accuracy"),
                    (axes[1], "server_asr", "Attack success rate"),
                ):
                    mean = grouped[metric].mean()
                    low = grouped[metric].min()
                    high = grouped[metric].max()
                    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=DEFENSE_LABELS[defense])
                    ax.fill_between(mean.index, low.values, high.values, color=color, alpha=0.12)
                    ax.set_title(label)
                    ax.set_ylim(0, 1)
                    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                    ax.axvline(11, color="#5F6368", linestyle=":", linewidth=1)
                    ax.grid(color="#D9DEE3", linewidth=0.7)
                    ax.spines[["top", "right"]].set_visible(False)
            axes[0].legend(frameon=False)
            for ax in axes:
                ax.set_xlabel("Server round")
            fig.suptitle(f"{ATTACK_LABELS.get(attack, attack)} trajectories — seeds 42/43/44")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return path


def write_chart_map(paths: Iterable[Path], output: Path) -> Path:
    descriptions = {
        "paired_security_improvement_heatmap.png": "Attack×defense paired improvement over FedAvg; positive cells favor the defense.",
        "active_accuracy_drop_heatmap.png": "Attack×defense active-round utility degradation against the plan-matched clean FedAvg trajectory.",
        "aggregation_runtime.png": "Mean server aggregation cost and across-cell spread by defense.",
        "defense_assumption_violations.png": "Attack×defense rate of rounds exceeding nominal Krum/Trimmed-Mean malicious-count assumptions.",
        "rtc_mechanism_activation.png": "Attack×RTC mechanism heatmap for clipping, effective attacker weight, cone overflow, and semantic/time/exposure activation.",
        "aggregate_update_reference_distance.png": "Attack×defense mean cosine distance between the actual server update and the same-round plan-matched clean FedAvg update.",
        "fedavg_vs_rtc_timelines.pdf": "One page per attack with 60-round FedAvg/RTC-V3 accuracy and ASR trajectories; band spans the three seeds.",
    }
    rows = [
        {
            "artifact": path.name,
            "question": descriptions[path.name],
            "palette_policy": "hard two-root cap for comparisons; single-root sequential for magnitude",
            "source": "strict-gated formal round and summary CSVs",
        }
        for path in paths
    ]
    path = output / "chart_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_analysis(roots: Sequence[Path], freeze_path: Path, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    summary, round_files, freeze = load_and_merge(roots, freeze_path)
    matrix_gates = validate_complete_matrix(summary)
    matrix_gates.to_csv(output / "merged_quality_gates.csv", index=False)
    summary = periodic_attack.add_accuracy_drop(summary)
    # Recreate a single round directory so the existing paired AUC function can
    # operate across batch roots without silently missing counterfactual files.
    merged_rounds = output / "rounds"
    merged_rounds.mkdir(exist_ok=True)
    for run_id, source in round_files.items():
        target = merged_rounds / f"{run_id}.csv"
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    summary = periodic_attack.add_accuracy_drop_auc(summary, merged_rounds)
    summary.to_csv(output / "byzantine_attack_runs.csv", index=False)
    periodic_attack.bootstrap_summary(summary).to_csv(output / "byzantine_attack_summary.csv", index=False)
    effects = paired_effects(summary)
    effects.to_csv(output / "paired_defense_effects.csv", index=False)
    paired_rtc_statistics(summary).to_csv(
        output / "paired_rtc_statistics.csv", index=False
    )
    assumption_audit = defense_assumption_audit(summary, round_files)
    assumption_audit.to_csv(output / "defense_assumption_audit.csv", index=False)
    mechanisms = rtc_mechanism_summary(summary, round_files)
    mechanisms.to_csv(output / "rtc_mechanism_summary.csv", index=False)
    update_distances = aggregate_update_reference_distances(summary, round_files)
    update_distances.to_csv(
        output / "aggregate_update_reference_distances.csv", index=False
    )
    runtimes = runtime_summary(summary, round_files)
    runtimes.to_csv(output / "runtime_summary.csv", index=False)
    metric_availability = pd.DataFrame(
        [
            {
                "metric": "recovery_half_life",
                "available": False,
                "reason": "formal attack window is continuous through round 60; no post-attack recovery observations",
            },
            {
                "metric": "post_attack_residual_asr",
                "available": False,
                "reason": "formal attack window is continuous through round 60",
            },
            {
                "metric": "defense_specific_clean_utility_penalty",
                "available": False,
                "reason": "protocol runs one shared FedAvg-clean counterfactual, not per-defense clean trajectories",
            },
        ]
    )
    metric_availability.to_csv(output / "metric_availability.csv", index=False)
    (output / "analysis_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "RTCByzantineAnalysisV1",
                "roots": [str(path.resolve()) for path in roots],
                "attack_freeze": str(freeze_path.resolve()),
                "implementation_source_sha256": freeze["implementation_source_sha256"],
                "run_count": len(summary),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    _style()
    paths = [
        plot_effect_heatmap(effects, plots),
        plot_accuracy_heatmap(summary, plots),
        plot_runtime(runtimes, plots),
        plot_assumption_violations(assumption_audit, plots),
        plot_rtc_mechanisms(mechanisms, plots),
        plot_update_reference_distances(update_distances, plots),
        plot_timelines(summary, round_files, plots),
    ]
    chart_map = write_chart_map(paths, plots)
    evidence_paths = [
        output / "merged_quality_gates.csv",
        output / "byzantine_attack_runs.csv",
        output / "byzantine_attack_summary.csv",
        output / "paired_defense_effects.csv",
        output / "paired_rtc_statistics.csv",
        output / "defense_assumption_audit.csv",
        output / "rtc_mechanism_summary.csv",
        output / "aggregate_update_reference_distances.csv",
        output / "runtime_summary.csv",
        output / "metric_availability.csv",
        chart_map,
        *paths,
    ]
    attacked = summary.loc[summary["attack"].astype(str) != "none"]
    attestation = {
        "schema_version": "RTCByzantineCompletionAttestationV1",
        "status": "complete",
        "scope": {
            "partition": "iid",
            "attacks": list(CANONICAL_ATTACKS),
            "defenses": list(DEFAULT_DEFENSES),
            "seeds": [42, 43, 44],
            "malicious_fraction": 0.2,
            "rounds": 60,
        },
        "observed_attacked_runs": int(len(attacked)),
        "expected_attacked_runs": len(CANONICAL_ATTACKS) * len(DEFAULT_DEFENSES) * 3,
        "attack_freeze_sha256": _file_sha256(freeze_path),
        "implementation_source_sha256": freeze["implementation_source_sha256"],
        "evidence": {
            str(path.relative_to(output)): _file_sha256(path)
            for path in evidence_paths
        },
        "claim_boundary": (
            "cross-family Byzantine-defense evidence under the registered IID "
            "CIFAR-10/ResNet-18 conditions; not a universal Byzantine guarantee"
        ),
    }
    periodic_attack._atomic_write_json(
        attestation, output / "completion_attestation.json"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--attack-freeze", required=True)
    parser.add_argument("--output", default="logs/rtc_v3_byzantine_formal_merged")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    run_analysis(
        [Path(value) for value in cli.roots],
        Path(cli.attack_freeze),
        Path(cli.output),
    )
