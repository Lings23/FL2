"""Periodic-attack benchmark, ablations, statistics, and plots.

The full matrix is intentionally opt-in. Use ``--smoke`` for a 12-round
pipeline check and ``--mode all`` for the preregistered experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fedsec-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config, override_config
from main import run_simulation

logger = logging.getLogger(__name__)

TARGETED_ATTACKS = ("backdoor", "dba", "model_replacement")
UNTARGETED_ATTACKS = ("label_flip", "byzantine", "gaussian_noise")
PERIODS = {"short_1_1": (1, 1), "long_3_3": (3, 3)}
SEEDS = (42, 43, 44)
MALICIOUS_FRACTIONS = (0.2, 0.4)
BENCHMARK_VERSION = 2

DEFENSES: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "fedavg": ("none", {}),
    "freqfed": ("freqfed", {
        "min_cluster_size": 2,
        "min_samples": 1,
        "min_selected_clients": 2,
        "allow_single_cluster": True,
    }),
    "foolsgold": ("foolsgold", {}),
    "fltrust": ("fltrust", {}),
    "tc_full": ("time_consistency", {
        "enable_fft_features": True,
        "enable_direction_features": True,
        "periodic_entropy_threshold": 0.45,
        "periodic_freq_var_threshold": 0.02,
    }),
}
ABLATIONS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "tc_full": DEFENSES["tc_full"],
    "tc_no_fft": ("time_consistency", {
        "enable_fft_features": False,
        "enable_direction_features": True,
        "periodic_entropy_threshold": 0.45,
        "periodic_freq_var_threshold": 0.02,
    }),
    "tc_no_direction": ("time_consistency", {
        "enable_fft_features": True,
        "enable_direction_features": False,
        "periodic_entropy_threshold": 0.45,
        "periodic_freq_var_threshold": 0.02,
    }),
    "tc_neither": ("time_consistency", {
        "enable_fft_features": False,
        "enable_direction_features": False,
        "periodic_entropy_threshold": 0.45,
        "periodic_freq_var_threshold": 0.02,
    }),
}


def attack_active(round_number: int, start: int, on_rounds: int, off_rounds: int) -> bool:
    if round_number < start:
        return False
    cycle = max(1, on_rounds) + max(0, off_rounds)
    return ((round_number - start) % cycle) < max(1, on_rounds)


def build_matrix(mode: str = "all", smoke: bool = False) -> List[Dict[str, Any]]:
    if smoke:
        attacks = ("model_replacement",)
        periods = {"short_1_1": (1, 1)}
        fractions = (0.2,)
        seeds = (42,)
        defenses = {**DEFENSES, **ABLATIONS}
    elif mode == "main":
        attacks, periods, fractions, seeds, defenses = (
            TARGETED_ATTACKS, PERIODS, MALICIOUS_FRACTIONS, SEEDS, DEFENSES
        )
    elif mode == "ablation":
        attacks, periods, fractions, seeds, defenses = (
            TARGETED_ATTACKS, PERIODS, MALICIOUS_FRACTIONS, SEEDS, ABLATIONS
        )
    elif mode == "untargeted":
        attacks, periods, fractions, seeds, defenses = (
            UNTARGETED_ATTACKS, PERIODS, MALICIOUS_FRACTIONS, SEEDS, DEFENSES
        )
    elif mode == "all":
        return _deduplicate(build_matrix("main") + build_matrix("ablation") + build_matrix("untargeted"))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rows = []
    for attack, (period_name, period), fraction, seed, defense_name in itertools.product(
        attacks, periods.items(), fractions, seeds, defenses
    ):
        defense_type, custom = defenses[defense_name]
        rows.append({
            "benchmark_version": BENCHMARK_VERSION,
            "attack": attack,
            "period": period_name,
            "on_rounds": period[0],
            "off_rounds": period[1],
            "malicious_fraction": fraction,
            "seed": seed,
            "defense": defense_name,
            "defense_type": defense_type,
            "custom_params": dict(custom),
        })
    return rows


def _deduplicate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for row in rows:
        key = tuple((name, json.dumps(value, sort_keys=True)) for name, value in sorted(row.items()))
        unique[key] = row
    return list(unique.values())


def run_id(spec: Dict[str, Any]) -> str:
    readable = (
        f"{spec['attack']}__{spec['period']}__m{spec['malicious_fraction']:.1f}__"
        f"s{spec['seed']}__{spec['defense']}"
    )
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{readable}__{digest}"


def tracker_to_rounds(records: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> pd.DataFrame:
    server = {
        int(row["round"]): row for row in records if row.get("split") == "server"
    }
    fit = {int(row["round"]): row for row in records if row.get("split") == "fit"}
    rounds = sorted(set(server) | set(fit))
    rows = []
    for rnd in rounds:
        combined: Dict[str, Any] = {
            "round": rnd,
            "planned_attack_active": float(
                spec["attack"] != "none" and attack_active(
                    rnd, int(spec.get("attack_start_round", 11)),
                    int(spec["on_rounds"]), int(spec["off_rounds"]),
                )
            ),
        }
        for prefix, source in (("server_", server.get(rnd, {})), ("fit_", fit.get(rnd, {}))):
            for key, value in source.items():
                if key not in {"round", "split"}:
                    combined[prefix + key] = value
        combined.update({key: value for key, value in spec.items() if key != "custom_params"})
        rows.append(combined)
    return pd.DataFrame(rows)


def summarize_run(rounds: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    result = {key: value for key, value in spec.items() if key != "custom_params"}
    result["run_id"] = run_id(spec)
    active = rounds[rounds["planned_attack_active"] == 1]
    inactive = rounds[
        (rounds["planned_attack_active"] == 0)
        & (rounds["round"] >= int(spec.get("attack_start_round", 11)))
    ]
    accuracy_col = "server_accuracy"
    asr_col = "server_asr"
    if "server_loss" in rounds:
        losses = pd.to_numeric(rounds["server_loss"], errors="coerce")
        result["numerical_divergence"] = bool((~np.isfinite(losses)).any())
    if spec["attack"] in TARGETED_ATTACKS and asr_col in rounds:
        active_asr = pd.to_numeric(active[asr_col], errors="coerce").dropna()
        inactive_asr = pd.to_numeric(inactive[asr_col], errors="coerce").dropna()
        result.update({
            "active_asr": float(active_asr.mean()) if len(active_asr) else math.nan,
            "active_asr_auc": _auc(active_asr),
            "peak_asr": float(pd.to_numeric(rounds[asr_col], errors="coerce").max()),
            "residual_asr": float(inactive_asr.mean()) if len(inactive_asr) else math.nan,
        })
    if accuracy_col in rounds:
        accuracy = pd.to_numeric(rounds[accuracy_col], errors="coerce")
        active_accuracy = pd.to_numeric(active[accuracy_col], errors="coerce").dropna()
        result.update({
            "final_accuracy": float(accuracy.dropna().iloc[-1]) if accuracy.notna().any() else math.nan,
            "min_accuracy": float(accuracy.min()),
            "active_accuracy": float(active_accuracy.mean()) if len(active_accuracy) else math.nan,
        })
    result["detection_lag"] = detection_lag(rounds)
    for column in (
        "fit_malicious_aggregation_weight_share", "fit_malicious_impact_share",
        "fit_benign_quarantine_rate", "fit_aggregation_time_seconds",
        "fit_fft_periodic_penalty_clients", "fit_direction_penalty_clients",
        "fit_freqfed_fallback",
    ):
        if column in rounds:
            result[column.removeprefix("fit_")] = float(
                pd.to_numeric(rounds[column], errors="coerce").mean()
            )
    return result


def _auc(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    if not len(array):
        return math.nan
    if len(array) == 1:
        return float(array[0])
    return float(np.trapezoid(array, dx=1.0))


def detection_lag(rounds: pd.DataFrame, threshold: float = 0.05) -> float:
    weight_col = "fit_active_attacker_weight_share"
    selected_col = "fit_selected_active_attackers"
    if weight_col not in rounds:
        return math.nan
    lags = []
    previous_active = False
    segment_start = None
    for _, row in rounds.sort_values("round").iterrows():
        active = bool(row["planned_attack_active"])
        if active and not previous_active:
            segment_start = int(row["round"])
        if active and segment_start is not None:
            selected = float(row.get(selected_col, 1) or 0)
            weight = float(row.get(weight_col, math.nan))
            if selected > 0 and np.isfinite(weight) and weight < threshold:
                lags.append(int(row["round"]) - segment_start)
                segment_start = None
        if not active:
            segment_start = None
        previous_active = active
    return float(np.mean(lags)) if lags else math.nan


def add_accuracy_drop(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    clean = summary[summary["attack"] == "none"] if "attack" in summary else pd.DataFrame()
    if clean.empty:
        summary["accuracy_drop"] = np.nan
        return summary
    baselines = clean[clean["defense"] == "fedavg"].set_index("seed")["final_accuracy"]
    summary["accuracy_drop"] = summary.apply(
        lambda row: float(baselines.get(row["seed"], np.nan) - row.get("final_accuracy", np.nan)),
        axis=1,
    )
    return summary


def add_accuracy_drop_auc(summary: pd.DataFrame, rounds_dir: Path) -> pd.DataFrame:
    """Compare every attacked run with the same-seed clean FedAvg trajectory."""
    summary = summary.copy()
    summary["active_accuracy_drop"] = np.nan
    summary["accuracy_drop_auc"] = np.nan
    frames = []
    expected_ids = set(summary["run_id"].astype(str)) if "run_id" in summary else None
    for path in rounds_dir.glob("*.csv"):
        if expected_ids is not None and path.stem not in expected_ids:
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    clean = {}
    attacked = {}
    for frame in frames:
        first = frame.iloc[0]
        if first.get("attack") == "none" and first.get("defense") == "fedavg":
            clean[int(first["seed"])] = frame
        else:
            key = (str(first.get("attack")), str(first.get("period")),
                   float(first.get("malicious_fraction")), int(first.get("seed")),
                   str(first.get("defense")))
            attacked[key] = frame
    for index, row in summary.iterrows():
        if row["attack"] == "none" or int(row["seed"]) not in clean:
            continue
        key = (str(row["attack"]), str(row["period"]), float(row["malicious_fraction"]),
               int(row["seed"]), str(row["defense"]))
        attack_frame = attacked.get(key)
        if attack_frame is None or "server_accuracy" not in attack_frame:
            continue
        merged = attack_frame[["round", "planned_attack_active", "server_accuracy"]].merge(
            clean[int(row["seed"])][["round", "server_accuracy"]],
            on="round", suffixes=("_attack", "_clean"),
        )
        active = merged[merged["planned_attack_active"] == 1]
        drops = (
            pd.to_numeric(active["server_accuracy_clean"], errors="coerce")
            - pd.to_numeric(active["server_accuracy_attack"], errors="coerce")
        ).dropna()
        if len(drops):
            summary.at[index, "active_accuracy_drop"] = float(drops.mean())
            summary.at[index, "accuracy_drop_auc"] = _auc(drops)
    return summary


def bootstrap_summary(summary: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    metrics = [
        name for name in ("active_asr", "active_asr_auc", "peak_asr", "residual_asr",
                          "final_accuracy", "min_accuracy", "active_accuracy", "detection_lag",
                          "accuracy_drop", "active_accuracy_drop", "accuracy_drop_auc",
                          "malicious_aggregation_weight_share",
                          "benign_quarantine_rate", "aggregation_time_seconds")
        if name in summary
    ]
    keys = ["attack", "period", "malicious_fraction", "defense"]
    rng = np.random.default_rng(seed)
    rows = []
    for group_key, group in summary.groupby(keys, dropna=False):
        base = dict(zip(keys, group_key))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            boot = np.asarray([
                np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(2000)
            ])
            rows.append({**base, "metric": metric, "mean": float(np.mean(values)),
                         "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                         "count": len(values), "ci_low": float(np.quantile(boot, 0.025)),
                         "ci_high": float(np.quantile(boot, 0.975))})
    return pd.DataFrame(rows)


def statistical_tests(summary: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    comparisons = ["freqfed", "foolsgold", "fltrust", "tc_no_fft", "tc_no_direction", "tc_neither"]
    rows = []
    for attack in TARGETED_ATTACKS:
        attack_rows = summary[summary["attack"] == attack]
        for comparator in comparisons:
            paired = _paired_values(attack_rows, "tc_full", comparator, "active_asr_auc")
            if paired.empty:
                continue
            differences = paired["left"] - paired["right"]
            descriptive_only = len(paired) < 8
            rows.append({"attack": attack, "metric": "active_asr_auc", "left": "tc_full",
                         "right": comparator, "n": len(paired),
                         "descriptive_only": descriptive_only,
                         "median_difference": float(np.median(differences)),
                         "cliffs_delta": cliffs_delta(paired["left"], paired["right"]),
                         "p_value": (
                             math.nan if descriptive_only
                             else paired_permutation_pvalue(differences.to_numpy(), seed)
                         )})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = np.nan
        valid = result["p_value"].notna()
        if valid.any():
            result.loc[valid, "p_holm"] = holm_adjust(
                result.loc[valid, "p_value"].to_numpy()
            )
    return result


def _paired_values(data: pd.DataFrame, left: str, right: str, metric: str) -> pd.DataFrame:
    keys = ["period", "malicious_fraction", "seed"]
    pivot = data[data["defense"].isin([left, right])].pivot_table(
        index=keys, columns="defense", values=metric, aggfunc="first"
    )
    if left not in pivot or right not in pivot:
        return pd.DataFrame(columns=["left", "right"])
    return pivot[[left, right]].dropna().rename(columns={left: "left", right: "right"})


def paired_permutation_pvalue(differences: np.ndarray, seed: int, samples: int = 20000) -> float:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return math.nan
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(samples, len(differences)))
    permuted = np.abs(np.mean(signs * differences, axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (samples + 1))


def cliffs_delta(left: Iterable[float], right: Iterable[float]) -> float:
    left_arr, right_arr = np.asarray(list(left)), np.asarray(list(right))
    if not len(left_arr) or not len(right_arr):
        return math.nan
    comparisons = left_arr[:, None] - right_arr[None, :]
    return float((np.count_nonzero(comparisons > 0) - np.count_nonzero(comparisons < 0)) / comparisons.size)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted


def run_experiments(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output)
    rounds_dir = output / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(args.mode, args.smoke)
    clean = clean_baselines(matrix, args.smoke)
    matrix = _deduplicate(matrix + clean)
    write_experiment_manifest(matrix, output)

    if args.smoke:
        execution_order = matrix
    else:
        pilot_attacks = [
            spec for spec in matrix
            if spec["attack"] != "none" and spec["defense"] == "fedavg"
            and spec["seed"] == 42 and spec["period"] == "short_1_1"
            and spec["malicious_fraction"] == 0.2
        ]
        pilot_clean = [
            spec for spec in clean
            if spec["seed"] == 42 and spec["defense"] in {"fedavg", "freqfed", "fltrust"}
        ]
        pilots = _deduplicate(pilot_clean + pilot_attacks)
        pilot_summary = _run_specs(pilots, args, output, rounds_dir)
        pilot_summary = add_accuracy_drop_auc(add_accuracy_drop(pilot_summary), rounds_dir)
        gates = validate_formal_preflight(pilot_summary)
        write_quality_gates(gates, output / "preflight_gates.csv")
        failed = [gate for gate in gates if not gate["passed"]]
        excluded_attacks = {
            gate["gate"].split(":", 1)[1]
            for gate in failed if gate["gate"].startswith("attack_strength:")
        }
        excluded_defenses = set()
        if any(gate["gate"] == "freqfed_clean_fallback" for gate in failed):
            excluded_defenses.add("freqfed")
        if any(gate["gate"] == "fltrust_clean_utility" for gate in failed):
            excluded_defenses.add("fltrust")
        if failed:
            logger.warning(
                "Preflight exclusions | attacks=%s defenses=%s",
                sorted(excluded_attacks), sorted(excluded_defenses),
            )
        pilot_ids = {run_id(spec) for spec in pilots}
        eligible = [
            spec for spec in matrix
            if spec["attack"] not in excluded_attacks
            and spec["defense"] not in excluded_defenses
        ]
        attacked_eligible = [spec for spec in eligible if spec["attack"] != "none"]
        if not attacked_eligible:
            raise RuntimeError("All attack families failed preflight; formal matrix was not started")
        execution_order = pilots + [spec for spec in eligible if run_id(spec) not in pilot_ids]

    summary = _run_specs(execution_order, args, output, rounds_dir)
    summary = add_accuracy_drop_auc(add_accuracy_drop(summary), rounds_dir)
    expected_ids = set(summary["run_id"].astype(str))
    schedule_gates = validate_sampling_manifests(
        rounds_dir, output / "raw", expected_ids=expected_ids
    )
    if args.smoke:
        quality_gates = schedule_gates + validate_smoke(summary, rounds_dir, output / "raw")
    else:
        quality_gates = schedule_gates
    write_quality_gates(quality_gates, output / "quality_gates.csv")
    failed = [gate for gate in quality_gates if not gate["passed"]]
    if failed:
        raise RuntimeError(
            "Benchmark quality gates failed: " + "; ".join(gate["gate"] for gate in failed)
        )

    summary.to_csv(output / "periodic_attack_runs.csv", index=False)
    bootstrap_summary(summary).to_csv(output / "periodic_attack_summary.csv", index=False)
    statistical_tests(summary).to_csv(output / "statistical_tests.csv", index=False)
    generate_plots(summary, rounds_dir, output / "plots")
    return summary


def _run_specs(
    specs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output: Path,
    rounds_dir: Path,
) -> pd.DataFrame:
    summaries = []
    for index, base_spec in enumerate(specs, 1):
        spec = dict(base_spec)
        spec = {**spec, "attack_start_round": 3 if args.smoke else 11}
        identifier = run_id(spec)
        round_path = rounds_dir / f"{identifier}.csv"
        logger.info("[%d/%d] %s", index, len(specs), identifier)
        if round_path.exists() and not args.rerun:
            rounds = pd.read_csv(round_path)
        else:
            cfg = load_config(args.config)
            custom = dict(cfg.security.defense.custom_params or {})
            custom.update(spec["custom_params"])
            overrides = {
                "project.seed": spec["seed"],
                "project.log_dir": str(output / "raw"),
                "dataset.name": "cifar10",
                "model.architecture": "resnet18",
                "federation.num_rounds": 12 if args.smoke else 60,
                "federation.num_clients": 20,
                "federation.clients_per_round": 20 if args.smoke else 10,
                "federation.min_fit_clients": 20 if args.smoke else 10,
                "federation.min_available_clients": 20,
                "federation.max_client_samples": 100 if args.smoke else 0,
                "client.local_epochs": 1 if args.smoke else cfg.client.local_epochs,
                "strategy.name": "fedavg",
                "security.attack.enabled": spec["attack"] != "none",
                "security.attack.type": spec["attack"],
                "security.attack.malicious_fraction": spec["malicious_fraction"],
                "security.attack.attack_start_round": spec["attack_start_round"],
                "security.attack.attack_on_rounds": spec["on_rounds"],
                "security.attack.attack_off_rounds": spec["off_rounds"],
                "security.attack.poison_fraction": (
                    0.5 if spec["attack"] in {"backdoor", "dba"} else 0.1
                ),
                "security.attack.dba_boost_factor": 10.0,
                "security.attack.model_replacement_boost_factor": 10.0,
                "security.defense.enabled": spec["defense_type"] != "none",
                "security.defense.type": spec["defense_type"],
                "security.defense.reserve_root_for_all": True,
                "security.defense.root_dataset_size": 100,
                "security.defense.custom_params": custom,
                "evaluation.save_best_model": False,
                "evaluation.max_test_samples": 1000 if args.smoke else 0,
            }
            cfg = override_config(cfg, overrides)
            tracker = run_simulation(cfg, experiment_name=identifier)
            rounds = tracker_to_rounds(tracker.to_list(), spec)
            rounds.to_csv(round_path, index=False)
        summaries.append(summarize_run(rounds, spec))
        pd.DataFrame(summaries).to_csv(output / "periodic_attack_runs.csv", index=False)
    return pd.DataFrame(summaries)


def write_experiment_manifest(specs: Sequence[Dict[str, Any]], output: Path) -> None:
    payload = {"benchmark_version": BENCHMARK_VERSION, "specs": list(specs)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "experiment_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _gate(name: str, passed: bool, observed: Any, required: str) -> Dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "observed": observed, "required": required}


def write_quality_gates(gates: Sequence[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame(gates).to_csv(path, index=False)


def validate_sampling_manifests(
    rounds_dir: Path,
    raw_dir: Path,
    expected_ids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    paths = list(rounds_dir.glob("*.csv"))
    if expected_ids is not None:
        paths = [path for path in paths if path.stem in expected_ids]
    frames = [pd.read_csv(path) for path in paths]
    gates: List[Dict[str, Any]] = []
    per_run_match = True
    schedules: Dict[Tuple[Any, ...], Dict[int, str]] = {}
    for frame in frames:
        if frame.empty or "fit_selected_partition_ids" not in frame:
            per_run_match = False
            continue
        fit = frame.dropna(subset=["fit_selected_partition_ids"])
        if "fit_planned_partition_ids" not in fit:
            per_run_match = False
            continue
        per_run_match &= bool((
            fit["fit_selected_partition_ids"].astype(str)
            == fit["fit_planned_partition_ids"].astype(str)
        ).all())
        first = frame.iloc[0]
        key = (first.get("attack"), first.get("period"), first.get("malicious_fraction"), first.get("seed"))
        schedule = dict(zip(fit["round"].astype(int), fit["fit_selected_partition_ids"].astype(str)))
        previous = schedules.setdefault(key, schedule)
        if previous != schedule:
            per_run_match = False
    gates.append(_gate("selected_partition_ids_match", per_run_match, per_run_match, "true"))

    hashes_by_seed: Dict[int, set[str]] = {}
    manifests = list(raw_dir.glob("*_data_manifest.json"))
    if expected_ids is not None:
        manifests = [
            path for path in manifests
            if path.name.removesuffix("_data_manifest.json") in expected_ids
        ]
    for path in manifests:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        hashes_by_seed.setdefault(int(data["seed"]), set()).add(str(data["sha256"]))
    data_match = bool(manifests) and all(len(values) == 1 for values in hashes_by_seed.values())
    gates.append(_gate("data_manifest_hash_match", data_match,
                       {seed: len(values) for seed, values in hashes_by_seed.items()},
                       "one hash per seed"))
    return gates


def validate_smoke(summary: pd.DataFrame, rounds_dir: Path, raw_dir: Path) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    attacked = summary[summary["attack"] == "model_replacement"]
    fedavg = attacked[attacked["defense"] == "fedavg"].iloc[0]
    gates.append(_gate("fedavg_active_asr", fedavg["active_asr"] >= 0.30,
                       fedavg["active_asr"], ">= 0.30"))
    gates.append(_gate("fedavg_peak_asr", fedavg["peak_asr"] >= 0.50,
                       fedavg["peak_asr"], ">= 0.50"))

    frames = {}
    expected_ids = set(summary["run_id"].astype(str))
    critical_complete = True
    for path in rounds_dir.glob("*.csv"):
        if path.stem not in expected_ids:
            continue
        frame = pd.read_csv(path)
        if not frame.empty and frame.iloc[0].get("attack") == "model_replacement":
            frames[str(frame.iloc[0]["defense"])] = frame
        expected_rounds = set(range(13))
        critical_complete &= set(frame["round"].astype(int)) == expected_rounds
        critical_complete &= pd.to_numeric(
            frame["server_accuracy"], errors="coerce"
        ).notna().all()
        if frame.iloc[0].get("attack") == "model_replacement":
            critical_complete &= pd.to_numeric(
                frame["server_asr"], errors="coerce"
            ).notna().all()
    gates.append(_gate("critical_metrics_complete", critical_complete,
                       critical_complete, "rounds 0-12 with finite accuracy/ASR"))
    full_fft = pd.to_numeric(frames["tc_full"]["fit_fft_periodic_penalty_clients"], errors="coerce").fillna(0).sum()
    no_fft = frames["tc_no_fft"]
    no_fft_sum = pd.to_numeric(no_fft["fit_fft_periodic_penalty_clients"], errors="coerce").fillna(0).sum()
    no_fft_features = sum(
        pd.to_numeric(no_fft[column], errors="coerce").fillna(0).abs().sum()
        for column in ("fit_mean_low_frequency_power", "fit_mean_dominant_frequency_energy")
    )
    no_direction_sum = pd.to_numeric(
        frames["tc_no_direction"]["fit_direction_penalty_clients"], errors="coerce"
    ).fillna(0).sum()
    no_direction_features = pd.to_numeric(
        frames["tc_no_direction"]["fit_mean_direction_anomaly"], errors="coerce"
    ).fillna(0).abs().sum()
    gates.append(_gate("tc_full_fft_triggered", full_fft > 0, full_fft, "> 0"))
    gates.append(_gate("tc_no_fft_is_zero", no_fft_sum == 0 and no_fft_features == 0,
                       {"triggers": no_fft_sum, "features": no_fft_features}, "both zero"))
    gates.append(_gate("tc_no_direction_is_zero",
                       no_direction_sum == 0 and no_direction_features == 0,
                       {"triggers": no_direction_sum, "features": no_direction_features},
                       "both zero"))

    freq = attacked[attacked["defense"] == "freqfed"].iloc[0]
    fallback = float(freq.get("freqfed_fallback", math.nan))
    gates.append(_gate("freqfed_fallback_rate", np.isfinite(fallback) and fallback <= 0.25,
                       fallback, "<= 0.25"))
    expected_ids = [str(row["run_id"]) for _, row in summary.iterrows()]
    unique_logs = all(
        (raw_dir / f"{identifier}.csv").exists() and (raw_dir / f"{identifier}.json").exists()
        for identifier in expected_ids
    )
    gates.append(_gate("unique_metric_files", unique_logs, unique_logs, "true"))
    return gates


def validate_formal_preflight(summary: pd.DataFrame) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    for _, row in summary[(summary["defense"] == "fedavg") & (summary["attack"].isin(TARGETED_ATTACKS))].iterrows():
        passed = row.get("active_asr", 0) >= 0.20 and row.get("peak_asr", 0) >= 0.30
        gates.append(_gate(f"attack_strength:{row['attack']}", passed,
                           {"active_asr": row.get("active_asr"), "peak_asr": row.get("peak_asr")},
                           "active_asr>=0.20 and peak_asr>=0.30"))
    for _, row in summary[(summary["defense"] == "fedavg") & (summary["attack"].isin(UNTARGETED_ATTACKS))].iterrows():
        drop = float(row.get("active_accuracy_drop", math.nan))
        gates.append(_gate(f"attack_strength:{row['attack']}", np.isfinite(drop) and drop >= 0.05,
                           drop, ">= 0.05 active accuracy drop"))
    clean = summary[summary["attack"] == "none"]
    if "freqfed" in set(clean["defense"]):
        fallback = float(clean[clean["defense"] == "freqfed"].iloc[0].get("freqfed_fallback", math.nan))
        gates.append(_gate("freqfed_clean_fallback", np.isfinite(fallback) and fallback <= 0.25,
                           fallback, "<= 0.25"))
    if {"fedavg", "fltrust"} <= set(clean["defense"]):
        fedavg_acc = float(clean[clean["defense"] == "fedavg"].iloc[0]["final_accuracy"])
        fltrust_acc = float(clean[clean["defense"] == "fltrust"].iloc[0]["final_accuracy"])
        gates.append(_gate("fltrust_clean_utility", fedavg_acc - fltrust_acc <= 0.05,
                           fedavg_acc - fltrust_acc, "<= 0.05 accuracy loss"))
    return gates


def clean_baselines(matrix: Sequence[Dict[str, Any]], smoke: bool) -> List[Dict[str, Any]]:
    """Add same-seed clean trajectories for utility and accuracy-drop metrics."""
    seeds = sorted({int(spec["seed"]) for spec in matrix})
    defense_names = sorted({str(spec["defense"]) for spec in matrix})
    if smoke:
        defense_names = [name for name in defense_names if name in {"fedavg", "tc_full"}]
    rows = []
    definitions = {**DEFENSES, **ABLATIONS}
    for seed, defense_name in itertools.product(seeds, defense_names):
        defense_type, custom = definitions[defense_name]
        rows.append({"attack": "none", "period": "clean", "on_rounds": 1,
                     "off_rounds": 0, "malicious_fraction": 0.0, "seed": seed,
                     "benchmark_version": BENCHMARK_VERSION,
                     "defense": defense_name, "defense_type": defense_type,
                     "custom_params": dict(custom)})
    return rows


def generate_plots(summary: pd.DataFrame, rounds_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    if "active_asr_auc" in summary and summary["active_asr_auc"].notna().any():
        plt.figure(figsize=(9, 5))
        sns.barplot(data=summary, x="defense", y="active_asr_auc", hue="attack", errorbar="sd")
        plt.xticks(rotation=25); plt.tight_layout()
        plt.savefig(output / "active_asr_auc.png", dpi=180); plt.close()
    if "final_accuracy" in summary:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=summary, x="active_asr_auc", y="final_accuracy", hue="defense", style="attack")
        plt.tight_layout(); plt.savefig(output / "security_utility_tradeoff.png", dpi=180); plt.close()
    ablation = summary[summary["defense"].isin(ABLATIONS)]
    if not ablation.empty and "active_asr_auc" in ablation:
        plt.figure(figsize=(8, 5))
        sns.pointplot(data=ablation, x="defense", y="active_asr_auc", hue="period", errorbar="sd")
        plt.xticks(rotation=20); plt.tight_layout()
        plt.savefig(output / "ablation_effects.png", dpi=180); plt.close()

    timeline_frames = []
    expected_ids = set(summary["run_id"].astype(str)) if "run_id" in summary else None
    for path in rounds_dir.glob("*.csv"):
        if expected_ids is not None and path.stem not in expected_ids:
            continue
        frame = pd.read_csv(path)
        if frame.get("defense", pd.Series(dtype=str)).eq("tc_full").any():
            timeline_frames.append(frame)
    if timeline_frames:
        timeline = pd.concat(timeline_frames, ignore_index=True)
        columns = [name for name in (
            "server_asr", "fit_active_attacker_weight_share",
            "fit_fft_periodic_penalty_clients", "fit_direction_penalty_clients"
        ) if name in timeline]
        for metric in columns:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=timeline, x="round", y=metric, hue="attack", style="period", errorbar="sd")
            plt.tight_layout(); plt.savefig(output / f"timeline_{metric}.png", dpi=180); plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic attack defense benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="logs/periodic_attack_v2")
    parser.add_argument("--mode", choices=("main", "ablation", "untargeted", "all"), default="all")
    parser.add_argument("--smoke", action="store_true", help="Run a 12-round single-condition matrix")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiments(parse_args())
