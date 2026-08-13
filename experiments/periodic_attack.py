"""Periodic-attack benchmark, ablations, statistics, and plots.

The full matrix is intentionally opt-in. Use ``--smoke`` for a 6-round
pipeline check and ``--mode all`` for the preregistered experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import multiprocessing
import os
import platform
import re
import sys
import tempfile
import traceback
from datetime import datetime, timezone
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
from experiments.trial_plan import (
    TrialPlanV1,
    attach_trial_plans,
    paired_fedavg_clean_specs,
)
from main import (
    RayResourcePreflightError,
    run_simulation,
    shutdown_ray_runtime,
    wait_for_available_memory,
)

logger = logging.getLogger(__name__)

LABEL_FLIP_TARGETED = "label_flip_targeted"
LABEL_FLIP_ALL_REVERSE = "label_flip_all_reverse"
LABEL_FLIP_ATTACKS = {LABEL_FLIP_TARGETED, LABEL_FLIP_ALL_REVERSE}
TARGETED_ATTACKS = ("backdoor", "dba", "model_replacement", LABEL_FLIP_TARGETED)
UNTARGETED_ATTACKS = (LABEL_FLIP_ALL_REVERSE, "byzantine", "gaussian_noise")
PERIODS = {
    "continuous_1_0": (1, 0),
    "short_1_1": (1, 1),
    "long_3_3": (3, 3),
}
# Keep the preregistered/default benchmark unchanged.  ``continuous_1_0`` is
# an explicit attack-strength validation schedule selected through the focused
# RTC/FedAvg profile, not an extra cell silently added to every legacy matrix.
BENCHMARK_PERIODS = {
    name: PERIODS[name] for name in ("short_1_1", "long_3_3")
}
SEEDS = (42, 43, 44)
MALICIOUS_FRACTIONS = (0.2, 0.4)
BENCHMARK_VERSION = 2
CONDITION_KEYS = ("partition", "dirichlet_alpha", "participation_rate", "boost_factor")

RTC_V2_BASE: Dict[str, Any] = {
    "enable_temporal_features": True,
    "enable_direction_features": True,
    "enable_influence_features": True,
    "trust_beta": 0.75,
    "watch_threshold": 0.55,
    "restricted_threshold": 0.70,
    "quarantine_threshold": 0.80,
    "immediate_quarantine_threshold": 0.95,
    "quarantine_window": 4,
    "quarantine_strikes": 2,
    "recovery_threshold": 0.35,
    "recovery_observations": 3,
    "state_weight_cap_multipliers": {
        "normal": 2.0,
        "watch": 1.0,
        "restricted": 0.1,
        "quarantined": 0.0,
    },
    "enable_exposure_budgets": True,
    "exposure_window": 4,
    "client_exposure_budget_multiplier": 4.0,
    "direction_exposure_budget_multiplier": 4.0,
    "clip_multiplier": 1.0,
}

DEFENSES: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "fedavg": ("none", {}),
    "clip_only": ("time_consistency", {
        "enable_soft_trust_weighting": False,
        "enable_delta_clipping": True,
        "enable_trust_caps": False,
        "enable_exposure_budgets": False,
        "enable_temporal_features": False,
        "enable_direction_features": False,
        "enable_influence_features": False,
    }),
    "freqfed": ("freqfed", {
        "min_cluster_size": 2,
        "min_samples": 1,
        "min_selected_clients": 2,
        "allow_single_cluster": True,
    }),
    "foolsgold": ("foolsgold", {}),
    "fltrust": ("fltrust", {}),
    "rtc_full": ("time_consistency", dict(RTC_V2_BASE)),
}
ABLATIONS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "rtc_full": DEFENSES["rtc_full"],
    "rtc_no_temporal": ("time_consistency", {
        **RTC_V2_BASE,
        "enable_temporal_features": False,
    }),
    "rtc_no_direction": ("time_consistency", {
        **RTC_V2_BASE,
        "enable_direction_features": False,
    }),
    "rtc_no_influence": ("time_consistency", {
        **RTC_V2_BASE,
        "enable_influence_features": False,
    }),
    "rtc_trust_only": ("time_consistency", {
        **RTC_V2_BASE,
        "enable_delta_clipping": False,
        "enable_trust_caps": False,
        "enable_exposure_budgets": False,
    }),
}


def attack_active(
    round_number: int,
    start: int,
    on_rounds: int,
    off_rounds: int,
    end: int = -1,
) -> bool:
    if round_number < start or (end >= 0 and round_number > end):
        return False
    cycle = max(1, on_rounds) + max(0, off_rounds)
    return ((round_number - start) % cycle) < max(1, on_rounds)


def build_matrix(mode: str = "all", smoke: bool = False) -> List[Dict[str, Any]]:
    if smoke:
        attacks = ("model_replacement",)
        periods = {"short_1_1": (1, 1)}
        fractions = (0.2,)
        seeds = (42,)
        smoke_names = {"fedavg", "freqfed", "clip_only", "rtc_full", "rtc_no_temporal", "rtc_no_direction"}
        definitions = {**DEFENSES, **ABLATIONS}
        defenses = {name: definitions[name] for name in smoke_names}
    elif mode == "main":
        attacks, periods, fractions, seeds, defenses = (
            TARGETED_ATTACKS, BENCHMARK_PERIODS, MALICIOUS_FRACTIONS, SEEDS, DEFENSES
        )
    elif mode == "ablation":
        attacks, periods, fractions, seeds, defenses = (
            (*TARGETED_ATTACKS, LABEL_FLIP_ALL_REVERSE),
            BENCHMARK_PERIODS,
            MALICIOUS_FRACTIONS,
            SEEDS,
            ABLATIONS,
        )
    elif mode == "untargeted":
        attacks, periods, fractions, seeds, defenses = (
            UNTARGETED_ATTACKS, BENCHMARK_PERIODS, MALICIOUS_FRACTIONS, SEEDS, DEFENSES
        )
    elif mode == "mpaf":
        attacks, periods, fractions, seeds, defenses = (
            ("mpaf",), BENCHMARK_PERIODS, MALICIOUS_FRACTIONS, SEEDS, DEFENSES
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
    canonical_spec = dict(spec)
    # The content hash identifies a strict plan; its output-directory-specific
    # path must not change the run identity.
    canonical_spec.pop("trial_plan_path", None)
    if canonical_spec.get("attack") in LABEL_FLIP_ATTACKS:
        canonical_spec.pop("boost_factor", None)
    if canonical_spec.get("attack") == LABEL_FLIP_ALL_REVERSE:
        canonical_spec.pop("label_flip_source_label", None)
        canonical_spec.pop("label_flip_target_label", None)
    readable = (
        f"{spec['attack']}__{spec['period']}__m{spec['malicious_fraction']:.1f}__"
        f"s{spec['seed']}__{spec['defense']}__{spec.get('partition', 'iid')}__"
        f"p{spec.get('participation_rate', 0.5):.2f}"
    )
    if spec["attack"] == LABEL_FLIP_TARGETED:
        readable += (
            f"__lf{int(spec.get('label_flip_source_label', 5))}to"
            f"{int(spec.get('label_flip_target_label', 3))}-p"
            f"{float(spec.get('label_flip_poison_fraction', 1.0)):g}"
        )
    elif spec["attack"] == LABEL_FLIP_ALL_REVERSE:
        readable += (
            f"__lfrev-p{float(spec.get('label_flip_poison_fraction', 1.0)):g}"
        )
    else:
        readable += f"__b{spec.get('boost_factor', 10.0):g}"
    digest = hashlib.sha1(
        json.dumps(canonical_spec, sort_keys=True).encode()
    ).hexdigest()[:8]
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
                    int(spec.get("attack_end_round", -1)),
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


def _reverse_mapping_asr_from_confusion(value: Any) -> float:
    """Recover y -> C-1-y ASR from a serialized confusion matrix.

    This keeps complete pre-column all-reverse runs reusable without weakening
    the metric gate: the exact objective is reconstructed from immutable
    per-round evidence and malformed/non-finite inputs still fail closed.
    """
    matrix = np.asarray(json.loads(str(value)), dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"invalid all-reverse confusion-matrix shape: {matrix.shape}")
    total = int(matrix.sum())
    if total <= 0:
        raise ValueError("all-reverse confusion matrix has no observations")
    labels = np.arange(matrix.shape[0])
    success = int(matrix[labels, matrix.shape[0] - 1 - labels].sum())
    return float(success / total)


def _objective_asr_series(rounds: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series | None:
    """Return reported ASR, or exact legacy all-reverse ASR when recoverable."""
    if "server_asr" in rounds:
        reported = pd.to_numeric(rounds["server_asr"], errors="coerce")
        if reported.notna().all() and np.isfinite(reported.to_numpy(dtype=float)).all():
            return reported
    if (
        str(spec.get("attack")) == LABEL_FLIP_ALL_REVERSE
        and "server_confusion_matrix_json" in rounds
    ):
        try:
            derived = rounds["server_confusion_matrix_json"].map(
                _reverse_mapping_asr_from_confusion
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if derived.notna().all() and np.isfinite(derived.to_numpy(dtype=float)).all():
            return derived.astype(float)
    return None


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
    objective_asr = _objective_asr_series(rounds, spec)
    if (
        spec["attack"] in TARGETED_ATTACKS
        or spec["attack"] == LABEL_FLIP_ALL_REVERSE
    ) and objective_asr is not None:
        active_asr = objective_asr.loc[active.index].dropna()
        inactive_asr = objective_asr.loc[inactive.index].dropna()
        raw_auc = _auc(active_asr)
        result.update({
            "active_asr": float(active_asr.mean()) if len(active_asr) else math.nan,
            # Backward-compatible alias; new reports should use the explicit fields.
            "active_asr_auc": raw_auc,
            "active_asr_auc_raw": raw_auc,
            "active_asr_auc_normalized": _normalized_auc(active_asr),
            "peak_asr": float(objective_asr.max()),
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
    if spec["attack"] in LABEL_FLIP_ATTACKS:
        numerator = pd.to_numeric(
            active.get("fit_label_flip_poisoned_exposures", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0).sum()
        selected = pd.to_numeric(
            active.get("fit_selected_training_exposures", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0).sum()
        eligible = pd.to_numeric(
            active.get("fit_label_flip_eligible_exposures", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0).sum()
        active_malicious_rate = pd.to_numeric(
            active.get(
                "fit_label_flip_active_malicious_exposure_rate",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
        result.update({
            "label_flip_poisoned_exposures": float(numerator),
            "label_flip_global_exposure_rate": (
                float(numerator / selected) if selected > 0 else 0.0
            ),
            "label_flip_eligible_poison_rate": (
                float(numerator / eligible) if eligible > 0 else 0.0
            ),
            "label_flip_active_malicious_exposure_rate": (
                float(active_malicious_rate.mean())
                if len(active_malicious_rate) else 0.0
            ),
        })
    result["detection_lag"] = detection_lag(rounds)
    for state in ("watch", "restricted", "quarantined"):
        result[f"first_malicious_{state}_lag"] = first_state_lag(
            rounds, f"fit_malicious_{state}_rate"
        )
    for column in (
        "fit_malicious_aggregation_weight_share", "fit_malicious_impact_share",
        "fit_benign_quarantine_rate", "fit_benign_watch_rate",
        "fit_benign_restricted_rate", "fit_malicious_watch_rate",
        "fit_malicious_restricted_rate", "fit_malicious_quarantined_rate",
        "fit_aggregation_time_seconds",
        "fit_fft_periodic_penalty_clients", "fit_direction_penalty_clients",
        "fit_rtc_total_risk_mean", "fit_rtc_temporal_risk_mean",
        "fit_rtc_influence_risk_mean", "fit_rtc_event_risk_mean",
        "fit_time_consistency_fallback_used",
        "fit_freqfed_fallback",
        "fit_freqfed_selected_clients", "fit_freqfed_rejected_clients",
        "fit_freqfed_noise_clients", "fit_freqfed_cluster_count",
        "fit_freqfed_selected_ratio",
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


def _normalized_auc(values: pd.Series) -> float:
    """AUC normalized by the active-window span.

    A one-round active window is defined as that round's value.  For two or
    more observations this is the trapezoidal area divided by ``n - 1``.
    """
    raw = _auc(values)
    count = len(values)
    if count == 0:
        return math.nan
    if count == 1:
        return raw
    return float(raw / (count - 1))


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


def first_state_lag(rounds: pd.DataFrame, rate_column: str) -> float:
    """Rounds from first active attack to the first malicious state transition."""
    if rate_column not in rounds or "planned_attack_active" not in rounds:
        return math.nan
    active_rounds = rounds.loc[rounds["planned_attack_active"] == 1, "round"]
    if active_rounds.empty:
        return math.nan
    start = int(active_rounds.min())
    rates = pd.to_numeric(rounds[rate_column], errors="coerce").fillna(0.0)
    reached = rounds.loc[(rounds["round"] >= start) & (rates > 0), "round"]
    return float(int(reached.min()) - start) if not reached.empty else math.nan


def add_accuracy_drop(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    clean = summary[summary["attack"] == "none"] if "attack" in summary else pd.DataFrame()
    if clean.empty:
        summary["accuracy_drop"] = np.nan
        return summary
    pairing_key = "trial_plan_hash" if "trial_plan_hash" in summary else "seed"
    baselines = clean[clean["defense"] == "fedavg"].set_index(pairing_key)["final_accuracy"]
    summary["accuracy_drop"] = summary.apply(
        lambda row: float(
            baselines.get(row[pairing_key], np.nan) - row.get("final_accuracy", np.nan)
        ),
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
            pairing_key = str(first.get("trial_plan_hash", first.get("seed")))
            clean[pairing_key] = frame
        else:
            key = (str(first.get("attack")), str(first.get("period")),
                   float(first.get("malicious_fraction")), int(first.get("seed")),
                   str(first.get("defense")))
            attacked[key] = frame
    for index, row in summary.iterrows():
        pairing_key = str(row.get("trial_plan_hash", row.get("seed")))
        if row["attack"] == "none" or pairing_key not in clean:
            continue
        key = (str(row["attack"]), str(row["period"]), float(row["malicious_fraction"]),
               int(row["seed"]), str(row["defense"]))
        attack_frame = attacked.get(key)
        if attack_frame is None or "server_accuracy" not in attack_frame:
            continue
        merged = attack_frame[["round", "planned_attack_active", "server_accuracy"]].merge(
            clean[pairing_key][["round", "server_accuracy"]],
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
                          "first_malicious_watch_lag", "first_malicious_restricted_lag",
                          "first_malicious_quarantined_lag",
                          "accuracy_drop", "active_accuracy_drop", "accuracy_drop_auc",
                          "label_flip_global_exposure_rate",
                          "label_flip_eligible_poison_rate",
                          "label_flip_active_malicious_exposure_rate",
                          "malicious_aggregation_weight_share",
                          "benign_quarantine_rate", "aggregation_time_seconds",
                          "freqfed_fallback", "freqfed_selected_clients",
                          "freqfed_rejected_clients", "freqfed_noise_clients",
                          "freqfed_cluster_count", "freqfed_selected_ratio")
        if name in summary
    ]
    keys = ["attack", "period", "malicious_fraction", *CONDITION_KEYS, "defense"]
    keys = [key for key in keys if key in summary]
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
    if "active_asr_auc" not in summary or "attack" not in summary:
        return pd.DataFrame()
    comparisons = ["clip_only", "freqfed", "foolsgold", "fltrust",
                   "rtc_no_temporal", "rtc_no_direction", "rtc_no_influence", "rtc_trust_only"]
    rows = []
    for attack in TARGETED_ATTACKS:
        attack_rows = summary[summary["attack"] == attack]
        for comparator in comparisons:
            paired = _paired_values(attack_rows, "rtc_full", comparator, "active_asr_auc")
            if paired.empty:
                continue
            differences = paired["left"] - paired["right"]
            descriptive_only = len(paired) < 8
            rows.append({"attack": attack, "metric": "active_asr_auc", "left": "rtc_full",
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
    keys = ["trial_plan_hash", "period", "malicious_fraction", *CONDITION_KEYS, "seed"]
    keys = [key for key in keys if key in data and data[key].notna().any()]
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
    matrix = select_matrix(build_matrix(args.mode, args.smoke), args)
    if str(getattr(args, "pairing_mode", "strict")) == "strict":
        matrix = attach_trial_plans(matrix, args, output)
        clean = [] if args.skip_clean else paired_fedavg_clean_specs(matrix)
    else:
        clean = [] if args.skip_clean else clean_baselines(matrix, args.smoke)
    matrix = _deduplicate(clean if args.clean_only else matrix + clean)
    write_experiment_manifest(matrix, output)

    if args.smoke or args.clean_only:
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
        quality_gates = schedule_gates + validate_target_acceptance(summary)
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
    status_dir = output / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    _write_execution_environment(args, output)

    for index, base_spec in enumerate(specs, 1):
        spec = dict(base_spec)
        identifier = run_id(spec)
        round_path = rounds_dir / f"{identifier}.csv"
        status_path = status_dir / f"{identifier}.json"
        logger.info("[%d/%d] %s", index, len(specs), identifier)
        if round_path.exists() and not args.rerun:
            valid, reason, _ = validate_round_cache(
                round_path, _expected_rounds(args), spec,
            )
            if valid:
                _write_status(status_path, {
                    "run_id": identifier,
                    "state": "completed_cached",
                    "attempt": 0,
                    "pid": None,
                    "exit_code": 0,
                    "last_round": _expected_rounds(args),
                    "finished_at": _utc_now(),
                    "error_type": None,
                    "error_summary": None,
                })
                _write_current_summary(specs, args, output, rounds_dir)
                continue
            archived = _archive_invalid_round_cache(round_path)
            logger.warning("Invalid cached rounds archived | %s | %s", reason, archived)
            _write_status(status_path, {
                "run_id": identifier,
                "state": "invalid_cache",
                "attempt": 0,
                "last_round": 0,
                "error_type": "invalid_round_cache",
                "error_summary": reason,
                "archived_round_path": str(archived),
            })

        try:
            _run_spec_with_retries(
                spec, args, output, round_path, status_path,
            )
        except BaseException:
            _write_current_summary(specs, args, output, rounds_dir)
            raise
        _write_current_summary(specs, args, output, rounds_dir)

    return _write_current_summary(specs, args, output, rounds_dir)


def _expected_rounds(args: argparse.Namespace) -> int:
    return 6 if bool(getattr(args, "smoke", False)) else int(args.rounds)


def _build_spec_config(
    spec: Dict[str, Any], args: argparse.Namespace, output: Path,
):
    cfg = load_config(args.config)
    custom = dict(cfg.security.defense.custom_params or {})
    custom.update(spec["custom_params"])
    if args.smoke and spec["defense_type"] == "time_consistency":
        custom.setdefault("log_client_weights", False)
    num_clients = 10 if args.smoke else int(args.num_clients)
    clients_per_round = max(1, min(
        num_clients, int(math.ceil(num_clients * float(spec["participation_rate"])))
    ))
    overrides = {
        "project.seed": spec["seed"],
        "project.log_dir": str(output / "raw"),
        "dataset.name": "cifar10",
        "model.architecture": "resnet18",
        "federation.num_rounds": _expected_rounds(args),
        "federation.num_clients": num_clients,
        "federation.clients_per_round": num_clients if args.smoke else clients_per_round,
        "federation.min_fit_clients": num_clients if args.smoke else clients_per_round,
        "federation.min_available_clients": num_clients,
        "federation.max_client_samples": 50 if args.smoke else int(args.max_client_samples),
        "federation.pairing_mode": str(spec.get("pairing_mode", getattr(args, "pairing_mode", "legacy"))),
        "federation.sampling_protocol": str(spec.get("sampling_protocol", getattr(args, "sampling_protocol", "endpoint_uniform"))),
        "federation.trial_plan_path": str(spec.get("trial_plan_path", "")),
        "federation.trial_plan_hash": str(spec.get("trial_plan_hash", "")),
        "federation.deterministic_client_training": bool(
            spec.get("pairing_mode", getattr(args, "pairing_mode", "legacy")) == "strict"
        ),
        "dataset.partition": spec["partition"],
        "dataset.dirichlet_alpha": spec["dirichlet_alpha"],
        "client.local_epochs": 1 if args.smoke else cfg.client.local_epochs,
        "client.batch_size": int(getattr(args, "batch_size", 48)),
        "strategy.name": "fedavg",
        "security.attack.enabled": spec["attack"] != "none",
        "security.attack.type": spec["attack"],
        "security.attack.malicious_fraction": spec["malicious_fraction"],
        "security.attack.attack_start_round": spec["attack_start_round"],
        "security.attack.attack_end_round": spec["attack_end_round"],
        "security.attack.attack_on_rounds": spec["on_rounds"],
        "security.attack.attack_off_rounds": spec["off_rounds"],
        "security.attack.poison_fraction": (
            0.5 if spec["attack"] in {"backdoor", "dba"} else 0.1
        ),
        "security.attack.source_label": int(spec.get("label_flip_source_label", 5)),
        "security.attack.target_label": int(spec.get("label_flip_target_label", 3)),
        "security.attack.label_flip_poison_fraction": float(
            spec.get("label_flip_poison_fraction", 1.0)
        ),
        "security.attack.dba_boost_factor": float(
            spec.get("boost_factor", getattr(args, "boost_factor", 10.0))
        ),
        "security.attack.model_replacement_boost_factor": float(
            spec.get("boost_factor", getattr(args, "boost_factor", 10.0))
        ),
        "security.attack.mpaf_lambda": float(
            spec.get("boost_factor", getattr(args, "boost_factor", 1.0))
        ),
        "security.defense.enabled": spec["defense_type"] != "none",
        "security.defense.type": spec["defense_type"],
        "security.defense.reserve_root_for_all": True,
        "security.defense.root_dataset_size": 100,
        "security.defense.custom_params": custom,
        "evaluation.save_best_model": False,
        "evaluation.max_test_samples": 500 if args.smoke else int(args.max_test_samples),
        "ray.object_store_memory_mb": int(
            getattr(args, "ray_object_store_memory_mb", 0) or 0
        ),
        "ray.min_available_memory_mb": int(
            getattr(args, "ray_min_available_memory_mb", 0) or 0
        ),
        "ray.memory_wait_seconds": float(
            getattr(args, "ray_memory_wait_seconds", 120.0)
        ),
        "ray.include_dashboard": False,
        "ray.client_num_cpus": float(getattr(args, "ray_client_num_cpus", 1.0)),
        "ray.client_num_gpus": float(getattr(args, "ray_client_num_gpus", 0.0)),
    }
    return override_config(cfg, overrides)


def _run_spec_worker(
    spec: Dict[str, Any],
    args_payload: Dict[str, Any],
    output_value: str,
    round_path_value: str,
    status_path_value: str,
    attempt: int,
    log_start_offset: int,
) -> None:
    args = argparse.Namespace(**args_payload)
    output = Path(output_value)
    round_path = Path(round_path_value)
    status_path = Path(status_path_value)
    identifier = run_id(spec)
    log_path = output / "raw" / f"{identifier}.log"
    _write_status(status_path, {
        "run_id": identifier,
        "state": "running",
        "attempt": attempt,
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "exit_code": None,
        "last_round": 0,
        "ray_session": None,
        "error_type": None,
        "error_summary": None,
    })
    try:
        cfg = _build_spec_config(spec, args, output)
        tracker = run_simulation(cfg, experiment_name=identifier)
        rounds = tracker_to_rounds(tracker.to_list(), spec)
        _atomic_write_csv(rounds, round_path)
        _write_status(status_path, {
            "run_id": identifier,
            "state": "completed",
            "attempt": attempt,
            "pid": os.getpid(),
            "exit_code": 0,
            "last_round": _expected_rounds(args),
            "finished_at": _utc_now(),
            "ray_session": _find_ray_session(os.getpid()),
            "error_type": None,
            "error_summary": None,
        })
    except BaseException as exc:
        _write_status(status_path, {
            "run_id": identifier,
            "state": "failed",
            "attempt": attempt,
            "pid": os.getpid(),
            "exit_code": 1,
            "last_round": _last_logged_round(log_path, log_start_offset),
            "finished_at": _utc_now(),
            "ray_session": _find_ray_session(os.getpid()),
            "error_type": "python_exception",
            "error_summary": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        shutdown_ray_runtime()


def _run_spec_with_retries(
    spec: Dict[str, Any],
    args: argparse.Namespace,
    output: Path,
    round_path: Path,
    status_path: Path,
    launch_fn=None,
) -> None:
    identifier = run_id(spec)
    max_retries = max(0, int(getattr(args, "max_spec_retries", 1)))
    attempts = max_retries + 1
    launcher = launch_fn or _launch_spec_process
    for attempt in range(1, attempts + 1):
        if round_path.exists():
            archived = _archive_round_cache(round_path, f"attempt{attempt}-previous")
            logger.info("Previous round output preserved before attempt | %s", archived)
        try:
            wait_for_available_memory(
                int(getattr(args, "ray_min_available_memory_mb", 0) or 0),
                float(getattr(args, "ray_memory_wait_seconds", 120.0)),
            )
        except RayResourcePreflightError as exc:
            _write_status(status_path, {
                "run_id": identifier,
                "state": "resource_blocked",
                "attempt": attempt,
                "pid": None,
                "exit_code": None,
                "last_round": 0,
                "finished_at": _utc_now(),
                "error_type": "ray_memory_preflight",
                "error_summary": str(exc),
            })
            raise

        log_path = output / "raw" / f"{identifier}.log"
        log_start_offset = log_path.stat().st_size if log_path.exists() else 0
        _write_status(status_path, {
            "run_id": identifier,
            "state": "launching",
            "attempt": attempt,
            "pid": None,
            "started_at": _utc_now(),
            "exit_code": None,
            "last_round": 0,
            "log_start_offset": log_start_offset,
            "error_type": None,
            "error_summary": None,
        })
        exit_code, child_pid = launcher(
            spec,
            dict(vars(args)),
            output,
            round_path,
            status_path,
            attempt,
            log_start_offset,
        )
        valid, reason, _ = validate_round_cache(
            round_path, _expected_rounds(args), spec,
        )
        if exit_code == 0 and valid:
            _write_status(status_path, {
                "run_id": identifier,
                "state": "completed",
                "attempt": attempt,
                "pid": child_pid,
                "exit_code": 0,
                "last_round": _expected_rounds(args),
                "finished_at": _utc_now(),
                "error_type": None,
                "error_summary": None,
            })
            return

        current = _read_status(status_path)
        error_type, error_summary, ray_session = _classify_child_failure(
            child_pid, exit_code, current, reason,
        )
        failed_payload = {
            "run_id": identifier,
            "state": "retrying" if attempt < attempts else "failed",
            "attempt": attempt,
            "pid": child_pid,
            "exit_code": exit_code,
            "last_round": _last_logged_round(log_path, log_start_offset),
            "finished_at": _utc_now(),
            "ray_session": ray_session,
            "error_type": error_type,
            "error_summary": error_summary,
        }
        _write_status(status_path, failed_payload)
        if attempt < attempts:
            logger.warning(
                "Specification failed; retrying after resource preflight | run=%s attempt=%d/%d error=%s",
                identifier, attempt, attempts, error_summary,
            )
            continue
        raise RuntimeError(
            f"Specification failed after {attempts} attempts: {identifier}: {error_summary}"
        )


def _launch_spec_process(
    spec: Dict[str, Any],
    args_payload: Dict[str, Any],
    output: Path,
    round_path: Path,
    status_path: Path,
    attempt: int,
    log_start_offset: int,
) -> Tuple[int, int | None]:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_spec_worker,
        args=(
            spec,
            args_payload,
            str(output),
            str(round_path),
            str(status_path),
            attempt,
            log_start_offset,
        ),
        name=f"fedsec-{run_id(spec)}-attempt{attempt}",
    )
    process.start()
    child_pid = process.pid
    try:
        process.join()
    except BaseException:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        raise
    return int(process.exitcode if process.exitcode is not None else -1), child_pid


def validate_round_cache(
    path: Path,
    expected_rounds: int,
    spec: Dict[str, Any] | None = None,
) -> Tuple[bool, str, pd.DataFrame | None]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return False, f"unreadable CSV: {exc}", None
    required_columns = {"round", "server_loss", "server_accuracy"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        return False, f"missing columns: {','.join(missing)}", frame
    numeric_rounds = pd.to_numeric(frame["round"], errors="coerce")
    if numeric_rounds.isna().any() or not np.equal(numeric_rounds, numeric_rounds.astype(int)).all():
        return False, "round values must be finite integers", frame
    round_values = numeric_rounds.astype(int)
    if round_values.duplicated().any():
        return False, "duplicate round values", frame
    expected = set(range(1, int(expected_rounds) + 1))
    observed = set(round_values.tolist())
    missing_rounds = sorted(expected - observed)
    unexpected = sorted(observed - expected - {0})
    if missing_rounds:
        return False, f"missing rounds: {missing_rounds[:10]}", frame
    if unexpected:
        return False, f"unexpected rounds: {unexpected[:10]}", frame
    selected = round_values.isin(expected | {0})
    accuracy = pd.to_numeric(frame.loc[selected, "server_accuracy"], errors="coerce")
    if accuracy.isna().any() or not np.isfinite(accuracy.to_numpy()).all():
        return False, "server accuracy must be finite", frame
    raw_loss = frame.loc[selected, "server_loss"]
    numeric_loss = pd.to_numeric(raw_loss, errors="coerce")
    invalid_loss = raw_loss.notna() & numeric_loss.isna()
    if invalid_loss.any():
        return False, "server loss contains non-numeric values", frame
    if spec is not None and (
        str(spec.get("attack_group")) == "targeted"
        or str(spec.get("attack")) in TARGETED_ATTACKS
        or str(spec.get("attack")) == LABEL_FLIP_ALL_REVERSE
    ):
        asr = _objective_asr_series(frame.loc[selected], spec)
        if asr is None:
            return False, "attack-objective ASR must be finite or exactly recoverable", frame
    if spec is not None and spec.get("pairing_mode") == "strict":
        required_pairing = {
            "fit_trial_plan_hash", "fit_planned_partition_ids_json",
            "fit_completed_partition_ids_json", "fit_fit_seed_digest",
        }
        missing_pairing = sorted(required_pairing.difference(frame.columns))
        if missing_pairing:
            return False, f"missing strict-pairing columns: {','.join(missing_pairing)}", frame
        expected_hash = str(spec.get("trial_plan_hash", ""))
        fit_rows = frame[round_values.isin(expected)].copy()
        if set(fit_rows["fit_trial_plan_hash"].astype(str)) != {expected_hash}:
            return False, "trial-plan hash differs from experiment spec", frame
        plan = TrialPlanV1.load(str(spec.get("trial_plan_path", "")))
        for _, row in fit_rows.iterrows():
            round_plan = plan.round(int(row["round"]))
            planned = json.loads(str(row["fit_planned_partition_ids_json"]))
            completed = json.loads(str(row["fit_completed_partition_ids_json"]))
            if planned != list(round_plan["partition_ids"]):
                return False, f"planned sequence differs at round {int(row['round'])}", frame
            if completed != planned:
                return False, f"completed sequence differs at round {int(row['round'])}", frame
            if str(row["fit_fit_seed_digest"]) != str(round_plan["fit_seed_digest"]):
                return False, f"fit random stream differs at round {int(row['round'])}", frame
    return True, "complete", frame


def _write_current_summary(
    specs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output: Path,
    rounds_dir: Path,
) -> pd.DataFrame:
    rows = []
    for base_spec in specs:
        spec = dict(base_spec)
        path = rounds_dir / f"{run_id(spec)}.csv"
        if not path.exists():
            continue
        valid, _, frame = validate_round_cache(path, _expected_rounds(args), spec)
        if valid and frame is not None:
            rows.append(summarize_run(frame, spec))
    summary = pd.DataFrame(rows)
    _atomic_write_csv(summary, output / "periodic_attack_runs.csv")
    return summary


def _archive_invalid_round_cache(path: Path) -> Path:
    return _archive_round_cache(path, "incomplete")


def _archive_round_cache(path: Path, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = path.with_name(f"{path.stem}.{label}-{timestamp}{path.suffix}")
    counter = 1
    while archived.exists():
        archived = path.with_name(
            f"{path.stem}.{label}-{timestamp}-{counter}{path.suffix}"
        )
        counter += 1
    os.replace(path, archived)
    return archived


def _write_execution_environment(args: argparse.Namespace, output: Path) -> None:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    payload = {
        "created_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "ray": version("ray"),
            "flwr": version("flwr"),
            "torch": version("torch"),
        },
        "execution_mode": "spawn_per_spec",
        "ray_object_store_memory_mb": int(
            getattr(args, "ray_object_store_memory_mb", 0) or 0
        ),
        "ray_min_available_memory_mb": int(
            getattr(args, "ray_min_available_memory_mb", 0) or 0
        ),
        "ray_memory_wait_seconds": float(
            getattr(args, "ray_memory_wait_seconds", 120.0)
        ),
        "max_spec_retries": int(getattr(args, "max_spec_retries", 1)),
    }
    _atomic_write_json(payload, output / "execution_environment.json")


def _classify_child_failure(
    child_pid: int | None,
    exit_code: int,
    current_status: Dict[str, Any],
    cache_reason: str,
) -> Tuple[str, str, str | None]:
    ray_session = _find_ray_session(child_pid)
    if ray_session:
        raylet_error = Path(ray_session) / "logs" / "raylet.err"
        if raylet_error.exists():
            text = raylet_error.read_text(encoding="utf-8", errors="replace")
            if "GetLastError() = 1450" in text or "CreateFileMapping() failed" in text:
                return (
                    "ray_object_store_resource",
                    "Ray shared-memory mapping failed with Windows error 1450",
                    ray_session,
                )
    if current_status.get("error_type"):
        return (
            str(current_status["error_type"]),
            str(current_status.get("error_summary") or cache_reason),
            ray_session or current_status.get("ray_session"),
        )
    if exit_code == 0:
        return "invalid_round_output", cache_reason, ray_session
    return (
        "native_child_exit",
        f"child exited with code {exit_code}; round output: {cache_reason}",
        ray_session,
    )


def _find_ray_session(driver_pid: int | None) -> str | None:
    if not driver_pid:
        return None
    root = Path(tempfile.gettempdir()) / "ray"
    candidates = list(root.glob(f"session_*_{int(driver_pid)}")) if root.exists() else []
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _last_logged_round(path: Path, start_offset: int = 0) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as handle:
            handle.seek(min(max(0, int(start_offset)), path.stat().st_size))
            content = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0
    matches = [int(value) for value in re.findall(r"\bRound (\d+) (?:evaluate|aggregation done)", content)]
    return max(matches, default=0)


def _read_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    current = _read_status(path)
    current.update(payload)
    _atomic_write_json(current, path)


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_experiment_manifest(specs: Sequence[Dict[str, Any]], output: Path) -> None:
    payload = {"benchmark_version": BENCHMARK_VERSION, "specs": list(specs)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "experiment_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def select_matrix(matrix: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Apply explicit low-cost screening filters without changing benchmark definitions."""
    selected = list(matrix)
    if "label_flip" in _csv_values(getattr(args, "attacks", "")):
        raise ValueError(
            "Attack 'label_flip' was removed; use 'label_flip_targeted' or "
            "'label_flip_all_reverse'."
        )
    filters = {
        "attack": _csv_values(getattr(args, "attacks", "")),
        "defense": _csv_values(getattr(args, "defenses", "")),
        "period": _csv_values(getattr(args, "periods", "")),
    }
    seed_values = _csv_values(getattr(args, "seeds", ""))
    fraction_values = _csv_values(getattr(args, "malicious_fractions", ""))
    for key, allowed in filters.items():
        if allowed:
            selected = [spec for spec in selected if str(spec[key]) in allowed]
    if seed_values:
        allowed_seeds = {int(value) for value in seed_values}
        selected = [spec for spec in selected if int(spec["seed"]) in allowed_seeds]
    if fraction_values:
        allowed_fractions = {float(value) for value in fraction_values}
        selected = [
            spec for spec in selected
            if float(spec["malicious_fraction"]) in allowed_fractions
        ]
    if not selected:
        raise ValueError("Experiment filters selected no runs")
    normalized = []
    for spec in selected:
        row = {
            **spec,
            "attack_start_round": 3 if args.smoke else int(args.attack_start_round),
            "attack_end_round": int(args.attack_end_round),
            "partition": args.partition,
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "participation_rate": 1.0 if args.smoke else float(args.participation_rate),
        }
        if spec["attack"] in LABEL_FLIP_ATTACKS:
            row["label_flip_poison_fraction"] = float(
                getattr(args, "label_flip_poison_fraction", 1.0)
            )
            if spec["attack"] == LABEL_FLIP_TARGETED:
                row.update({
                    "label_flip_source_label": int(
                        getattr(args, "label_flip_source_label", 5)
                    ),
                    "label_flip_target_label": int(
                        getattr(args, "label_flip_target_label", 3)
                    ),
                })
            row.pop("boost_factor", None)
        else:
            row["boost_factor"] = float(args.boost_factor)
        normalized.append(row)
    return normalized


def _csv_values(raw: str) -> set[str]:
    return {part.strip() for part in str(raw).split(",") if part.strip()}


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
    strict_seen = False
    plan_hash_match = True
    plan_sequence_match = True
    completed_sequence_match = True
    random_stream_match = True
    foolsgold_history_match = True
    foolsgold_seen = False
    schedules: Dict[Tuple[Any, ...], Dict[int, str]] = {}
    schedules_by_plan: Dict[str, Dict[int, str]] = {}
    seeds_by_plan: Dict[str, Dict[int, str]] = {}
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
        pairing_mode = str(first.get("pairing_mode", "legacy"))
        schedule = dict(zip(
            fit["round"].astype(int),
            fit["fit_selected_partition_ids"].astype(str),
        ))
        # Legacy runs have no TrialPlan identity, so their condition tuple is
        # the only available cross-defense pairing key.  Strict runs must not
        # use that tuple: attack-specific FedAvg-clean counterfactuals all have
        # attack=none in their round files while intentionally retaining
        # different owner-plan hashes.  Comparing those unrelated clean plans
        # creates a false mismatch.  Strict cross-run equality is checked below
        # exclusively within each cryptographically bound plan hash.
        if pairing_mode != "strict":
            key = (
                first.get("attack"), first.get("period"),
                first.get("malicious_fraction"), first.get("seed"),
            )
            previous = schedules.setdefault(key, schedule)
            if previous != schedule:
                per_run_match = False
        if pairing_mode == "strict":
            strict_seen = True
            required = {
                "trial_plan_hash", "trial_plan_path", "fit_trial_plan_hash",
                "fit_planned_partition_ids_json", "fit_completed_partition_ids_json",
                "fit_fit_seed_digest",
            }
            if not required.issubset(frame.columns):
                plan_hash_match = plan_sequence_match = False
                completed_sequence_match = random_stream_match = False
                continue
            plan_hash = str(first["trial_plan_hash"])
            try:
                plan = TrialPlanV1.load(str(first["trial_plan_path"]))
            except Exception:
                plan_hash_match = False
                continue
            fit = frame[pd.to_numeric(frame["round"], errors="coerce") >= 1]
            plan_hash_match &= (
                plan.trial_plan_hash == plan_hash
                and set(fit["fit_trial_plan_hash"].astype(str)) == {plan_hash}
            )
            plan_schedule: Dict[int, str] = {}
            seed_schedule: Dict[int, str] = {}
            for _, row in fit.iterrows():
                rnd = int(row["round"])
                expected = list(plan.round(rnd)["partition_ids"])
                try:
                    planned = json.loads(str(row["fit_planned_partition_ids_json"]))
                    completed = json.loads(str(row["fit_completed_partition_ids_json"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    plan_sequence_match = completed_sequence_match = False
                    continue
                plan_sequence_match &= planned == expected
                completed_sequence_match &= completed == planned
                random_stream_match &= (
                    str(row["fit_fit_seed_digest"])
                    == str(plan.round(rnd)["fit_seed_digest"])
                )
                plan_schedule[rnd] = json.dumps(planned, separators=(",", ":"))
                seed_schedule[rnd] = str(row["fit_fit_seed_digest"])
            if str(first.get("defense")) == "foolsgold":
                foolsgold_seen = True
                if "fit_foolsgold_history_partition_ids_json" not in fit:
                    foolsgold_history_match = False
                else:
                    appeared: set[str] = set()
                    for _, row in fit.sort_values("round").iterrows():
                        appeared.update(str(value) for value in plan.round(int(row["round"]))["partition_ids"])
                        try:
                            history = set(json.loads(str(
                                row["fit_foolsgold_history_partition_ids_json"]
                            )))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            foolsgold_history_match = False
                            continue
                        foolsgold_history_match &= history == appeared
            previous_plan = schedules_by_plan.setdefault(plan_hash, plan_schedule)
            previous_seeds = seeds_by_plan.setdefault(plan_hash, seed_schedule)
            plan_sequence_match &= previous_plan == plan_schedule
            random_stream_match &= previous_seeds == seed_schedule
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
    if strict_seen:
        gates.extend([
            _gate("trial_plan_hash_match", plan_hash_match, plan_hash_match, "true"),
            _gate("trial_plan_sequence_match", plan_sequence_match, plan_sequence_match, "true"),
            _gate(
                "completed_sequence_equals_plan", completed_sequence_match,
                completed_sequence_match, "true",
            ),
            _gate("client_random_stream_match", random_stream_match, random_stream_match, "true"),
        ])

        pairing_manifests = list(raw_dir.glob("*_pairing_manifest.json"))
        if expected_ids is not None:
            pairing_manifests = [
                path for path in pairing_manifests
                if path.name.removesuffix("_pairing_manifest.json") in expected_ids
            ]
        malicious_by_plan: Dict[str, set[str]] = {}
        data_by_plan: Dict[str, set[str]] = {}
        initial_by_plan: Dict[str, set[str]] = {}
        for path in pairing_manifests:
            with path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            plan_hash = str(manifest.get("trial_plan_hash", ""))
            malicious_by_plan.setdefault(plan_hash, set()).add(
                str(manifest.get("malicious_identity_sha256", ""))
            )
            data_by_plan.setdefault(plan_hash, set()).add(
                str(manifest.get("data_manifest_sha256", ""))
            )
            initial_by_plan.setdefault(plan_hash, set()).add(
                str(manifest.get("initial_model_sha256", ""))
            )
        expected_plan_hashes = set(schedules_by_plan)
        manifests_complete = expected_plan_hashes == set(malicious_by_plan)
        malicious_match = manifests_complete and all(
            len(values) == 1 and "" not in values for values in malicious_by_plan.values()
        )
        runtime_data_match = manifests_complete and all(
            len(values) == 1 and "" not in values for values in data_by_plan.values()
        )
        initial_match = manifests_complete and all(
            len(values) == 1 and "" not in values for values in initial_by_plan.values()
        )
        gates.extend([
            _gate("malicious_identity_match", malicious_match,
                  {key: len(value) for key, value in malicious_by_plan.items()}, "one per plan"),
            _gate("paired_data_manifest_match", runtime_data_match,
                  {key: len(value) for key, value in data_by_plan.items()}, "one per plan"),
            _gate("initial_model_match", initial_match,
                  {key: len(value) for key, value in initial_by_plan.items()}, "one per plan"),
        ])
        if foolsgold_seen:
            gates.append(_gate(
                "foolsgold_history_equals_planned_union",
                foolsgold_history_match, foolsgold_history_match, "true",
            ))
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
        expected_rounds = set(range(0, int(frame["round"].max()) + 1))
        critical_complete &= set(frame["round"].astype(int)) == expected_rounds
        critical_complete &= pd.to_numeric(
            frame["server_accuracy"], errors="coerce"
        ).notna().all()
        if frame.iloc[0].get("attack") == "model_replacement":
            critical_complete &= pd.to_numeric(
                frame["server_asr"], errors="coerce"
            ).notna().all()
    gates.append(_gate("critical_metrics_complete", critical_complete,
                       critical_complete, "complete smoke rounds with finite accuracy/ASR"))
    rtc = attacked[attacked["defense"] == "rtc_full"].iloc[0]
    clip = attacked[attacked["defense"] == "clip_only"].iloc[0]
    no_temporal = frames["rtc_no_temporal"]
    no_temporal_values = (
        pd.to_numeric(no_temporal["fit_rtc_temporal_risk_mean"], errors="coerce").fillna(0)
        if "fit_rtc_temporal_risk_mean" in no_temporal
        else pd.Series([0.0])
    )
    no_direction_frame = frames["rtc_no_direction"]
    no_direction_values = (
        pd.to_numeric(no_direction_frame["fit_mean_direction_anomaly"], errors="coerce").fillna(0)
        if "fit_mean_direction_anomaly" in no_direction_frame
        else pd.Series([0.0])
    )
    no_temporal_sum = no_temporal_values.abs().sum()
    no_direction_sum = no_direction_values.abs().sum()
    gates.append(_gate("rtc_beats_fedavg_active_asr_auc",
                       rtc["active_asr_auc"] <= 0.5 * fedavg["active_asr_auc"],
                       {"rtc": rtc["active_asr_auc"], "fedavg": fedavg["active_asr_auc"]},
                       "<= 50% of fedavg"))
    rtc_auc = float(rtc.get("active_asr_auc", math.nan))
    clip_auc = float(clip.get("active_asr_auc", math.nan))
    gates.append(_gate("rtc_beats_clip_only_active_asr_auc",
                       np.isfinite(rtc_auc) and np.isfinite(clip_auc) and rtc_auc <= 0.8 * clip_auc,
                       {"rtc": rtc.get("active_asr_auc"), "clip_only": clip.get("active_asr_auc")},
                       "<= 80% of clip_only"))
    gates.append(_gate("rtc_no_temporal_is_zero", no_temporal_sum == 0,
                       no_temporal_sum, "temporal risk mean zero"))
    gates.append(_gate("rtc_no_direction_is_zero", no_direction_sum == 0,
                       no_direction_sum, "direction risk mean zero"))

    freq_rows = attacked[attacked["defense"] == "freqfed"]
    if not freq_rows.empty:
        freq = freq_rows.iloc[0]
        fallback = float(freq.get("freqfed_fallback", math.nan))
        gates.append(_gate("freqfed_fallback_rate", np.isfinite(fallback) and fallback <= 0.25,
                           fallback, "<= 0.25"))
    clean_freq_rows = summary[(summary["attack"] == "none") & (summary["defense"] == "freqfed")]
    if not clean_freq_rows.empty:
        fallback = float(clean_freq_rows.iloc[0].get("freqfed_fallback", math.nan))
        gates.append(_gate("freqfed_clean_fallback_rate",
                           np.isfinite(fallback) and fallback <= 0.25,
                           fallback, "<= 0.25"))
    expected_ids = [str(row["run_id"]) for _, row in summary.iterrows()]
    unique_logs = all(
        (raw_dir / f"{identifier}.csv").exists() and (raw_dir / f"{identifier}.json").exists()
        for identifier in expected_ids
    )
    gates.append(_gate("unique_metric_files", unique_logs, unique_logs, "true"))
    return gates


def validate_formal_preflight(summary: pd.DataFrame) -> List[Dict[str, Any]]:
    if summary.empty or not {"defense", "attack"}.issubset(summary.columns):
        return []
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


def validate_target_acceptance(summary: pd.DataFrame) -> List[Dict[str, Any]]:
    """Evaluate target-mode gates wherever the required paired rows exist."""
    gates: List[Dict[str, Any]] = []
    if summary.empty:
        return gates
    if "active_asr_auc" in summary and "attack" in summary:
        attacked = summary[
            (summary["attack"] != "none")
            & summary["active_asr_auc"].notna()
        ]
    else:
        attacked = pd.DataFrame()
    group_keys = [
        key for key in ("attack", "period", "malicious_fraction", *CONDITION_KEYS)
        if key in attacked
    ]
    for group_key, group in attacked.groupby(group_keys, dropna=False):
        label = "|".join(map(str, group_key if isinstance(group_key, tuple) else (group_key,)))
        pivot = group.pivot_table(index="seed", columns="defense", values="active_asr_auc", aggfunc="first")
        if {"rtc_full", "fedavg"}.issubset(pivot.columns):
            paired = pivot[["rtc_full", "fedavg"]].dropna()
            ratio = float(paired["rtc_full"].mean() / max(paired["fedavg"].mean(), 1e-12))
            gates.append(_gate(f"target_rtc_vs_fedavg:{label}", ratio <= 0.5,
                               ratio, "mean RTC/FedAvg <= 0.50"))
            gates.append(_gate(f"target_seed_direction_fedavg:{label}",
                               bool((paired["rtc_full"] < paired["fedavg"]).all()),
                               int((paired["rtc_full"] < paired["fedavg"]).sum()),
                               f"all {len(paired)} paired seeds improve"))
        if {"rtc_full", "clip_only"}.issubset(pivot.columns):
            paired = pivot[["rtc_full", "clip_only"]].dropna()
            ratio = float(paired["rtc_full"].mean() / max(paired["clip_only"].mean(), 1e-12))
            gates.append(_gate(f"target_rtc_vs_clip:{label}", ratio <= 0.8,
                               ratio, "mean RTC/clip-only <= 0.80"))
        rtc = group[group["defense"] == "rtc_full"]
        if not rtc.empty and "benign_quarantine_rate" in rtc:
            rate = float(pd.to_numeric(rtc["benign_quarantine_rate"], errors="coerce").mean())
            gates.append(_gate(f"target_benign_quarantine:{label}", rate <= 0.05,
                               rate, "<= 0.05"))

    clean = summary[summary["attack"] == "none"] if "attack" in summary else pd.DataFrame()
    if not clean.empty:
        clean_keys = [key for key in (*CONDITION_KEYS, "seed") if key in clean]
        pivot = clean.pivot_table(index=clean_keys, columns="defense", values="final_accuracy", aggfunc="first")
        if {"rtc_full", "fedavg"}.issubset(pivot.columns):
            drops = pivot["fedavg"] - pivot["rtc_full"]
            gates.append(_gate("target_clean_accuracy_drop", bool((drops <= 0.03).all()),
                               float(drops.max()), "every paired seed <= 0.03"))
    return gates


def clean_baselines(matrix: Sequence[Dict[str, Any]], smoke: bool) -> List[Dict[str, Any]]:
    """Add same-seed clean trajectories for utility and accuracy-drop metrics."""
    seeds = sorted({int(spec["seed"]) for spec in matrix})
    selected_names = {str(spec["defense"]) for spec in matrix}
    # One clean FedAvg trajectory supports utility deltas for every attacked
    # defense. RTC gets its own clean trajectory; extra clean baselines are
    # retained only for defenses whose preflight explicitly validates them.
    defense_names = ["fedavg"]
    if "rtc_full" in selected_names:
        defense_names.append("rtc_full")
    if (not smoke) or ("freqfed" in selected_names):
        defense_names.extend(
            name for name in ("freqfed", "fltrust") if name in selected_names
        )
    rows = []
    definitions = {**DEFENSES, **ABLATIONS}
    clean_condition_keys = (
        "partition", "dirichlet_alpha", "participation_rate",
        "attack_start_round", "attack_end_round",
    )
    conditions = {
        tuple((key, spec.get(key)) for key in clean_condition_keys)
        for spec in matrix
    }
    representative_boost = next(
        (float(spec["boost_factor"]) for spec in matrix if "boost_factor" in spec),
        10.0,
    )
    for seed, defense_name, condition_items in itertools.product(seeds, defense_names, conditions):
        defense_type, custom = definitions[defense_name]
        rows.append({**dict(condition_items), "attack": "none", "period": "clean", "on_rounds": 1,
                     "off_rounds": 0, "malicious_fraction": 0.0, "seed": seed,
                     "boost_factor": representative_boost,
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
    if "final_accuracy" in summary and "active_asr_auc" in summary:
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
        if frame.get("defense", pd.Series(dtype=str)).eq("rtc_full").any():
            timeline_frames.append(frame)
    if timeline_frames:
        timeline = pd.concat(timeline_frames, ignore_index=True)
        columns = [name for name in (
            "server_asr", "fit_active_attacker_weight_share",
            "fit_fft_periodic_penalty_clients", "fit_direction_penalty_clients",
            "fit_rtc_total_risk_mean", "fit_rtc_temporal_risk_mean",
            "fit_rtc_influence_risk_mean"
        ) if name in timeline]
        for metric in columns:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=timeline, x="round", y=metric, hue="attack", style="period", errorbar="sd")
            plt.tight_layout(); plt.savefig(output / f"timeline_{metric}.png", dpi=180); plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic attack defense benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="logs/periodic_attack_v2")
    parser.add_argument(
        "--mode",
        choices=("main", "ablation", "untargeted", "mpaf", "all"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a 6-round single-condition matrix")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-clean", action="store_true", help="Run attacked screening cells only")
    parser.add_argument("--clean-only", action="store_true", help="Run only generated clean baselines")
    parser.add_argument("--attacks", default="", help="Comma-separated attack filter")
    parser.add_argument("--defenses", default="", help="Comma-separated defense filter")
    parser.add_argument("--periods", default="", help="Comma-separated period filter")
    parser.add_argument("--seeds", default="", help="Comma-separated seed filter")
    parser.add_argument("--malicious-fractions", default="", help="Comma-separated malicious-fraction filter")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--partition", choices=("iid", "non_iid", "dirichlet"), default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--boost-factor", type=float, default=10.0)
    parser.add_argument("--label-flip-source-label", type=int, default=5)
    parser.add_argument("--label-flip-target-label", type=int, default=3)
    parser.add_argument("--label-flip-poison-fraction", type=float, default=1.0)
    parser.add_argument("--attack-start-round", type=int, default=11)
    parser.add_argument("--attack-end-round", type=int, default=-1)
    parser.add_argument("--max-client-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--ray-object-store-memory-mb", type=int, default=3072)
    parser.add_argument("--ray-min-available-memory-mb", type=int, default=10240)
    parser.add_argument("--ray-memory-wait-seconds", type=float, default=120.0)
    parser.add_argument("--max-spec-retries", type=int, default=1)
    parser.add_argument(
        "--pairing-mode", choices=("strict", "legacy"), default="strict",
    )
    parser.add_argument(
        "--sampling-protocol",
        choices=("principal_uniform", "endpoint_uniform"),
        default="principal_uniform",
    )
    args = parser.parse_args()
    if min(
        args.ray_object_store_memory_mb,
        args.ray_min_available_memory_mb,
        args.ray_memory_wait_seconds,
        args.max_spec_retries,
    ) < 0:
        parser.error("Ray resource and retry parameters must be non-negative")
    if not 0.0 <= args.label_flip_poison_fraction <= 1.0:
        parser.error("--label-flip-poison-fraction must be in [0, 1]")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiments(parse_args())
