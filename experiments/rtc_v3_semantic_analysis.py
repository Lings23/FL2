"""Compute audit-grade semantic/time/exposure metrics from valid strict runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.periodic_attack import _objective_asr_series, run_id
from defenses.rtc.calibration import CalibrationManifest
from defenses.rtc.semantic import hard_exposure_coefficients, select_top_pair_indices


def _finite(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series([numeric], dtype=float)
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def principal_recovery_metrics(
    *,
    rounds: pd.DataFrame,
    clients: pd.DataFrame,
    recovery_risk_threshold: float = 0.05,
    recovery_observations: int = 3,
) -> dict[str, Any]:
    """Measure recovery using actual post-attack principal observations.

    Missing rounds never advance a streak.  Each principal that actively
    attacked must independently recover; the run-level delay is the slowest
    principal, preventing changing participant composition from looking like
    risk release.
    """
    required = {"round", "principal_id", "is_malicious", "attack_active", "semantic_risk"}
    if not required.issubset(clients):
        return {
            "time_to_recovery": np.nan,
            "risk_release_half_life": np.nan,
            "recovery_principal_count": 0,
            "recovered_principal_count": 0,
            "half_life_principal_count": 0,
        }
    active_rounds = pd.to_numeric(
        rounds.loc[
            pd.to_numeric(rounds.get("planned_attack_active"), errors="coerce") > 0,
            "round",
        ],
        errors="coerce",
    ).dropna()
    if active_rounds.empty:
        return {
            "time_to_recovery": np.nan,
            "risk_release_half_life": np.nan,
            "recovery_principal_count": 0,
            "recovered_principal_count": 0,
            "half_life_principal_count": 0,
        }
    stop = int(active_rounds.max())
    frame = clients.copy()
    frame["_round"] = pd.to_numeric(frame["round"], errors="coerce")
    frame["_risk"] = pd.to_numeric(frame["semantic_risk"], errors="coerce")
    malicious = frame[frame["is_malicious"].astype(bool)]
    attackers = malicious[
        malicious["attack_active"].astype(bool)
        & malicious["_round"].le(stop)
        & malicious["_risk"].notna()
    ]
    principal_ids = sorted(set(attackers["principal_id"].astype(str)))
    recovered_delays: list[int] = []
    half_life_delays: list[int] = []
    for principal in principal_ids:
        active = attackers[attackers["principal_id"].astype(str) == principal].sort_values("_round")
        baseline_risk = float(active.iloc[-1]["_risk"])
        post = malicious[
            (malicious["principal_id"].astype(str) == principal)
            & malicious["_round"].gt(stop)
            & malicious["_risk"].notna()
        ].sort_values("_round")
        streak = 0
        recovered_round: int | None = None
        for _, observation in post.iterrows():
            if float(observation["_risk"]) <= float(recovery_risk_threshold):
                streak += 1
                if streak >= int(recovery_observations):
                    recovered_round = int(observation["_round"])
                    break
            else:
                streak = 0
        if recovered_round is not None:
            recovered_delays.append(recovered_round - stop)
        half = post[post["_risk"] <= baseline_risk / 2.0]
        if len(half):
            half_life_delays.append(int(half.iloc[0]["_round"]) - stop)
    fully_recovered = bool(principal_ids and len(recovered_delays) == len(principal_ids))
    all_halved = bool(principal_ids and len(half_life_delays) == len(principal_ids))
    return {
        "time_to_recovery": max(recovered_delays) if fully_recovered else np.nan,
        "risk_release_half_life": max(half_life_delays) if all_halved else np.nan,
        "recovery_principal_count": len(principal_ids),
        "recovered_principal_count": len(recovered_delays),
        "half_life_principal_count": len(half_life_delays),
    }


def clean_false_persistence_metrics(
    *,
    clients: pd.DataFrame,
    watch_threshold: float,
) -> dict[str, Any]:
    """Measure benign semantic persistence over actual observations only."""
    required = {"round", "principal_id", "is_malicious", "semantic_risk", "semantic_q"}
    if not required.issubset(clients):
        return {
            "false_persistence": np.nan,
            "false_soft_downweight_rate": np.nan,
            "false_persistent_principal_rate": np.nan,
            "false_persistence_max_observation_streak": 0,
        }
    benign = clients[~clients["is_malicious"].astype(bool)].copy()
    benign["_round"] = pd.to_numeric(benign["round"], errors="coerce")
    benign["_risk"] = pd.to_numeric(benign["semantic_risk"], errors="coerce")
    benign["_q"] = pd.to_numeric(benign["semantic_q"], errors="coerce")
    benign = benign.dropna(subset=["_round", "_risk", "_q"])
    if benign.empty:
        return {
            "false_persistence": np.nan,
            "false_soft_downweight_rate": np.nan,
            "false_persistent_principal_rate": np.nan,
            "false_persistence_max_observation_streak": 0,
        }
    watched = benign["_risk"] >= float(watch_threshold)
    soft = benign["_q"] < 1.0 - 1e-12
    principal_streaks: list[int] = []
    persistent_principals = 0
    for _, frame in benign.groupby(benign["principal_id"].astype(str), sort=True):
        current = maximum = 0
        for active in (frame.sort_values("_round")["_risk"] >= float(watch_threshold)):
            current = current + 1 if bool(active) else 0
            maximum = max(maximum, current)
        principal_streaks.append(maximum)
        persistent_principals += int(maximum > 0)
    return {
        "false_persistence": float(watched.mean()),
        "false_soft_downweight_rate": float(soft.mean()),
        "false_persistent_principal_rate": float(
            persistent_principals / len(principal_streaks)
        ),
        "false_persistence_max_observation_streak": max(principal_streaks),
    }


def strict_root_is_valid(root: Path) -> bool:
    path = root / "execution_validation.csv"
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    return bool(
        {"gate", "passed"}.issubset(frame)
        and len(frame)
        and frame["passed"].astype(str).str.lower().isin({"true", "1", "1.0"}).all()
    )


def analyze_run(
    spec: Mapping[str, Any],
    rounds: pd.DataFrame,
    clients: pd.DataFrame,
    *,
    semantic_watch_threshold: float = 0.1,
) -> dict[str, Any]:
    ordered = rounds.sort_values("round").copy()
    round_numbers = pd.to_numeric(ordered["round"], errors="coerce")
    fit = ordered[round_numbers > 0].copy()
    active_mask = pd.to_numeric(
        fit.get("planned_attack_active"), errors="coerce"
    ).fillna(0.0) > 0.0
    active = fit[active_mask]
    asr = _objective_asr_series(fit, dict(spec))
    asr_values = _finite(asr) if asr is not None else pd.Series(dtype=float)
    active_asr = (
        _finite(asr.loc[active.index])
        if asr is not None and len(active)
        else pd.Series(dtype=float)
    )
    asr_auc_raw = (
        float(active_asr.iloc[0])
        if len(active_asr) == 1
        else (
            float(np.trapezoid(active_asr.to_numpy()))
            if len(active_asr) > 1
            else np.nan
        )
    )
    asr_auc_normalized = (
        asr_auc_raw
        if len(active_asr) == 1
        else (
            float(asr_auc_raw / (len(active_asr) - 1))
            if len(active_asr) > 1
            else np.nan
        )
    )
    accuracy = _finite(fit.get("server_accuracy"))
    active_accuracy = _finite(active.get("server_accuracy"))
    result: dict[str, Any] = {
        "trial_plan_hash": str(spec.get("trial_plan_hash", "")),
        "seed": int(spec["seed"]),
        "attack": str(spec["attack"]),
        "period": str(spec["period"]),
        "attack_start_round": int(spec["attack_start_round"]),
        "attack_end_round": int(spec["attack_end_round"]),
        "defense": str(spec["defense"]),
        "active_mean_asr": float(active_asr.mean()) if len(active_asr) else np.nan,
        # Keep the historical name as a raw-area alias while exposing the
        # unitless normalized form explicitly for cross-schedule reporting.
        "asr_auc": asr_auc_raw,
        "asr_auc_raw": asr_auc_raw,
        "asr_auc_normalized": asr_auc_normalized,
        "peak_asr": float(active_asr.max()) if len(active_asr) else np.nan,
        "final_accuracy": float(accuracy.iloc[-1]) if len(accuracy) else np.nan,
        "best_accuracy": float(accuracy.max()) if len(accuracy) else np.nan,
        "last10_accuracy": float(accuracy.iloc[-10:].mean()) if len(accuracy) else np.nan,
        "active_accuracy": float(active_accuracy.mean()) if len(active_accuracy) else np.nan,
    }
    selected_k = pd.to_numeric(
        active.get("fit_selected_active_attackers"), errors="coerce"
    )
    for k in (3, 4):
        values = (
            _finite(asr.loc[active.index[selected_k == k]])
            if asr is not None and len(active)
            else pd.Series(dtype=float)
        )
        result[f"asr_k{k}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"asr_k{k}_worst"] = float(values.max()) if len(values) else np.nan
    for source, destination in (
        ("fit_malicious_aggregation_weight_share", "malicious_effective_weight"),
        ("fit_rtc_v3_zero_update_mass", "zero_update_mass"),
        ("fit_rtc_v3_semantic_exposure", "semantic_exposure"),
    ):
        values = _finite(active.get(source) if len(active) else fit.get(source))
        result[destination] = float(values.mean()) if len(values) else np.nan

    # Evaluate detection on every participating client in attack-active rounds.
    # Filtering directly on ``attack_active`` keeps only active attackers and
    # makes benign FPR and ROC AUC undefined by construction.
    active_round_ids = set(
        pd.to_numeric(active.get("round"), errors="coerce").dropna().astype(int)
    )
    client_rounds = pd.to_numeric(clients.get("round"), errors="coerce")
    active_round_clients = clients[client_rounds.isin(active_round_ids)].copy()
    if "is_malicious" in active_round_clients:
        malicious_mask = active_round_clients["is_malicious"].astype(bool)
        benign_clients = active_round_clients[~malicious_mask]
        active_clients = active_round_clients[
            malicious_mask
            & (
                pd.to_numeric(
                    active_round_clients.get("attack_active"), errors="coerce"
                ).fillna(0.0)
                > 0.0
            )
        ]
    else:
        benign_clients = active_round_clients.iloc[0:0]
        active_clients = active_round_clients.iloc[0:0]
    result["clip_recall"] = float(
        pd.to_numeric(active_clients.get("clipped"), errors="coerce").mean()
    ) if len(active_clients) else np.nan
    result["clip_fpr"] = float(
        pd.to_numeric(benign_clients.get("clipped"), errors="coerce").mean()
    ) if len(benign_clients) else np.nan
    if (
        "semantic_risk" in active_round_clients
        and "is_malicious" in active_round_clients
    ):
        labels = active_round_clients["is_malicious"].astype(bool).to_numpy(dtype=int)
        risks = pd.to_numeric(
            active_round_clients["semantic_risk"], errors="coerce"
        ).to_numpy()
        valid = np.isfinite(risks)
        result["semantic_detection_auc"] = (
            float(roc_auc_score(labels[valid], risks[valid]))
            if valid.any() and len(np.unique(labels[valid])) == 2
            else np.nan
        )
    else:
        result["semantic_detection_auc"] = np.nan

    active_rounds = pd.to_numeric(active.get("round"), errors="coerce").dropna()
    risk_mean = _finite(fit.get("fit_rtc_v3_semantic_risk_mean"))
    if "fit_rtc_v3_semantic_risk_mean" in fit:
        risk_by_round = pd.Series(
            pd.to_numeric(
                fit["fit_rtc_v3_semantic_risk_mean"], errors="coerce"
            ).to_numpy(),
            index=pd.to_numeric(fit["round"], errors="coerce").to_numpy(),
        ).dropna()
    else:
        risk_by_round = pd.Series(dtype=float)
    if len(active_rounds) and len(risk_by_round):
        attack_start = int(active_rounds.min())
        detected = risk_by_round[(risk_by_round.index >= attack_start) & (risk_by_round > 0.0)]
        result["detection_lag"] = (
            int(detected.index.min()) - attack_start if len(detected) else np.nan
        )
    else:
        result["detection_lag"] = np.nan
    if len(active_rounds) and int(spec.get("attack_end_round", -1)) > 0:
        result.update(principal_recovery_metrics(rounds=fit, clients=clients))
    else:
        result["time_to_recovery"] = np.nan
        result["risk_release_half_life"] = np.nan
        result["recovery_principal_count"] = 0
        result["recovered_principal_count"] = 0
        result["half_life_principal_count"] = 0
    # Validation-only clean trajectories may retain the attacked condition's
    # label so they share its TrialPlan, while scheduling the attack beyond the
    # final round.  Classify clean by observed schedule, not by the label.
    is_clean_trajectory = bool(not active_mask.any())
    if is_clean_trajectory:
        result.update(
            clean_false_persistence_metrics(
                clients=clients,
                watch_threshold=float(semantic_watch_threshold),
            )
        )
    else:
        result["false_persistence"] = np.nan
        result["false_soft_downweight_rate"] = np.nan
        result["false_persistent_principal_rate"] = np.nan
        result["false_persistence_max_observation_streak"] = 0

    semantic_columns = [
        column
        for column in fit
        if column.startswith("fit_rtc_v3_semantic_active_w")
    ]
    result["semantic_trigger_count"] = sum(
        int((_finite(fit[column]) > 0.0).sum()) for column in semantic_columns
    )
    for family, column in (
        ("cumulative", "fit_rtc_v3_cumulative_active_count"),
        ("direction", "fit_rtc_v3_direction_active_count"),
        ("principal", "fit_rtc_v3_principal_budget_active_count"),
    ):
        result[f"{family}_trigger_count"] = int(_finite(fit.get(column)).sum())
    server_count = 0
    for column in (
        "fit_rtc_v3_full_anchor_active_w1",
        "fit_rtc_v3_full_anchor_active_w4",
        "fit_rtc_v3_full_anchor_active_w8",
        "fit_rtc_v3_full_residual_active_w1",
        "fit_rtc_v3_full_residual_active_w4",
        "fit_rtc_v3_full_residual_active_w8",
        "fit_rtc_v3_full_total_active_w1",
        "fit_rtc_v3_full_total_active_w4",
        "fit_rtc_v3_full_total_active_w8",
    ):
        server_count += int((_finite(fit.get(column)) > 0.0).sum())
    result["server_trigger_count"] = server_count
    cone_count = 0
    for used_column in [
        column
        for column in fit
        if "_cone_" in column and "_used_w" in column
    ]:
        budget_column = used_column.replace("_used_w", "_budget_w")
        if budget_column in fit:
            used = pd.to_numeric(fit[used_column], errors="coerce")
            budget = pd.to_numeric(fit[budget_column], errors="coerce")
            cone_count += int((used >= budget - 1e-8).fillna(False).sum())
    result["cone_trigger_count"] = cone_count
    result["prototype_contamination_events"] = int(
        _finite(fit.get("fit_rtc_v3_cone_update_applied_count")).sum()
    )
    for column in (
        "fit_semantic_extraction_seconds",
        "fit_semantic_temporal_seconds",
        "fit_semantic_group_seconds",
        "fit_semantic_ledger_seconds",
        "fit_semantic_total_seconds",
        "fit_solver_seconds",
        "fit_anchor_seconds",
        "fit_sketch_seconds",
        "fit_total_defense_seconds",
    ):
        values = _finite(fit.get(column))
        key = column.removeprefix("fit_")
        result[f"{key}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"{key}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
    return result


def replay_semantic_budget_triggers(
    *,
    rounds: pd.DataFrame,
    clients: pd.DataFrame,
    manifest: CalibrationManifest,
) -> dict[str, int]:
    """Replay compact semantic ledgers from per-client strict artifacts."""
    config = manifest.payload["semantic_temporal_exposure"]
    required = {
        "round",
        "principal_id",
        "nominal_mass",
        "aggregation_weight",
        "rtc_v3_semantic_head_norm",
        "semantic_risk",
        "semantic_top_source",
        "semantic_top_target",
    }
    if not required.issubset(clients):
        raise ValueError(
            "semantic budget replay is missing client columns: "
            f"{sorted(required - set(clients))}"
        )
    phase_by_round = {
        int(row["round"]): str(row["fit_rtc_v3_cone_profile"])
        for _, row in rounds.iterrows()
        if pd.notna(row.get("fit_rtc_v3_cone_profile"))
        and int(row["round"]) > 0
    }
    global_events: dict[int, float] = {}
    group_events: dict[str, dict[int, float]] = {}
    principal_events: dict[str, list[float]] = {}
    counts = {"global": 0, "group": 0, "principal": 0}

    def scalar(raw: Any, profile: str) -> float:
        if isinstance(raw, Mapping):
            if profile in raw:
                return scalar(raw[profile], profile)
            if "default" in raw:
                return scalar(raw["default"], profile)
            raise ValueError(f"semantic budget has no {profile!r} entry")
        return float(raw)

    for round_number, frame in clients.groupby("round", sort=True):
        server_round = int(round_number)
        if server_round <= 0:
            continue
        profile = phase_by_round[server_round]
        risk = pd.to_numeric(frame["semantic_risk"], errors="raise").to_numpy(float)
        head = pd.to_numeric(
            frame["rtc_v3_semantic_head_norm"], errors="raise"
        ).to_numpy(float)
        head_scale = float(config["phases"][profile]["head_exposure_scale"])
        coefficients = hard_exposure_coefficients(
            risk,
            head / head_scale,
            float(config.get("hard_exposure_risk_floor", 0.0)),
        )
        nominal = pd.to_numeric(frame["nominal_mass"], errors="raise").to_numpy(float)
        weights = pd.to_numeric(
            frame["aggregation_weight"], errors="raise"
        ).to_numpy(float)
        pairs = tuple(
            (int(source), int(target))
            for source, target in zip(
                frame["semantic_top_source"], frame["semantic_top_target"]
            )
        )
        num_classes = int(config["num_classes"])
        pair_universe = tuple(
            (source, target)
            for source in range(num_classes)
            for target in range(num_classes)
            if source != target
        )
        pair_lookup = {pair: index for index, pair in enumerate(pair_universe)}
        pair_indices = np.asarray([pair_lookup[pair] for pair in pairs], dtype=np.int64)
        selected = set(
            select_top_pair_indices(
                client_pair_indices=pair_indices,
                exposure_coefficients=coefficients,
                nominal_masses=nominal,
                pairs=pair_universe,
                top_k=int(config["top_k_pairs"]),
            )
        )
        global_current = float(np.dot(coefficients, weights))
        for window in map(int, config["global_windows"]):
            history = sum(
                global_events.get(previous, 0.0)
                for previous in range(max(1, server_round - window + 1), server_round)
            )
            budget = scalar(config["global_budgets"][str(window)], profile)
            counts["global"] += int(history + global_current >= budget - 1e-8)
        current_groups: dict[str, float] = {}
        for pair_index in selected:
            source, target = pair_universe[pair_index]
            key = f"pair:{source}->{target}"
            current_groups[key] = float(
                np.dot(coefficients * (pair_indices == pair_index), weights)
            )
        current_groups["other"] = float(
            np.dot(
                coefficients
                * np.asarray([index not in selected for index in pair_indices], dtype=float),
                weights,
            )
        )
        for key, current in current_groups.items():
            history_by_round = group_events.setdefault(key, {})
            for window in map(int, config["pair_windows"]):
                history = sum(
                    history_by_round.get(previous, 0.0)
                    for previous in range(
                        max(1, server_round - window + 1), server_round
                    )
                )
                budget = scalar(config["pair_budgets"][str(window)], profile)
                counts["group"] += int(history + current >= budget - 1e-8)
            history_by_round[server_round] = current
        global_events[server_round] = global_current

        principals = frame["principal_id"].astype(str).to_numpy()
        for principal in sorted(set(principals)):
            current = float(np.dot(coefficients * (principals == principal), weights))
            history = principal_events.setdefault(principal, [])
            for window in map(int, config["principal_windows"]):
                used = sum(history[-max(0, window - 1) :]) + current if window > 1 else current
                budget = scalar(config["principal_budgets"][str(window)], profile)
                counts["principal"] += int(used >= budget - 1e-8)
            history.append(current)
    return counts


def analyze_root(root: str | Path) -> pd.DataFrame:
    root_path = Path(root).resolve()
    if not strict_root_is_valid(root_path):
        raise RuntimeError(
            "strict gates failed or are missing; invalid runs are excluded from analysis"
        )
    manifest = json.loads(
        (root_path / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    specs = manifest.get("specs", manifest) if isinstance(manifest, Mapping) else manifest
    rows = []
    for spec in specs:
        identifier = run_id(dict(spec))
        round_path = root_path / "rounds" / f"{identifier}.csv"
        client_path = root_path / "raw" / f"{identifier}_clients.csv"
        if not round_path.is_file() or not client_path.is_file():
            raise FileNotFoundError(f"strict run artifacts missing for {identifier}")
        round_frame = pd.read_csv(round_path)
        client_frame = pd.read_csv(client_path)
        calibration_source = (spec.get("custom_params") or {}).get(
            "calibration_manifest",
            (spec.get("custom_params") or {}).get("calibration_path"),
        )
        loaded_calibration = (
            CalibrationManifest.load(calibration_source)
            if calibration_source
            else None
        )
        watch_threshold = (
            float(
                loaded_calibration.payload["semantic_temporal_exposure"]
                ["state_thresholds"]["watch"]
            )
            if loaded_calibration is not None
            and "semantic_temporal_exposure" in loaded_calibration.payload
            else 0.1
        )
        row = analyze_run(
            spec,
            round_frame,
            client_frame,
            semantic_watch_threshold=watch_threshold,
        )
        if calibration_source and "semantic_risk" in client_frame:
            replay = replay_semantic_budget_triggers(
                rounds=round_frame,
                clients=client_frame,
                manifest=loaded_calibration,
            )
            row.update(
                {
                    "semantic_global_trigger_count_replayed": replay["global"],
                    "semantic_group_trigger_count_replayed": replay["group"],
                    "semantic_principal_trigger_count_replayed": replay["principal"],
                }
            )
            if "semantic_trigger_count" in row and (
                int(row["semantic_trigger_count"]) != int(replay["global"])
            ):
                raise ValueError(
                    "logged and replayed global semantic trigger counts disagree"
                )
        rows.append(row)
    result = pd.DataFrame(rows)
    join_keys = ["trial_plan_hash", "seed", "attack", "period"]
    if result.duplicated(join_keys + ["defense"]).any():
        raise ValueError("analysis contains duplicate strict statistical keys")
    return result
