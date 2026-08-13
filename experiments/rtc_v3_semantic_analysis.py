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


def _finite(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series([numeric], dtype=float)
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


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
        "asr_auc": (
            float(np.trapezoid(active_asr.to_numpy()))
            if len(active_asr)
            else np.nan
        ),
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
        stop = int(active_rounds.max())
        post = risk_by_round[risk_by_round.index > stop]
        recovered = post[post <= 0.05]
        result["time_to_recovery"] = (
            int(recovered.index.min()) - stop if len(recovered) else np.nan
        )
        if len(post):
            peak = float(post.max())
            half = post[post <= peak / 2.0]
            result["risk_release_half_life"] = (
                int(half.index.min()) - stop if len(half) else np.nan
            )
        else:
            result["risk_release_half_life"] = np.nan
    else:
        result["time_to_recovery"] = np.nan
        result["risk_release_half_life"] = np.nan
    result["false_persistence"] = (
        float((risk_mean > 0.0).mean())
        if str(spec["attack"]) == "none" and len(risk_mean)
        else np.nan
    )

    semantic_columns = [
        column for column in fit if column.startswith("fit_rtc_v3_semantic_active_")
    ]
    result["semantic_trigger_count"] = sum(
        int((_finite(fit[column]) > 0.0).sum()) for column in semantic_columns
    )
    for family, column in (
        ("cumulative", "fit_rtc_v3_cumulative_active_count"),
        ("direction", "fit_rtc_v3_direction_active_count"),
    ):
        result[f"{family}_trigger_count"] = int(_finite(fit.get(column)).sum())
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
        "fit_anchor_seconds",
        "fit_sketch_seconds",
        "fit_total_defense_seconds",
    ):
        values = _finite(fit.get(column))
        key = column.removeprefix("fit_")
        result[f"{key}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"{key}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
    return result


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
        rows.append(
            analyze_run(spec, pd.read_csv(round_path), pd.read_csv(client_path))
        )
    result = pd.DataFrame(rows)
    join_keys = ["trial_plan_hash", "seed", "attack", "period"]
    if result.duplicated(join_keys + ["defense"]).any():
        raise ValueError("analysis contains duplicate strict statistical keys")
    return result
