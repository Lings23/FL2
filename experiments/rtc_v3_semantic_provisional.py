"""Build a hash-bound, explicitly non-formal snapshot of completed cells.

This utility is for a long matrix that is still running.  It never treats a
partial root as formal evidence, even when all currently selected cells have
valid round artifacts.  The final auditor remains authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import CalibrationManifest
from experiments.periodic_attack import run_id
from experiments.rtc_v3_semantic_analysis import (
    analyze_root,
    analyze_run,
    replay_semantic_budget_triggers,
    strict_root_is_valid,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _completed_rows(root: Path, attack: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest_path = root / "experiment_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = payload.get("specs", payload)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = [
        {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)}
    ]
    for spec in specs:
        if str(spec["attack"]) != str(attack):
            continue
        identifier = run_id(dict(spec))
        round_path = root / "rounds" / f"{identifier}.csv"
        client_path = root / "raw" / f"{identifier}_clients.csv"
        if not round_path.is_file() or not client_path.is_file():
            continue
        rounds = pd.read_csv(round_path)
        clients = pd.read_csv(client_path)
        calibration = (spec.get("custom_params") or {}).get("calibration_path")
        loaded_calibration = CalibrationManifest.load(calibration) if calibration else None
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
            spec, rounds, clients, semantic_watch_threshold=watch_threshold
        )
        if calibration and "semantic_risk" in clients:
            replay = replay_semantic_budget_triggers(
                rounds=rounds,
                clients=clients,
                manifest=loaded_calibration,
            )
            row.update(
                {
                    "semantic_global_trigger_count_replayed": replay["global"],
                    "semantic_group_trigger_count_replayed": replay["group"],
                    "semantic_principal_trigger_count_replayed": replay["principal"],
                }
            )
            if int(row["semantic_trigger_count"]) != int(replay["global"]):
                raise ValueError("logged/replayed semantic global triggers disagree")
            violation = pd.to_numeric(
                rounds.get("fit_rtc_v3_max_constraint_violation"), errors="coerce"
            ).dropna()
            updates = pd.to_numeric(
                rounds.get("fit_rtc_v3_cone_update_applied_count"), errors="coerce"
            ).dropna()
            row["max_constraint_violation"] = (
                float(violation.max()) if len(violation) else np.nan
            )
            row["prototype_update_count"] = (
                int(updates.sum()) if len(updates) else -1
            )
            row["candidate_manifest_hashes"] = sorted(
                str(value)
                for value in rounds["fit_rtc_v3_calibration_hash"].dropna().unique()
            )
        rows.append(row)
        sources.extend(
            [
                {"path": str(round_path.resolve()), "sha256": _sha256(round_path)},
                {"path": str(client_path.resolve()), "sha256": _sha256(client_path)},
            ]
        )
    if not rows:
        raise FileNotFoundError(f"no completed cells for {attack} in {root}")
    return pd.DataFrame(rows), sources


def build_provisional_snapshot(
    *,
    attack_root: str | Path,
    baseline_root: str | Path,
    attack: str,
) -> dict[str, Any]:
    root = Path(attack_root).resolve()
    baseline_path = Path(baseline_root).resolve()
    completed, sources = _completed_rows(root, attack)
    candidate = completed[completed["defense"].astype(str) == "rtc_v3"]
    clip = completed[completed["defense"].astype(str) == "clip_only"]
    if len(candidate) != 1 or len(clip) != 1:
        raise ValueError("provisional snapshot needs exactly one candidate and clip-only cell")
    baseline = analyze_root(baseline_path)
    baseline = baseline[
        (baseline["defense"].astype(str) == "rtc_v3")
        & (baseline["attack"].astype(str) == str(attack))
        & (pd.to_numeric(baseline["seed"], errors="coerce") == int(candidate.iloc[0]["seed"]))
    ]
    if len(baseline) != 1:
        raise ValueError("provisional snapshot needs exactly one promoted V2 baseline")
    keys = ("trial_plan_hash", "seed", "attack")
    candidate_key = tuple(candidate.iloc[0][key] for key in keys)
    if candidate_key != tuple(clip.iloc[0][key] for key in keys):
        raise ValueError("candidate and clip-only provisional keys differ")
    if tuple(baseline.iloc[0][key] for key in keys) != candidate_key:
        raise ValueError("candidate and promoted V2 provisional keys differ")

    def metrics(row: Mapping[str, Any]) -> dict[str, Any]:
        names = (
            "active_mean_asr", "asr_auc", "asr_auc_raw", "asr_auc_normalized",
            "peak_asr", "final_accuracy",
            "best_accuracy", "last10_accuracy", "asr_k3_worst", "asr_k4_worst",
            "malicious_effective_weight", "zero_update_mass", "semantic_exposure",
            "clip_recall", "clip_fpr", "semantic_detection_auc", "detection_lag",
            "semantic_trigger_count", "semantic_global_trigger_count_replayed",
            "semantic_group_trigger_count_replayed",
            "semantic_principal_trigger_count_replayed",
            "semantic_total_seconds_mean", "semantic_total_seconds_p95",
            "total_defense_seconds_mean", "total_defense_seconds_p95",
            "max_constraint_violation", "prototype_update_count",
            "candidate_manifest_hashes",
        )
        result: dict[str, Any] = {}
        for name in names:
            if name not in row:
                continue
            value = row[name]
            if isinstance(value, (list, tuple)):
                result[name] = [_json_scalar(item) for item in value]
            else:
                result[name] = None if pd.isna(value) else _json_scalar(value)
        return result

    candidate_metrics = metrics(candidate.iloc[0])
    baseline_metrics = metrics(baseline.iloc[0])
    clip_metrics = metrics(clip.iloc[0])
    delta_names = (
        "active_mean_asr", "peak_asr", "final_accuracy", "asr_k3_worst",
        "asr_k4_worst", "malicious_effective_weight",
    )
    delta_v2 = {
        name: float(candidate_metrics[name] - baseline_metrics[name])
        for name in delta_names
    }
    delta_clip = {
        name: float(candidate_metrics[name] - clip_metrics[name])
        for name in delta_names
    }
    formal = bool(strict_root_is_valid(root))
    return {
        "artifact_kind": "rtc_v3_semantic_provisional_single_condition_v1",
        # A provisional artifact is never itself formal.  If strict validation
        # appears later, users must run the final audit instead.
        "formal_statistics_eligible": False,
        "attack_root_strict_validation_present_and_passed": formal,
        "reason": (
            "Use the final audit; provisional snapshots never enter formal statistics."
            if formal
            else "The complete attack root has not yet passed execution_validation."
        ),
        "join_key": {
            name: _json_scalar(value) for name, value in zip(keys, candidate_key)
        },
        "candidate": candidate_metrics,
        "promoted_v2": baseline_metrics,
        "clip_only": clip_metrics,
        "candidate_delta_vs_promoted_v2": delta_v2,
        "candidate_delta_vs_clip_only": delta_clip,
        "source_artifacts": sources,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_provisional_snapshot(
        attack_root=args.attack_root,
        baseline_root=args.baseline_root,
        attack=args.attack,
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    main()
