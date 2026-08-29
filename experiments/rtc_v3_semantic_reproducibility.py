"""Strict reproducibility audit for two executions of one RTC-V3 cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.periodic_attack import run_id
from experiments.rtc_v3_semantic_analysis import strict_root_is_valid


EXACT_ROUND_COLUMNS = {
    "round",
    "planned_attack_active",
    "fit_trial_plan_hash",
    "fit_fit_seed_digest",
    "fit_rtc_v3_calibration_hash",
    "fit_rtc_v3_semantic_top_source",
    "fit_rtc_v3_semantic_top_target",
    "fit_rtc_v3_semantic_active_w1",
    "fit_rtc_v3_semantic_active_w4",
    "fit_rtc_v3_semantic_active_w8",
    "fit_rtc_v3_cone_update_applied_count",
    "fit_rtc_v3_cone_prototype_ids_full",
}
EXACT_CLIENT_COLUMNS = {
    "round",
    "partition_id",
    "principal_id",
    "fit_seed",
    "attack_seed",
    "is_malicious",
    "attack_active",
    "clipped",
    "semantic_top_source",
    "semantic_top_target",
    "semantic_signature_digest",
}
FLOAT_ROUND_COLUMNS = {
    "server_accuracy",
    "server_loss",
    "server_asr",
    "fit_rtc_v3_semantic_risk_mean",
    "fit_rtc_v3_semantic_exposure",
    "fit_rtc_v3_zero_update_mass",
    "fit_rtc_v3_total_mass",
    "fit_rtc_v3_max_constraint_violation",
}
FLOAT_CLIENT_COLUMNS = {
    "nominal_mass",
    "aggregation_weight",
    "semantic_raw_score",
    "semantic_z",
    "semantic_risk",
    "semantic_q",
}


def _single_rtc_spec(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    specs = [
        item for item in payload.get("specs", payload)
        if str(item.get("defense")) == "rtc_v3"
    ]
    if len(specs) != 1:
        raise ValueError(f"root must contain exactly one rtc_v3 cell: {root}")
    return specs[0]


def _compare_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    sort_keys: list[str],
    exact_columns: set[str],
    float_columns: set[str],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left = left.sort_values(sort_keys).reset_index(drop=True)
    right = right.sort_values(sort_keys).reset_index(drop=True)
    if len(left) != len(right):
        return {"passed": False, "row_count": [len(left), len(right)]}
    exact_mismatches = []
    for column in sorted(exact_columns & set(left) & set(right)):
        a = left[column].fillna("<NA>").astype(str)
        b = right[column].fillna("<NA>").astype(str)
        if not a.equals(b):
            exact_mismatches.append(column)
    missing_exact = sorted(
        column for column in exact_columns if column not in left or column not in right
    )
    float_differences: dict[str, float] = {}
    float_failures = []
    for column in sorted(float_columns & set(left) & set(right)):
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        mismatch_nan = np.isnan(a) != np.isnan(b)
        maximum = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        float_differences[column] = maximum
        if mismatch_nan.any() or not np.allclose(a[finite], b[finite], atol=atol, rtol=rtol):
            float_failures.append(column)
    missing_float = sorted(
        column for column in float_columns if column not in left or column not in right
    )
    return {
        "passed": not exact_mismatches
        and not missing_exact
        and not float_failures
        and not missing_float,
        "row_count": len(left),
        "exact_mismatches": exact_mismatches,
        "missing_exact_columns": missing_exact,
        "float_failures": float_failures,
        "missing_float_columns": missing_float,
        "max_absolute_differences": float_differences,
    }


def audit_reproducibility(
    left_root: str | Path,
    right_root: str | Path,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    left_root = Path(left_root).resolve()
    right_root = Path(right_root).resolve()
    if not strict_root_is_valid(left_root) or not strict_root_is_valid(right_root):
        raise RuntimeError("both reproducibility roots must pass all strict execution gates")
    left_spec = _single_rtc_spec(left_root)
    right_spec = _single_rtc_spec(right_root)
    keys = ("trial_plan_hash", "seed", "attack", "period")
    key_match = all(left_spec.get(key) == right_spec.get(key) for key in keys)
    left_id = run_id(dict(left_spec))
    right_id = run_id(dict(right_spec))
    left_rounds = pd.read_csv(left_root / "rounds" / f"{left_id}.csv")
    right_rounds = pd.read_csv(right_root / "rounds" / f"{right_id}.csv")
    left_clients = pd.read_csv(left_root / "raw" / f"{left_id}_clients.csv")
    right_clients = pd.read_csv(right_root / "raw" / f"{right_id}_clients.csv")
    round_check = _compare_frame(
        left_rounds, right_rounds,
        sort_keys=["round"], exact_columns=EXACT_ROUND_COLUMNS,
        float_columns=FLOAT_ROUND_COLUMNS, atol=atol, rtol=rtol,
    )
    client_check = _compare_frame(
        left_clients, right_clients,
        sort_keys=["round", "partition_id"], exact_columns=EXACT_CLIENT_COLUMNS,
        float_columns=FLOAT_CLIENT_COLUMNS, atol=atol, rtol=rtol,
    )
    passed = bool(key_match and round_check["passed"] and client_check["passed"])
    return {
        "artifact_kind": "rtc_v3_semantic_reproducibility_audit_v1",
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "statistical_key_match": key_match,
        "statistical_key": {key: left_spec.get(key) for key in keys},
        "rounds": round_check,
        "clients": client_check,
        "roots": [str(left_root), str(right_root)],
    }


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", required=True)
    parser.add_argument("--right-root", required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = audit_reproducibility(
        args.left_root, args.right_root, atol=args.atol, rtol=args.rtol
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not payload["passed"]:
        raise RuntimeError("RTC-V3 reproducibility audit failed")
    return destination


if __name__ == "__main__":
    main()

