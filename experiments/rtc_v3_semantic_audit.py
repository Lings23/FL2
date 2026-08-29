"""Fail-closed audit for the RTC-V3 semantic candidate.

The auditor only writes evidence and a Markdown report.  It intentionally has
no code path that creates a promotion attestation or changes promotion_ready.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import CalibrationManifest, SCHEMA_VERSION_V3
from experiments.rtc_v3_semantic_analysis import analyze_root

CLEAN_BENIGN_CLIPPING_MAX = 0.03
REQUIRED_EXTRA_HELDOUT_SEEDS = 4
REQUIRED_ABLATION_DEFENSES = {
    "fedavg",
    "clip_only",
    "rtc_v3_v2",
    "semantic_observe",
    "semantic_soft",
    "semantic_temporal",
    "semantic_exposure",
}


def _gate(
    name: str,
    passed: bool,
    observed: Any,
    expected: str,
    *,
    promotion_only: bool = False,
) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "promotion_only": bool(promotion_only),
    }


def _execution_gates(root: Path) -> tuple[bool, list[dict[str, Any]]]:
    path = root / "execution_validation.csv"
    if not path.is_file():
        return False, [_gate("execution_validation_present", False, str(path), "file")]
    frame = pd.read_csv(path)
    if not {"gate", "passed"}.issubset(frame):
        return False, [_gate("execution_validation_schema", False, list(frame), "gate,passed")]
    rows = []
    for _, row in frame.iterrows():
        value = str(row["passed"]).strip().lower() in {"true", "1", "1.0"}
        rows.append(_gate(f"strict:{row['gate']}", value, row["passed"], "true"))
    return bool(rows and all(row["passed"] for row in rows)), rows


def _round_frames(root: Path) -> pd.DataFrame:
    paths = sorted((root / "rounds").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no round files in {root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        source = pd.Series(path.name, index=frame.index, name="source_file")
        frames.append(pd.concat([frame, source], axis=1))
    return pd.concat(frames, ignore_index=True)


def _as_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths = [Path(item).resolve() for item in values]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("artifact roots must be non-empty and unique")
    return paths


def _attack_summary(
    roots: Path | Sequence[Path], defense: str
) -> pd.DataFrame:
    root_list = [roots] if isinstance(roots, Path) else list(roots)
    frames = []
    for root in root_list:
        # Recompute the audit-grade metrics from strict round/client artifacts.
        # The legacy comparison summary does not contain conditional worst-k
        # ASR and therefore cannot prove the preregistered acceptance gate.
        frame = analyze_root(root)
        if "active_mean_asr" in frame:
            frame = frame.rename(columns={"active_mean_asr": "active_asr"})
        selected = frame[
            (frame["defense"].astype(str) == str(defense))
            & (frame["attack"].astype(str) != "none")
        ].copy()
        selected["source_root"] = str(root)
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    keys = ["trial_plan_hash", "seed", "attack"]
    if result.duplicated(keys).any():
        raise ValueError(f"duplicate {defense} held-out statistical keys")
    return result


def _ablation_summary(roots: Sequence[Path]) -> pd.DataFrame:
    if not roots:
        return pd.DataFrame()
    frames = []
    for root in roots:
        frame = analyze_root(root)
        frames.append(frame[frame["attack"].astype(str) != "none"].copy())
    result = pd.concat(frames, ignore_index=True)
    keys = ["trial_plan_hash", "seed", "attack", "defense"]
    if result.duplicated(keys).any():
        raise ValueError("duplicate ablation statistical keys")
    return result


def _assemble_ablation_summary(
    *,
    dedicated: pd.DataFrame,
    candidate: pd.DataFrame,
    clip_only: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    reused = []
    for frame, defense in (
        (candidate, "semantic_exposure"),
        (clip_only, "clip_only"),
        (baseline, "rtc_v3_v2"),
    ):
        normalized = frame.copy()
        if "active_asr" in normalized and "active_mean_asr" not in normalized:
            normalized = normalized.rename(columns={"active_asr": "active_mean_asr"})
        normalized["defense"] = defense
        reused.append(normalized)
    result = pd.concat([dedicated, *reused], ignore_index=True)
    keys = ["trial_plan_hash", "seed", "attack", "defense"]
    if result.duplicated(keys).any():
        raise ValueError("duplicate assembled first-layer ablation keys")
    return result


def _clean_summary(root: Path, defense: str) -> pd.DataFrame:
    # Use the same audit-grade reconstruction as attacked/recovery runs.  The
    # legacy comparison CSV lacks best/last-10 accuracy and is therefore too
    # weak to prove the preregistered reporting contract.
    frame = analyze_root(root)
    selected = frame[frame["defense"].astype(str) == str(defense)].copy()
    if len(selected) != 1:
        raise ValueError(
            f"clean root must contain exactly one {defense} trajectory: {root}"
        )
    return selected


def _clean_rounds_are_inactive(root: Path) -> bool:
    rounds = _round_frames(root)
    active = pd.to_numeric(
        rounds.get("planned_attack_active"), errors="coerce"
    ).fillna(0.0)
    return bool((active <= 0.0).all())


def _promotion_neutral_payload(manifest: CalibrationManifest) -> dict[str, Any]:
    """Return the defense payload with attestation-only metadata removed."""
    payload = copy.deepcopy(manifest.payload)
    payload.pop("content_hash", None)
    metadata = payload.get("metadata", {})
    for key in ("promotion_ready", "promotion"):
        metadata.pop(key, None)
    return payload


def _runtime_manifest_hashes(roots: Sequence[Path]) -> set[str]:
    hashes: set[str] = set()
    for root in roots:
        rounds = _round_frames(root)
        if "fit_rtc_v3_calibration_hash" not in rounds:
            continue
        hashes.update(
            str(value)
            for value in rounds["fit_rtc_v3_calibration_hash"].dropna().unique()
            if str(value).strip()
        )
    return hashes


def _frozen_cone_runtime_matches(
    roots: Sequence[Path], manifest: CalibrationManifest
) -> tuple[bool, list[dict[str, Any]]]:
    stable = manifest.payload.get("stable_cones", {})
    expected: dict[str, tuple[int, str]] = {}
    for phase, profile in stable.get("phases", {}).items():
        identifiers = sorted(
            str(item["cone_id"]) for item in profile.get("prototypes", [])
        )
        expected[str(phase)] = (len(identifiers), ",".join(identifiers))
    mismatches: list[dict[str, Any]] = []
    for root in roots:
        rounds = _round_frames(root)
        if "fit_rtc_v3_calibration_hash" not in rounds:
            continue
        selected = rounds[rounds["fit_rtc_v3_calibration_hash"].notna()]
        required = {
            "fit_rtc_v3_cone_profile",
            "fit_rtc_v3_cone_prototype_count",
            "fit_rtc_v3_cone_prototype_ids_full",
            "fit_rtc_v3_cone_online_updates_enabled",
            "fit_rtc_v3_cone_update_applied_count",
        }
        if not required.issubset(selected.columns):
            mismatches.append(
                {"root": str(root), "missing_columns": sorted(required - set(selected))}
            )
            continue
        for _, row in selected.iterrows():
            phase = str(row["fit_rtc_v3_cone_profile"])
            expected_phase = expected.get(phase)
            observed_ids = ",".join(
                sorted(
                    part.strip()
                    for part in str(row["fit_rtc_v3_cone_prototype_ids_full"]).split(",")
                    if part.strip()
                )
            )
            valid = bool(
                expected_phase is not None
                and int(float(row["fit_rtc_v3_cone_prototype_count"]))
                == expected_phase[0]
                and observed_ids == expected_phase[1]
                and float(row["fit_rtc_v3_cone_online_updates_enabled"]) == 0.0
                and float(row["fit_rtc_v3_cone_update_applied_count"]) == 0.0
            )
            if not valid:
                mismatches.append(
                    {
                        "root": str(root),
                        "round": row.get("round"),
                        "phase": phase,
                        "observed_count": row["fit_rtc_v3_cone_prototype_count"],
                        "observed_ids": observed_ids,
                    }
                )
    return not mismatches, mismatches[:20]


def audit_candidate(
    *,
    candidate_manifest: str | Path,
    baseline_manifest: str | Path,
    promoted_v2_manifest: str | Path,
    calibration_validation: str | Path,
    candidate_validation: str | Path,
    performance_gates: str | Path,
    clean_root: str | Path,
    baseline_clean_root: str | Path,
    attack_root: str | Path | Sequence[str | Path],
    recovery_root: str | Path,
    baseline_attack_root: str | Path | Sequence[str | Path],
    audit_stage: str = "promotion",
    excluded_runs: str | Path | None = None,
    reproduction_doc: str | Path | None = None,
    ablation_root: str | Path | Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    if audit_stage not in {"seed42", "promotion"}:
        raise ValueError("audit_stage must be seed42 or promotion")
    manifest = CalibrationManifest.load(candidate_manifest)
    baseline_manifest_loaded = CalibrationManifest.load(baseline_manifest)
    promoted_v2 = CalibrationManifest.load(promoted_v2_manifest)
    metadata = manifest.payload.get("metadata", {})
    calibration = json.loads(Path(calibration_validation).read_text(encoding="utf-8"))
    frozen = json.loads(Path(candidate_validation).read_text(encoding="utf-8"))
    performance = json.loads(Path(performance_gates).read_text(encoding="utf-8"))
    predevelopment_hash = str(
        metadata.get("development_provenance", {}).get(
            "predevelopment_candidate_hash", ""
        )
    )
    clean_path = Path(clean_root)
    baseline_clean_path = Path(baseline_clean_root)
    attack_paths = _as_paths(attack_root)
    recovery_path = Path(recovery_root)
    baseline_paths = _as_paths(baseline_attack_root)
    ablation_paths = _as_paths(ablation_root) if ablation_root else []
    excluded_path = Path(excluded_runs).resolve() if excluded_runs else None
    excluded_payload = (
        json.loads(excluded_path.read_text(encoding="utf-8"))
        if excluded_path is not None and excluded_path.is_file()
        else None
    )
    reproduction_path = Path(reproduction_doc).resolve() if reproduction_doc else None
    gates: list[dict[str, Any]] = [
        _gate(
            "schema_v3",
            manifest.payload["schema_version"] == SCHEMA_VERSION_V3,
            manifest.payload["schema_version"],
            SCHEMA_VERSION_V3,
        ),
        _gate(
            "candidate_not_promotion_ready",
            not bool(metadata.get("promotion_ready", False)),
            metadata.get("promotion_ready", False),
            "false",
        ),
        _gate(
            "baseline_functionally_equals_promoted_v2",
            _promotion_neutral_payload(baseline_manifest_loaded)
            == _promotion_neutral_payload(promoted_v2),
            {
                "historical_baseline_hash": baseline_manifest_loaded.hash,
                "promoted_v2_hash": promoted_v2.hash,
            },
            "payloads equal after removing attestation-only metadata and content hash",
        ),
        _gate(
            "formal_calibration_passed",
            bool(calibration.get("passed", False))
            and str(calibration.get("content_hash", "")) == predevelopment_hash,
            {
                "passed": calibration.get("passed", False),
                "content_hash": calibration.get("content_hash"),
                "expected_predevelopment_hash": predevelopment_hash,
            },
            "passed=true and content_hash equals predevelopment hash",
        ),
        _gate(
            "frozen_candidate_validation",
            bool(frozen.get("passed", False))
            and str(frozen.get("content_hash", "")) == manifest.hash
            and not bool(frozen.get("promotion_ready", True)),
            {
                "passed": frozen.get("passed", False),
                "content_hash": frozen.get("content_hash"),
                "candidate_hash": manifest.hash,
                "promotion_ready": frozen.get("promotion_ready"),
            },
            "passed=true, candidate hash bound, promotion_ready=false",
        ),
        _gate(
            "performance_passed",
            bool(performance.get("passed", False)),
            performance.get("passed", False),
            "true",
        ),
        _gate(
            "semantic_mean_performance",
            np.isfinite(float(performance.get("semantic_mean_seconds", np.nan)))
            and float(performance.get("semantic_mean_seconds", np.nan)) <= 0.15,
            performance.get("semantic_mean_seconds"),
            "<=0.15s",
        ),
        _gate(
            "semantic_p95_performance",
            np.isfinite(float(performance.get("semantic_p95_seconds", np.nan)))
            and float(performance.get("semantic_p95_seconds", np.nan)) <= 0.25,
            performance.get("semantic_p95_seconds"),
            "<=0.25s",
        ),
        _gate(
            "defense_mean_performance_increase",
            np.isfinite(
                float(performance.get("defense_mean_increase_fraction", np.nan))
            )
            and float(performance.get("defense_mean_increase_fraction", np.nan))
            <= 0.05,
            performance.get("defense_mean_increase_fraction"),
            "<=0.05",
        ),
        _gate(
            "defense_p95_performance_increase",
            np.isfinite(
                float(performance.get("defense_p95_increase_fraction", np.nan))
            )
            and float(performance.get("defense_p95_increase_fraction", np.nan))
            <= 0.10,
            performance.get("defense_p95_increase_fraction"),
            "<=0.10",
        ),
    ]
    strict_roots = [
        ("clean", clean_path),
        ("baseline_clean", baseline_clean_path),
        ("recovery", recovery_path),
    ]
    strict_roots.extend(
        (f"attacks:{index}", root)
        for index, root in enumerate(attack_paths, start=1)
    )
    strict_roots.extend(
        (f"baseline_attacks:{index}", root)
        for index, root in enumerate(baseline_paths, start=1)
    )
    strict_roots.extend(
        (f"ablations:{index}", root)
        for index, root in enumerate(ablation_paths, start=1)
    )
    for label, root in strict_roots:
        passed, strict = _execution_gates(root)
        gates.append(_gate(f"{label}_all_strict_gates", passed, passed, "true"))
        gates.extend(strict)
    candidate_runtime_hashes = _runtime_manifest_hashes(
        [clean_path, recovery_path, *attack_paths]
    )
    baseline_runtime_hashes = _runtime_manifest_hashes(
        [baseline_clean_path, *baseline_paths]
    )
    cones_frozen, cone_mismatches = _frozen_cone_runtime_matches(
        [clean_path, recovery_path, *attack_paths], manifest
    )
    gates.extend(
        [
            _gate(
                "candidate_runtime_manifest_hash",
                candidate_runtime_hashes == {manifest.hash},
                sorted(candidate_runtime_hashes),
                manifest.hash,
            ),
            _gate(
                "baseline_runtime_manifest_hash",
                baseline_runtime_hashes == {baseline_manifest_loaded.hash},
                sorted(baseline_runtime_hashes),
                baseline_manifest_loaded.hash,
            ),
            _gate(
                "runtime_cone_set_frozen",
                cones_frozen,
                cone_mismatches,
                "every RTC round matches the phase manifest IDs/count with updates disabled and zero applied updates",
            ),
        ]
    )

    clean = _round_frames(clean_path)
    fit_clean = clean[
        (pd.to_numeric(clean["round"], errors="coerce") > 0)
        & clean.get("fit_rtc_v3_calibration_hash").notna()
    ]
    overflow = pd.to_numeric(
        fit_clean.get("fit_rtc_v3_cone_overflow_assignments"), errors="coerce"
    ).fillna(0.0)
    principals = pd.to_numeric(
        fit_clean.get("fit_rtc_v3_principal_count"), errors="coerce"
    ).replace(0.0, np.nan)
    overflow_rate = float((overflow / principals).mean())
    updates = float(
        pd.to_numeric(
            fit_clean.get("fit_rtc_v3_cone_update_applied_count"), errors="coerce"
        ).fillna(0.0).max()
    )
    violation = float(
        pd.to_numeric(
            fit_clean.get("fit_rtc_v3_max_constraint_violation"), errors="coerce"
        ).fillna(np.inf).max()
    )
    clipping_rate = float(
        pd.to_numeric(
            fit_clean.get("fit_rtc_v3_clipping_rate"), errors="coerce"
        ).mean()
    )
    clean_semantic_active_columns = [
        column
        for column in fit_clean
        if column.startswith("fit_rtc_v3_semantic_active_w")
    ]
    clean_semantic_activation_events = int(
        (
            fit_clean[clean_semantic_active_columns]
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0.0
        ).sum()
    ) if clean_semantic_active_columns else -1
    gates.extend(
        [
            _gate("clean_overflow", overflow_rate <= 0.05, overflow_rate, "<=0.05"),
            _gate("online_prototypes_frozen", updates == 0.0, updates, "0"),
            _gate("clean_budget_violation", violation <= 1e-8, violation, "<=1e-8"),
            _gate(
                "clean_benign_clipping",
                clipping_rate <= CLEAN_BENIGN_CLIPPING_MAX,
                clipping_rate,
                f"<={CLEAN_BENIGN_CLIPPING_MAX}",
            ),
            _gate(
                "clean_semantic_budget_activation",
                clean_semantic_activation_events == 0,
                clean_semantic_activation_events,
                "0 hard-budget activation events",
            ),
        ]
    )
    candidate_clean = _clean_summary(clean_path, "rtc_v3")
    baseline_clean = _clean_summary(baseline_clean_path, "rtc_v3")
    # Reuse the shared FedAvg clean counterfactual already paired in the V2
    # root; rerunning the identical trajectory would waste a full experiment.
    fedavg_clean = _clean_summary(baseline_clean_path, "fedavg")
    clean_keys = ["seed", "trial_plan_hash"]
    clean_comparison = candidate_clean.merge(
        baseline_clean,
        on=clean_keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    clean_comparison = clean_comparison.merge(
        fedavg_clean[[*clean_keys, "final_accuracy"]],
        on=clean_keys,
        how="left",
        validate="one_to_one",
    )
    clean_complete = bool(len(clean_comparison) == 1)
    clean_inactive = bool(
        _clean_rounds_are_inactive(clean_path)
        and _clean_rounds_are_inactive(baseline_clean_path)
    )
    clean_accuracy_delta = (
        float(
            clean_comparison.iloc[0]["final_accuracy_candidate"]
            - clean_comparison.iloc[0]["final_accuracy_baseline"]
        )
        if clean_complete
        else float("nan")
    )
    clean_accuracy_delta_fedavg = (
        float(
            clean_comparison.iloc[0]["final_accuracy_candidate"]
            - clean_comparison.iloc[0]["final_accuracy"]
        )
        if clean_complete
        else float("nan")
    )
    clean_false_persistence = (
        float(clean_comparison.iloc[0]["false_persistence_candidate"])
        if clean_complete
        else float("nan")
    )
    clean_false_soft_downweight = (
        float(clean_comparison.iloc[0]["false_soft_downweight_rate_candidate"])
        if clean_complete
        else float("nan")
    )
    clean_false_principal_rate = (
        float(clean_comparison.iloc[0]["false_persistent_principal_rate_candidate"])
        if clean_complete
        else float("nan")
    )
    clean_false_max_streak = (
        int(clean_comparison.iloc[0]["false_persistence_max_observation_streak_candidate"])
        if clean_complete
        else -1
    )
    gates.extend(
        [
            _gate(
                "clean_strict_plan_join",
                clean_complete,
                int(len(clean_comparison)),
                "exactly one candidate/V2/FedAvg row joined by seed+trial_plan_hash",
            ),
            _gate(
                "clean_no_attack_rounds",
                clean_inactive,
                clean_inactive,
                "planned_attack_active=0 for every candidate/V2/FedAvg round",
            ),
            _gate(
                "clean_final_accuracy_vs_v2",
                np.isfinite(clean_accuracy_delta) and clean_accuracy_delta >= -0.01,
                clean_accuracy_delta,
                ">=-0.01",
            ),
            _gate(
                "clean_final_accuracy_vs_fedavg",
                np.isfinite(clean_accuracy_delta_fedavg)
                and clean_accuracy_delta_fedavg >= -0.01,
                clean_accuracy_delta_fedavg,
                ">=-0.01",
            ),
            _gate(
                "clean_false_persistence_observation_rate",
                np.isfinite(clean_false_persistence)
                and clean_false_persistence <= 0.01,
                clean_false_persistence,
                "<=0.01 benign observations at or above manifest watch threshold",
            ),
            _gate(
                "clean_false_persistence_max_streak",
                clean_false_max_streak <= 1,
                clean_false_max_streak,
                "<=1 consecutive actual observation for every benign principal",
            ),
        ]
    )

    candidate = _attack_summary(attack_paths, "rtc_v3")
    baseline = _attack_summary(baseline_paths, "rtc_v3")
    clip_only = _attack_summary(attack_paths, "clip_only")
    keys = ["trial_plan_hash", "attack", "seed"]
    candidate_keys = set(map(tuple, candidate[keys].astype(str).to_numpy()))
    baseline_keys = set(map(tuple, baseline[keys].astype(str).to_numpy()))
    clip_keys = set(map(tuple, clip_only[keys].astype(str).to_numpy()))
    attacked_key_sets_match = bool(
        candidate_keys
        and candidate_keys == baseline_keys
        and candidate_keys == clip_keys
    )
    gates.append(
        _gate(
            "attacked_strict_statistical_keys_match",
            attacked_key_sets_match,
            {
                "candidate": sorted(candidate_keys),
                "v2": sorted(baseline_keys),
                "clip_only": sorted(clip_keys),
            },
            "identical trial_plan_hash+attack+seed sets",
        )
    )
    comparison = candidate.merge(
        baseline,
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    comparison = comparison.merge(
        clip_only[
            [
                *keys,
                "active_asr",
                "peak_asr",
                "final_accuracy",
                "asr_k3_worst",
                "asr_k4_worst",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_clip"),
    )
    gates.append(
        _gate(
            "attacked_strict_join_complete",
            attacked_key_sets_match and len(comparison) == len(candidate),
            {"joined": len(comparison), "candidate": len(candidate)},
            "all candidate rows joined one-to-one to V2 and clip-only",
        )
    )
    expected_attacks = {"label_flip_targeted", "label_flip_all_reverse"}
    ablations = _ablation_summary(ablation_paths)
    ablations = _assemble_ablation_summary(
        dedicated=ablations,
        candidate=candidate,
        clip_only=clip_only,
        baseline=baseline,
    )
    candidate_seed42_plans = {
        str(row["attack"]): str(row["trial_plan_hash"])
        for _, row in candidate[pd.to_numeric(candidate["seed"], errors="coerce") == 42].iterrows()
    }
    observed_ablation_keys = {
        (str(row["attack"]), str(row["defense"]))
        for _, row in ablations[
            pd.to_numeric(ablations.get("seed"), errors="coerce") == 42
        ].iterrows()
    } if len(ablations) else set()
    expected_ablation_keys = {
        (attack, defense)
        for attack in expected_attacks
        for defense in REQUIRED_ABLATION_DEFENSES
    }
    ablation_plan_mismatches = [
        {
            "attack": str(row["attack"]),
            "defense": str(row["defense"]),
            "observed": str(row["trial_plan_hash"]),
            "expected": candidate_seed42_plans.get(str(row["attack"])),
        }
        for _, row in ablations[
            pd.to_numeric(ablations.get("seed"), errors="coerce") == 42
        ].iterrows()
        if str(row["trial_plan_hash"])
        != candidate_seed42_plans.get(str(row["attack"]))
    ] if len(ablations) else []
    gates.extend(
        [
            _gate(
                "seed42_first_layer_ablation_complete",
                observed_ablation_keys == expected_ablation_keys,
                sorted(observed_ablation_keys),
                str(sorted(expected_ablation_keys)),
                promotion_only=True,
            ),
            _gate(
                "seed42_first_layer_ablation_plan_match",
                not ablations.empty and not ablation_plan_mismatches,
                ablation_plan_mismatches,
                "all ablations use the candidate TrialPlan for the same attack",
                promotion_only=True,
            ),
        ]
    )
    observed_attacks = set(candidate["attack"].astype(str))
    observed_seeds = set(int(value) for value in candidate["seed"])
    seed42_expected_keys = {
        ("42", attack) for attack in sorted(expected_attacks)
    }
    seed42_key_set = {
        (str(seed), str(attack))
        for seed, attack in candidate[["seed", "attack"]].to_numpy()
        if int(seed) == 42
    }
    attacks_by_seed = {
        int(seed): set(group["attack"].astype(str))
        for seed, group in candidate.groupby("seed", sort=True)
    }
    complete_extra_seeds = sorted(
        seed
        for seed, attacks in attacks_by_seed.items()
        if seed != 42 and attacks == expected_attacks
    )
    gates.extend(
        [
            _gate(
                "heldout_attacks_complete",
                bool(attacks_by_seed)
                and all(attacks == expected_attacks for attacks in attacks_by_seed.values()),
                {str(seed): sorted(attacks) for seed, attacks in attacks_by_seed.items()},
                f"every seed has exactly {sorted(expected_attacks)}",
                promotion_only=True,
            ),
            _gate(
                "seed42_attacks_complete",
                seed42_key_set == seed42_expected_keys,
                sorted(seed42_key_set),
                str(sorted(seed42_expected_keys)),
            ),
            _gate(
                "four_extra_heldout_seeds_present",
                len(complete_extra_seeds) >= REQUIRED_EXTRA_HELDOUT_SEEDS,
                complete_extra_seeds,
                (
                    "seed 42 plus at least "
                    f"{REQUIRED_EXTRA_HELDOUT_SEEDS} other held-out seeds"
                ),
                promotion_only=True,
            ),
        ]
    )
    metric_rows: list[dict[str, Any]] = []
    detailed_metric_names = (
        "asr_auc",
        "asr_auc_raw",
        "asr_auc_normalized",
        "best_accuracy",
        "last10_accuracy",
        "active_accuracy",
        "asr_k3_mean",
        "asr_k3_worst",
        "asr_k4_mean",
        "asr_k4_worst",
        "malicious_effective_weight",
        "zero_update_mass",
        "semantic_exposure",
        "clip_recall",
        "clip_fpr",
        "semantic_detection_auc",
        "detection_lag",
        "false_persistence",
        "semantic_trigger_count",
        "cumulative_trigger_count",
        "direction_trigger_count",
        "principal_trigger_count",
        "server_trigger_count",
        "cone_trigger_count",
        "prototype_contamination_events",
        "semantic_extraction_seconds_mean",
        "semantic_extraction_seconds_p95",
        "semantic_temporal_seconds_mean",
        "semantic_temporal_seconds_p95",
        "semantic_group_seconds_mean",
        "semantic_group_seconds_p95",
        "semantic_ledger_seconds_mean",
        "semantic_ledger_seconds_p95",
        "semantic_total_seconds_mean",
        "semantic_total_seconds_p95",
        "solver_seconds_mean",
        "solver_seconds_p95",
        "anchor_seconds_mean",
        "anchor_seconds_p95",
        "sketch_seconds_mean",
        "sketch_seconds_p95",
        "total_defense_seconds_mean",
        "total_defense_seconds_p95",
    )
    for _, row in comparison.iterrows():
        asr_delta = float(row["active_asr_candidate"] - row["active_asr_baseline"])
        peak_delta = float(row["peak_asr_candidate"] - row["peak_asr_baseline"])
        accuracy_delta = float(
            row["final_accuracy_candidate"] - row["final_accuracy_baseline"]
        )
        asr_delta_clip = float(row["active_asr_candidate"] - row["active_asr"])
        peak_delta_clip = float(row["peak_asr_candidate"] - row["peak_asr"])
        plans_match = bool(str(row["trial_plan_hash"]).strip())
        item = {
            "attack": str(row["attack"]),
            "seed": int(row["seed"]),
            "trial_plan_hash": str(row["trial_plan_hash"]),
            "candidate_active_asr": float(row["active_asr_candidate"]),
            "baseline_active_asr": float(row["active_asr_baseline"]),
            "clip_only_active_asr": float(row["active_asr"]),
            "candidate_peak_asr": float(row["peak_asr_candidate"]),
            "baseline_peak_asr": float(row["peak_asr_baseline"]),
            "clip_only_peak_asr": float(row["peak_asr"]),
            "candidate_final_accuracy": float(row["final_accuracy_candidate"]),
            "baseline_final_accuracy": float(row["final_accuracy_baseline"]),
            "clip_only_final_accuracy": float(row["final_accuracy"]),
            "active_asr_delta": asr_delta,
            "peak_asr_delta": peak_delta,
            "final_accuracy_delta": accuracy_delta,
            "active_asr_delta_vs_clip": asr_delta_clip,
            "peak_asr_delta_vs_clip": peak_delta_clip,
            "trial_plan_hash_match": plans_match,
            "candidate_metrics": {
                metric: float(row[f"{metric}_candidate"])
                for metric in detailed_metric_names
                if f"{metric}_candidate" in row.index
                and np.isfinite(row[f"{metric}_candidate"])
            },
        }
        metric_rows.append(item)
        prefix = f"{item['attack']}:seed{item['seed']}"
        promotion_only = item["seed"] != 42
        gates.append(
            _gate(
                f"{prefix}:plan_hash",
                plans_match,
                plans_match,
                "true",
                promotion_only=promotion_only,
            )
        )
        if str(row["attack"]) == "label_flip_targeted":
            candidate_k_worst = [
                float(row[column])
                for column in ("asr_k3_worst_candidate", "asr_k4_worst_candidate")
                if column in row.index and np.isfinite(row[column])
            ]
            baseline_k_worst = [
                float(row[column])
                for column in ("asr_k3_worst_baseline", "asr_k4_worst_baseline")
                if column in row.index and np.isfinite(row[column])
            ]
            clip_k_worst = [
                float(row[column])
                for column in ("asr_k3_worst", "asr_k4_worst")
                if column in row.index and np.isfinite(row[column])
            ]
            worst_k_delta = (
                max(candidate_k_worst) - max(baseline_k_worst)
                if candidate_k_worst and baseline_k_worst
                else float("nan")
            )
            worst_k_delta_clip = (
                max(candidate_k_worst) - max(clip_k_worst)
                if candidate_k_worst and clip_k_worst
                else float("nan")
            )
            gates.extend(
                [
                    _gate(f"{prefix}:active_asr_vs_v2", asr_delta <= -0.10, asr_delta, "<=-0.10", promotion_only=promotion_only),
                    _gate(f"{prefix}:peak_asr_vs_v2", peak_delta <= -0.15, peak_delta, "<=-0.15", promotion_only=promotion_only),
                    _gate(f"{prefix}:active_asr_vs_clip", asr_delta_clip <= -0.10, asr_delta_clip, "<=-0.10", promotion_only=promotion_only),
                    _gate(f"{prefix}:peak_asr_vs_clip", peak_delta_clip <= -0.15, peak_delta_clip, "<=-0.15", promotion_only=promotion_only),
                    _gate(f"{prefix}:accuracy", accuracy_delta >= -0.01, accuracy_delta, ">=-0.01", promotion_only=promotion_only),
                    _gate(
                        f"{prefix}:worst_k3_k4_asr",
                        np.isfinite(worst_k_delta) and worst_k_delta < 0.0,
                        worst_k_delta,
                        "<0 (strictly lower worst k=3/4 ASR than V2)",
                        promotion_only=promotion_only,
                    ),
                    _gate(
                        f"{prefix}:worst_k3_k4_asr_vs_clip",
                        np.isfinite(worst_k_delta_clip) and worst_k_delta_clip < 0.0,
                        worst_k_delta_clip,
                        "<0 (strictly lower worst k=3/4 ASR than clip-only)",
                        promotion_only=promotion_only,
                    ),
                ]
            )
        else:
            gates.extend(
                [
                    _gate(f"{prefix}:active_asr", asr_delta <= 0.02, asr_delta, "<=+0.02", promotion_only=promotion_only),
                    _gate(f"{prefix}:accuracy", accuracy_delta >= -0.01, accuracy_delta, ">=-0.01", promotion_only=promotion_only),
                ]
            )

    attack_rounds = pd.concat(
        [_round_frames(root) for root in attack_paths], ignore_index=True
    )
    candidate_attack_rounds = attack_rounds[
        attack_rounds.get("fit_rtc_v3_calibration_hash").notna()
        & (pd.to_numeric(attack_rounds.get("planned_attack_active"), errors="coerce") > 0)
    ]
    attack_active_columns = [
        column
        for column in candidate_attack_rounds
        if column.startswith("fit_rtc_v3_semantic_active_w")
    ]
    attack_activation_events = int(
        (
            candidate_attack_rounds[attack_active_columns]
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0.0
        ).sum()
    ) if attack_active_columns else 0
    attack_updates = float(
        pd.to_numeric(
            candidate_attack_rounds.get("fit_rtc_v3_cone_update_applied_count"),
            errors="coerce",
        ).fillna(0.0).max()
    )
    attack_violation = float(
        pd.to_numeric(
            candidate_attack_rounds.get("fit_rtc_v3_max_constraint_violation"),
            errors="coerce",
        ).fillna(np.inf).max()
    )
    gates.extend(
        [
            _gate(
                "attack_semantic_budget_activation",
                attack_activation_events > clean_semantic_activation_events,
                attack_activation_events,
                f">clean {clean_semantic_activation_events} activation events",
            ),
            _gate("attack_prototype_contamination", attack_updates == 0.0, attack_updates, "0"),
            _gate("attack_budget_violation", attack_violation <= 1e-8, attack_violation, "<=1e-8"),
        ]
    )

    recovery = _round_frames(recovery_path)
    recovery_metrics = analyze_root(recovery_path)
    recovery_candidate = recovery_metrics[
        (recovery_metrics["defense"].astype(str) == "rtc_v3")
        & (recovery_metrics["attack"].astype(str) == "label_flip_targeted")
        & (pd.to_numeric(recovery_metrics["seed"], errors="coerce") == 42)
    ]
    recovery_metric_complete = bool(len(recovery_candidate) == 1)
    time_to_recovery = (
        float(recovery_candidate.iloc[0]["time_to_recovery"])
        if recovery_metric_complete
        else float("nan")
    )
    risk_release_half_life = (
        float(recovery_candidate.iloc[0]["risk_release_half_life"])
        if recovery_metric_complete
        else float("nan")
    )
    recovery_principal_count = (
        int(recovery_candidate.iloc[0]["recovery_principal_count"])
        if recovery_metric_complete
        else 0
    )
    recovered_principal_count = (
        int(recovery_candidate.iloc[0]["recovered_principal_count"])
        if recovery_metric_complete
        else 0
    )
    half_life_principal_count = (
        int(recovery_candidate.iloc[0]["half_life_principal_count"])
        if recovery_metric_complete
        else 0
    )
    recovery_segments: list[pd.Series] = []
    for _, run in recovery.groupby("source_file", sort=True):
        ordered = run.sort_values("round")
        active = pd.to_numeric(
            ordered.get("planned_attack_active"), errors="coerce"
        ).fillna(0.0)
        active_rounds = pd.to_numeric(
            ordered.loc[active > 0.0, "round"], errors="coerce"
        ).dropna()
        if active_rounds.empty:
            continue
        post = ordered[
            pd.to_numeric(ordered["round"], errors="coerce")
            > int(active_rounds.max())
        ]
        series = pd.to_numeric(
            post.get("fit_rtc_v3_semantic_risk_mean"), errors="coerce"
        ).dropna()
        if len(series) >= 10:
            recovery_segments.append(series.reset_index(drop=True))
    recovery_declines = bool(
        recovery_segments
        and all(
            float(series.iloc[-5:].mean()) <= float(series.iloc[:5].mean())
            for series in recovery_segments
        )
    )
    first5 = (
        float(np.mean([series.iloc[:5].mean() for series in recovery_segments]))
        if recovery_segments
        else None
    )
    last5 = (
        float(np.mean([series.iloc[-5:].mean() for series in recovery_segments]))
        if recovery_segments
        else None
    )
    gates.append(
        _gate(
            "controlled_recovery_risk_declines",
            recovery_declines,
            {
                "segments": len(recovery_segments),
                "first5": first5,
                "last5": last5,
            },
            "last5<=first5 with >=10 observed rounds",
        )
    )
    gates.extend(
        [
            _gate(
                "recovery_strict_metric_row",
                recovery_metric_complete,
                int(len(recovery_candidate)),
                "exactly one seed42 targeted RTC-V3 recovery trajectory",
            ),
            _gate(
                "time_to_recovery_observed",
                np.isfinite(time_to_recovery) and time_to_recovery >= 0.0,
                time_to_recovery,
                "finite non-negative server-round delay",
            ),
            _gate(
                "risk_release_half_life_observed",
                np.isfinite(risk_release_half_life)
                and risk_release_half_life >= 0.0,
                risk_release_half_life,
                "finite non-negative server-round delay",
            ),
            _gate(
                "all_attacking_principals_recovered",
                recovery_principal_count > 0
                and recovered_principal_count == recovery_principal_count,
                {
                    "attacking": recovery_principal_count,
                    "recovered": recovered_principal_count,
                },
                "every attacking principal has the required consecutive low-risk observations",
            ),
            _gate(
                "all_attacking_principals_reached_half_life",
                recovery_principal_count > 0
                and half_life_principal_count == recovery_principal_count,
                {
                    "attacking": recovery_principal_count,
                    "halved": half_life_principal_count,
                },
                "every attacking principal reaches half its own last active risk",
            ),
        ]
    )
    seed42_passed = bool(
        gates
        and all(item["passed"] for item in gates if not item["promotion_only"])
    )
    promotion_evidence_passed = bool(
        gates and all(item["passed"] for item in gates)
    )
    passed = seed42_passed if audit_stage == "seed42" else promotion_evidence_passed
    return {
        "artifact_kind": "rtc_v3_semantic_temporal_exposure_gate_audit",
        "candidate_manifest_hash": manifest.hash,
        "audit_stage": audit_stage,
        "passed": passed,
        "seed42_passed": seed42_passed,
        "promotion_evidence_passed": promotion_evidence_passed,
        "promotion_attestation_generated": False,
        "artifact_paths": {
            "candidate_manifest": str(Path(candidate_manifest).resolve()),
            "candidate_validation": str(Path(candidate_validation).resolve()),
            "calibration_validation": str(Path(calibration_validation).resolve()),
            "baseline_manifest": str(Path(baseline_manifest).resolve()),
            "promoted_v2_manifest": str(Path(promoted_v2_manifest).resolve()),
            "performance_gates": str(Path(performance_gates).resolve()),
            "clean_root": str(clean_path.resolve()),
            "baseline_clean_root": str(baseline_clean_path.resolve()),
            "attack_roots": [str(path.resolve()) for path in attack_paths],
            "baseline_attack_roots": [str(path.resolve()) for path in baseline_paths],
            "ablation_roots": [str(path.resolve()) for path in ablation_paths],
            "recovery_root": str(recovery_path.resolve()),
            "excluded_runs": str(excluded_path) if excluded_path else None,
            "reproduction_doc": str(reproduction_path) if reproduction_path else None,
        },
        "excluded_runs": excluded_payload,
        "performance": performance,
        "clean_comparison": (
            {
                "seed": int(clean_comparison.iloc[0]["seed"]),
                "trial_plan_hash": str(clean_comparison.iloc[0]["trial_plan_hash"]),
                "candidate_final_accuracy": float(
                    clean_comparison.iloc[0]["final_accuracy_candidate"]
                ),
                "candidate_best_accuracy": float(
                    clean_comparison.iloc[0]["best_accuracy_candidate"]
                ),
                "candidate_last10_accuracy": float(
                    clean_comparison.iloc[0]["last10_accuracy_candidate"]
                ),
                "baseline_final_accuracy": float(
                    clean_comparison.iloc[0]["final_accuracy_baseline"]
                ),
                "fedavg_final_accuracy": float(
                    clean_comparison.iloc[0]["final_accuracy"]
                ),
                "candidate_delta_vs_v2": clean_accuracy_delta,
                "candidate_delta_vs_fedavg": clean_accuracy_delta_fedavg,
                "overflow_rate": overflow_rate,
                "benign_clipping_rate": clipping_rate,
                "semantic_budget_activation_events": clean_semantic_activation_events,
                "false_persistence": clean_false_persistence,
                "false_soft_downweight_rate": clean_false_soft_downweight,
                "false_persistent_principal_rate": clean_false_principal_rate,
                "false_persistence_max_observation_streak": clean_false_max_streak,
            }
            if clean_complete
            else None
        ),
        "attack_runtime": {
            "semantic_budget_activation_events": attack_activation_events,
            "prototype_update_max": attack_updates,
            "max_constraint_violation": attack_violation,
        },
        "ablation_metrics": ablations.replace({np.nan: None}).to_dict("records"),
        "gates": gates,
        "attack_comparison": metric_rows,
        "recovery_metrics": (
            recovery_candidate.iloc[0].replace({np.nan: None}).to_dict()
            if recovery_metric_complete
            else None
        ),
    }


def write_report(evidence: Mapping[str, Any], destination: Path) -> None:
    performance = evidence.get("performance", {})
    clean = evidence.get("clean_comparison")
    attacks = evidence.get("attack_comparison", [])
    recovery = evidence.get("recovery_metrics")
    runtime = evidence.get("attack_runtime", {})
    ablations = evidence.get("ablation_metrics", [])
    excluded = evidence.get("excluded_runs") or {}
    artifact_paths = evidence.get("artifact_paths", {})
    lines = [
        "# RTC-V3 语义—时间—暴露闭环实验报告",
        "",
        "## 审计结论",
        "",
        f"- Candidate hash: `{evidence['candidate_manifest_hash']}`",
        f"- Audit stage: `{evidence['audit_stage']}`",
        f"- 当前阶段门禁通过: `{str(bool(evidence['passed'])).lower()}`",
        f"- Seed-42 前置门禁通过: `{str(bool(evidence['seed42_passed'])).lower()}`",
        f"- Promotion 全证据门禁通过: `{str(bool(evidence['promotion_evidence_passed'])).lower()}`",
        "- Promotion attestation generated: `false`",
        "",
        "候选保持 `promotion_ready=false`；审计器只生成证据和报告，不会生成 promotion attestation 或改写 promoted V2。",
        "",
        "## 性能门禁",
        "",
        "| 指标 | 观测值 | 门槛 |",
        "|---|---:|---:|",
        f"| semantic total mean | {performance.get('semantic_mean_seconds', 'N/A')} s | <=0.15 s |",
        f"| semantic total P95 | {performance.get('semantic_p95_seconds', 'N/A')} s | <=0.25 s |",
        f"| defense mean 相对变化 | {performance.get('defense_mean_increase_fraction', 'N/A')} | <=0.05 |",
        f"| defense P95 相对变化 | {performance.get('defense_p95_increase_fraction', 'N/A')} | <=0.10 |",
        "",
    ]
    if clean:
        lines.extend(
            [
                "## Seed 42 clean",
                "",
                f"严格连接键：`{clean['trial_plan_hash']} + seed {clean['seed']} + clean`。",
                "",
                "| Candidate final/best/last-10 accuracy | V2 final | FedAvg final | Δ vs V2 | Δ vs FedAvg | Overflow | Benign clipping | Hard triggers | False watch rate/max streak | Soft downweight rate |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {clean['candidate_final_accuracy']:.6f} / "
                    f"{clean['candidate_best_accuracy']:.6f} / "
                    f"{clean['candidate_last10_accuracy']:.6f} | "
                    f"{clean['baseline_final_accuracy']:.6f} | "
                    f"{clean['fedavg_final_accuracy']:.6f} | "
                    f"{clean['candidate_delta_vs_v2']:.6f} | "
                    f"{clean['candidate_delta_vs_fedavg']:.6f} | "
                    f"{clean['overflow_rate']:.6f} | "
                    f"{clean['benign_clipping_rate']:.6f} | "
                    f"{clean['semantic_budget_activation_events']} | "
                    f"{clean['false_persistence']:.6f} / "
                    f"{clean['false_persistence_max_observation_streak']} | "
                    f"{clean['false_soft_downweight_rate']:.6f} |"
                ),
                "",
            ]
        )
    if attacks:
        lines.extend(
            [
                "## Held-out attack comparison",
                "",
                "每行由 `trial_plan_hash + seed + attack condition` 一对一连接；严格门禁失败的运行不会进入此表。",
                "",
                "| Attack | Seed | Candidate ASR | V2 ASR | Clip-only ASR | Candidate peak | V2 peak | Clip peak | Candidate accuracy | V2 accuracy | Clip accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in attacks:
            lines.append(
                f"| {row['attack']} | {row['seed']} | "
                f"{row['candidate_active_asr']:.6f} | {row['baseline_active_asr']:.6f} | {row['clip_only_active_asr']:.6f} | "
                f"{row['candidate_peak_asr']:.6f} | {row['baseline_peak_asr']:.6f} | {row['clip_only_peak_asr']:.6f} | "
                f"{row['candidate_final_accuracy']:.6f} | {row['baseline_final_accuracy']:.6f} | {row['clip_only_final_accuracy']:.6f} |"
            )
        lines.extend(
            [
                "",
                "### Candidate diagnostic metrics",
                "",
                "| Attack | Seed | ASR-AUC raw/normalized | Best/last-10 acc | k3 mean/worst | k4 mean/worst | Malicious weight | Clip recall/FPR | Semantic AUC/lag | Zero-update mass | Semantic exposure |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in attacks:
            metrics = row.get("candidate_metrics", {})
            value = lambda name: metrics.get(name, "N/A")
            lines.append(
                f"| {row['attack']} | {row['seed']} | "
                f"{value('asr_auc_raw')} / {value('asr_auc_normalized')} | "
                f"{value('best_accuracy')} / {value('last10_accuracy')} | "
                f"{value('asr_k3_mean')} / {value('asr_k3_worst')} | "
                f"{value('asr_k4_mean')} / {value('asr_k4_worst')} | "
                f"{value('malicious_effective_weight')} | "
                f"{value('clip_recall')} / {value('clip_fpr')} | "
                f"{value('semantic_detection_auc')} / {value('detection_lag')} | "
                f"{value('zero_update_mass')} | "
                f"{value('semantic_exposure')} |"
            )
        lines.extend(
            [
                "",
                "### Candidate module timing and trigger counts",
                "",
                "| Attack | Seed | Semantic mean/P95 | Solver mean/P95 | Anchor mean/P95 | Sketch mean/P95 | Defense mean/P95 | Semantic/Cumulative/Direction/Cone triggers |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in attacks:
            metrics = row.get("candidate_metrics", {})
            value = lambda name: metrics.get(name, "N/A")
            lines.append(
                f"| {row['attack']} | {row['seed']} | "
                f"{value('semantic_total_seconds_mean')} / {value('semantic_total_seconds_p95')} | "
                f"{value('solver_seconds_mean')} / {value('solver_seconds_p95')} | "
                f"{value('anchor_seconds_mean')} / {value('anchor_seconds_p95')} | "
                f"{value('sketch_seconds_mean')} / {value('sketch_seconds_p95')} | "
                f"{value('total_defense_seconds_mean')} / {value('total_defense_seconds_p95')} | "
                f"{value('semantic_trigger_count')} / {value('cumulative_trigger_count')} / "
                f"{value('direction_trigger_count')} / {value('cone_trigger_count')} |"
            )
        lines.extend(
            [
                "",
                f"攻击期 semantic budget 触发事件：`{runtime.get('semantic_budget_activation_events', 'N/A')}`；",
                f"prototype update 最大值：`{runtime.get('prototype_update_max', 'N/A')}`；",
                f"最大预算违反量：`{runtime.get('max_constraint_violation', 'N/A')}`。",
                "",
            ]
        )
    if recovery:
        lines.extend(
            [
                "## Attack-stop recovery",
                "",
                f"- Time-to-recovery: `{recovery.get('time_to_recovery')}` server rounds",
                f"- Risk-release half-life: `{recovery.get('risk_release_half_life')}` server rounds",
                f"- Recovered attacking principals: `{recovery.get('recovered_principal_count')}/{recovery.get('recovery_principal_count')}`",
                f"- Principals reaching half-life: `{recovery.get('half_life_principal_count')}/{recovery.get('recovery_principal_count')}`",
                f"- False persistence: `{recovery.get('false_persistence')}`",
                f"- Active mean / peak ASR: `{recovery.get('active_mean_asr')}` / `{recovery.get('peak_asr')}`",
                f"- Final / best / last-10 accuracy: `{recovery.get('final_accuracy')}` / `{recovery.get('best_accuracy')}` / `{recovery.get('last10_accuracy')}`",
                "",
            ]
        )
    if ablations:
        lines.extend(
            [
                "## 第一层消融",
                "",
                "| Attack | Seed | Defense | Active ASR | Peak ASR | Final accuracy | Malicious weight | Semantic exposure |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            ablations,
            key=lambda item: (
                str(item.get("attack")),
                int(item.get("seed", 0)),
                str(item.get("defense")),
            ),
        ):
            lines.append(
                f"| {row.get('attack')} | {row.get('seed')} | {row.get('defense')} | "
                f"{row.get('active_mean_asr')} | {row.get('peak_asr')} | "
                f"{row.get('final_accuracy')} | {row.get('malicious_effective_weight')} | "
                f"{row.get('semantic_exposure')} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 完整门禁",
            "",
            "| Gate | Scope | Passed | Observed | Expected |",
            "|---|---|---:|---|---|",
        ]
    )
    for gate in evidence["gates"]:
        observed = json.dumps(gate["observed"], ensure_ascii=False).replace("|", "\\|")
        scope = "promotion" if gate["promotion_only"] else "seed42"
        lines.append(
            f"| {gate['gate']} | {scope} | {str(bool(gate['passed'])).lower()} | "
            f"`{observed}` | `{gate['expected']}` |"
        )
    excluded_items = excluded.get("runs", [])
    if excluded_items:
        lines.extend(
            [
                "",
                "## 失败与排除运行",
                "",
                "| 状态 | Last round | 是否纳入正式证据 | 原因 |",
                "|---|---:|---:|---|",
            ]
        )
        for item in excluded_items:
            reason = str(item.get("reason", "")).replace("|", "\\|")
            lines.append(
                f"| {item.get('state')} | {item.get('last_round')} | "
                f"{str(bool(item.get('included_in_formal_evidence', False))).lower()} | {reason} |"
            )
    if artifact_paths:
        lines.extend(["", "## 关键产物", ""])
        for label, value in artifact_paths.items():
            if isinstance(value, list):
                for item in value:
                    lines.append(f"- `{label}`: `{item}`")
            elif value:
                lines.append(f"- `{label}`: `{value}`")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "当前审计阶段全部门禁通过；本工具仍不会生成 promotion attestation。"
                if evidence["passed"]
                else "至少一个当前阶段门禁失败；保留 candidate，并禁止生成 promotion attestation。"
            ),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--promoted-v2-manifest", required=True)
    parser.add_argument("--calibration-validation", required=True)
    parser.add_argument("--candidate-validation", required=True)
    parser.add_argument("--performance-gates", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--baseline-clean-root", required=True)
    parser.add_argument("--attack-root", action="append", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--baseline-attack-root", action="append", required=True)
    parser.add_argument(
        "--audit-stage",
        choices=("seed42", "promotion"),
        default="promotion",
    )
    parser.add_argument(
        "--evidence-output",
        default="logs/rtc_v3_semantic_gate_audit.json",
    )
    parser.add_argument(
        "--report-output",
        default="docs/rtc_v3_semantic_temporal_exposure_experiment_report.md",
    )
    parser.add_argument("--excluded-runs")
    parser.add_argument("--reproduction-doc")
    parser.add_argument("--ablation-root", action="append")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    evidence = audit_candidate(
        candidate_manifest=args.candidate_manifest,
        baseline_manifest=args.baseline_manifest,
        promoted_v2_manifest=args.promoted_v2_manifest,
        calibration_validation=args.calibration_validation,
        candidate_validation=args.candidate_validation,
        performance_gates=args.performance_gates,
        clean_root=args.clean_root,
        baseline_clean_root=args.baseline_clean_root,
        attack_root=args.attack_root,
        recovery_root=args.recovery_root,
        baseline_attack_root=args.baseline_attack_root,
        audit_stage=args.audit_stage,
        excluded_runs=args.excluded_runs,
        reproduction_doc=args.reproduction_doc,
        ablation_root=args.ablation_root,
    )
    evidence_path = Path(args.evidence_output).resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(evidence, Path(args.report_output).resolve())
    if not evidence["passed"]:
        raise RuntimeError("semantic candidate gates failed; attestation is forbidden")
    return evidence_path


if __name__ == "__main__":
    main()
