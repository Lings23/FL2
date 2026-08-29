"""Run the focused 60-round RTC versus FedAvg periodic-attack experiment.

The default matrix is deliberately small enough to audit and large enough to
cover both attack objectives:

* targeted: backdoor, DBA, model replacement, and targeted label flip;
* untargeted: all-label reverse, Byzantine, and Gaussian noise;
* default schedule: continuous_1_0;
* temporal ablations use short_1_1 and long_3_3 through the periodic profile;
* defenses: FedAvg and the promoted manifest-bound RTC-v3.

With one seed this produces 24 attacked runs plus two clean utility baselines.
The report is descriptive; it does not claim statistical significance.

``rtc_full`` and ``rtc_v3`` both select the promoted RTC-v3 implementation.
``rtc_v2_legacy`` remains available only through historical/explicit entrypoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.periodic_attack import (
    ABLATIONS,
    DEFENSES,
    PERIODS,
    LABEL_FLIP_ATTACKS,
    LABEL_FLIP_TARGETED,
    TARGETED_ATTACKS,
    UNTARGETED_ATTACKS,
    _run_specs,
    _objective_asr_series,
    RTC_V3_PROMOTED_MANIFEST,
    add_accuracy_drop,
    add_accuracy_drop_auc,
    attack_active,
    clean_baselines,
    generate_plots,
    run_id,
    validate_sampling_manifests,
    write_experiment_manifest,
)
from experiments.trial_plan import attach_trial_plans, paired_fedavg_clean_specs
from defenses.rtc.calibration import CalibrationManifest


logger = logging.getLogger(__name__)

ATTACK_GROUPS = {
    "targeted": tuple(TARGETED_ATTACKS),
    "untargeted": tuple(UNTARGETED_ATTACKS),
}
DEFAULT_ATTACKS = tuple((*TARGETED_ATTACKS, *UNTARGETED_ATTACKS))
DEFAULT_PERIODS = ("continuous_1_0",)
DEFAULT_DEFENSES = ("fedavg", "rtc_full")
RTC_V3_DEFENSE = "rtc_v3"
RTC_V3_DEFENSE_TYPE = "rtc_full"
RTC_COMPARISON_DEFENSES = ("rtc_full", RTC_V3_DEFENSE)
RTC_V3_SEMANTIC_ABLATIONS = (
    "semantic_observe",
    "semantic_soft",
    "semantic_temporal",
    "semantic_exposure",
)
RTC_V3_FAMILY = {"rtc_full", RTC_V3_DEFENSE, *RTC_V3_SEMANTIC_ABLATIONS}


def build_comparison_matrix(
    *,
    seed: int = 42,
    attacks: Sequence[str] = DEFAULT_ATTACKS,
    periods: Sequence[str] = DEFAULT_PERIODS,
    defenses: Sequence[str] = DEFAULT_DEFENSES,
    malicious_fraction: float = 0.2,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
    participation_rate: float = 0.5,
    num_clients: int = 20,
    boost_factor: float = 10.0,
    label_flip_source_label: int = 5,
    label_flip_target_label: int = 3,
    label_flip_poison_fraction: float = 1.0,
    attack_start_round: int = 11,
    attack_end_round: int = -1,
    rtc_v3_custom_params: Mapping[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Build the exact attacked matrix without formal-preflight exclusions."""
    if "label_flip" in attacks:
        raise ValueError(
            "Attack 'label_flip' was removed; use 'label_flip_targeted' or "
            "'label_flip_all_reverse'."
        )
    if set(defenses).intersection(RTC_V3_FAMILY) and rtc_v3_custom_params is None:
        rtc_v3_custom_params = build_rtc_v3_custom_params(
            manifest_path=RTC_V3_PROMOTED_MANIFEST,
            implementation_phase=6,
            num_clients=num_clients,
        )
    unknown_attacks = set(attacks) - set(DEFAULT_ATTACKS)
    unknown_periods = set(periods) - set(PERIODS)
    definitions = _defense_definitions(rtc_v3_custom_params)
    unknown_defenses = set(defenses) - set(definitions)
    if unknown_attacks:
        raise ValueError(f"Unsupported attacks: {sorted(unknown_attacks)}")
    if unknown_periods:
        raise ValueError(f"Unsupported periods: {sorted(unknown_periods)}")
    if unknown_defenses:
        raise ValueError(f"Unsupported defenses: {sorted(unknown_defenses)}")
    if not 0.0 < malicious_fraction < 1.0:
        raise ValueError("malicious_fraction must be between zero and one")
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("participation_rate must be in (0, 1]")

    group_by_attack = {
        attack: group for group, members in ATTACK_GROUPS.items() for attack in members
    }
    rows: list[Dict[str, Any]] = []
    for attack in attacks:
        for period_name in periods:
            on_rounds, off_rounds = PERIODS[period_name]
            for defense_name in defenses:
                defense_type, custom_params = definitions[defense_name]
                row = {
                    "benchmark_version": 3,
                    "attack_group": group_by_attack[attack],
                    "attack": attack,
                    "period": period_name,
                    "on_rounds": int(on_rounds),
                    "off_rounds": int(off_rounds),
                    "malicious_fraction": float(malicious_fraction),
                    "seed": int(seed),
                    "defense": defense_name,
                    "defense_type": defense_type,
                    "custom_params": dict(custom_params),
                    "attack_start_round": int(attack_start_round),
                    "attack_end_round": int(attack_end_round),
                    "partition": partition,
                    "dirichlet_alpha": float(dirichlet_alpha),
                    "participation_rate": float(participation_rate),
                }
                if attack in LABEL_FLIP_ATTACKS:
                    row["label_flip_poison_fraction"] = float(
                        label_flip_poison_fraction
                    )
                    if attack == LABEL_FLIP_TARGETED:
                        row.update({
                            "label_flip_source_label": int(label_flip_source_label),
                            "label_flip_target_label": int(label_flip_target_label),
                        })
                else:
                    row["boost_factor"] = float(boost_factor)
                rows.append(row)
    return rows


def add_clean_comparators(matrix: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    clean = clean_baselines(matrix, smoke=False)
    v3_clean: dict[tuple[Any, ...], Dict[str, Any]] = {}
    condition_keys = (
        "seed",
        "partition",
        "dirichlet_alpha",
        "participation_rate",
        "attack_start_round",
        "attack_end_round",
    )
    for spec in matrix:
        if spec["defense"] != RTC_V3_DEFENSE:
            continue
        key = tuple(spec.get(name) for name in condition_keys)
        if key in v3_clean:
            continue
        clean_spec = dict(spec)
        for attack_key in (
            "label_flip_source_label",
            "label_flip_target_label",
            "label_flip_poison_fraction",
        ):
            clean_spec.pop(attack_key, None)
        clean_spec.update({
            "attack_group": "clean",
            "attack": "none",
            "period": "clean",
            "on_rounds": 1,
            "off_rounds": 0,
            "malicious_fraction": 0.0,
            "boost_factor": 10.0,
        })
        v3_clean[key] = clean_spec
    clean.extend(v3_clean.values())
    for spec in clean:
        spec["attack_group"] = "clean"
    return [*matrix, *clean]


def build_pairwise_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Create one FedAvg/RTC row per attack schedule and selected RTC version."""
    attacked = summary[summary["attack"] != "none"].copy()
    keys = [
        "trial_plan_hash", "attack_group", "attack", "period", "seed",
        "malicious_fraction",
    ]
    keys = [key for key in keys if key in attacked]
    metrics = (
        "active_asr", "active_asr_auc", "peak_asr", "residual_asr",
        "active_accuracy_drop", "accuracy_drop_auc", "final_accuracy",
        "label_flip_global_exposure_rate", "label_flip_eligible_poison_rate",
        "label_flip_active_malicious_exposure_rate",
        "malicious_aggregation_weight_share", "benign_quarantine_rate",
        "detection_lag", "first_malicious_restricted_lag",
        "first_malicious_quarantined_lag",
    )
    rows: list[Dict[str, Any]] = []
    for group_key, frame in attacked.groupby(keys, dropna=False, sort=False):
        by_defense = frame.set_index("defense")
        if "fedavg" not in by_defense.index:
            continue
        rtc_defenses = [
            str(value) for value in frame["defense"].drop_duplicates()
            if str(value) in RTC_COMPARISON_DEFENSES
        ]
        for rtc_defense in rtc_defenses:
            row: Dict[str, Any] = dict(zip(keys, group_key))
            row["rtc_defense"] = rtc_defense
            for metric in metrics:
                if metric not in by_defense:
                    continue
                fedavg = _as_float(by_defense.at["fedavg", metric])
                rtc = _as_float(by_defense.at[rtc_defense, metric])
                row[f"fedavg_{metric}"] = fedavg
                row[f"rtc_{metric}"] = rtc
                row[f"rtc_minus_fedavg_{metric}"] = (
                    rtc - fedavg
                    if np.isfinite(rtc) and np.isfinite(fedavg)
                    else math.nan
                )
            if row["attack_group"] == "targeted":
                row["primary_metric"] = "active_asr_auc"
                row["rtc_over_fedavg"] = _safe_ratio(
                    row.get("rtc_active_asr_auc", math.nan),
                    row.get("fedavg_active_asr_auc", math.nan),
                )
            else:
                row["primary_metric"] = "active_accuracy_drop"
                row["rtc_over_fedavg"] = _safe_ratio(
                    row.get("rtc_active_accuracy_drop", math.nan),
                    row.get("fedavg_active_accuracy_drop", math.nan),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_category_report(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only across attack types, never across the two schedules."""
    if pairwise.empty:
        return pd.DataFrame()
    metrics = [
        column for column in (
            "fedavg_active_asr_auc", "rtc_active_asr_auc",
            "fedavg_active_accuracy_drop", "rtc_active_accuracy_drop",
            "fedavg_final_accuracy", "rtc_final_accuracy", "rtc_over_fedavg",
        )
        if column in pairwise
    ]
    group_keys = ["attack_group", "period"]
    if "rtc_defense" in pairwise:
        group_keys.append("rtc_defense")
    return pairwise.groupby(group_keys, as_index=False)[metrics].mean(
        numeric_only=True
    )


def validate_execution(
    specs: Sequence[Dict[str, Any]],
    rounds_dir: Path,
    raw_dir: Path,
    expected_rounds: int,
) -> list[Dict[str, Any]]:
    """Validate coverage, schedules, finite metrics, and RTC hard constraints."""
    rows: list[Dict[str, Any]] = []
    expected_ids = {run_id(spec) for spec in specs}
    paths = {path.stem: path for path in rounds_dir.glob("*.csv") if path.stem in expected_ids}
    rows.append(_validation(
        "matrix_complete", set(paths) == expected_ids,
        f"{len(paths)}/{len(expected_ids)} run files", f"{len(expected_ids)}/{len(expected_ids)}",
    ))

    finite_metrics = True
    complete_rounds = True
    schedules_match = True
    cap_violation = 0.0
    budget_violation = 0.0
    weight_mass = 0.0
    v3_constraint_violation = 0.0
    v3_weight_mass = 0.0
    v3_metrics_present = True
    v3_manifest_match = True
    has_v2 = any(spec["defense"] == "rtc_v2_legacy" for spec in specs)
    rtc_v3_family = RTC_V3_FAMILY
    has_v3 = any(spec["defense"] in rtc_v3_family for spec in specs)
    for spec in specs:
        path = paths.get(run_id(spec))
        if path is None:
            finite_metrics = complete_rounds = schedules_match = False
            continue
        frame = pd.read_csv(path)
        actual = set(pd.to_numeric(frame["round"], errors="coerce").dropna().astype(int))
        expected = set(range(0, expected_rounds + 1))
        expected_without_initial = set(range(1, expected_rounds + 1))
        complete_rounds &= actual in (expected, expected_without_initial)
        accuracy = _numeric_column(frame, "server_accuracy")
        finite_metrics &= bool(len(accuracy) and np.isfinite(accuracy).all())
        if spec["attack"] in LABEL_FLIP_ATTACKS or spec["attack_group"] == "targeted":
            # New runs report ``server_asr`` directly.  Pre-column
            # all-reverse runs remain exactly auditable because their immutable
            # confusion matrices encode the same y -> C-1-y objective.
            asr = _objective_asr_series(frame, spec)
            finite_metrics &= bool(
                asr is not None
                and len(asr) == len(frame)
                and np.isfinite(asr.to_numpy(dtype=float)).all()
            )
        planned = _numeric_column(frame, "planned_attack_active").fillna(0).astype(int)
        expected_active = frame["round"].astype(int).map(
            lambda rnd: int(spec["attack"] != "none" and attack_active(
                rnd, spec["attack_start_round"], spec["on_rounds"],
                spec["off_rounds"], spec["attack_end_round"],
            ))
        )
        schedules_match &= bool((planned == expected_active).all())
        if spec["defense"] == "rtc_v2_legacy":
            cap_violation = max(cap_violation, _finite_max(
                frame.get("fit_rtc_max_weight_cap_violation")
            ))
            budget_violation = max(budget_violation, _finite_max(
                frame.get("fit_rtc_max_exposure_budget_violation")
            ))
            weight_mass = max(weight_mass, _finite_max(
                frame.get("fit_rtc_aggregation_weight_sum")
            ))
        elif spec["defense"] in rtc_v3_family:
            required_v3_columns = {
                "fit_rtc_v3_calibration_hash",
                "fit_rtc_v3_max_constraint_violation",
                "fit_rtc_v3_weight_sum",
            }
            v3_metrics_present &= required_v3_columns.issubset(frame.columns)
            calibration_source = spec["custom_params"].get(
                "calibration_manifest",
                spec["custom_params"].get("calibration_path"),
            )
            expected_hash = CalibrationManifest.load(calibration_source).hash
            observed_hashes = (
                set(
                    frame["fit_rtc_v3_calibration_hash"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if "fit_rtc_v3_calibration_hash" in frame
                else set()
            )
            observed_hashes.discard("")
            v3_manifest_match &= observed_hashes == {expected_hash}
            v3_constraint_violation = max(v3_constraint_violation, _finite_max(
                frame.get("fit_rtc_v3_max_constraint_violation")
            ))
            v3_weight_mass = max(v3_weight_mass, _finite_max(
                frame.get("fit_rtc_v3_weight_sum")
            ))

    rows.extend([
        _validation("rounds_complete", complete_rounds, complete_rounds, f"rounds 1..{expected_rounds} (+ optional round 0)"),
        _validation("critical_metrics_finite", finite_metrics, finite_metrics, "all finite"),
        _validation("short_long_schedules_match", schedules_match, schedules_match, "true"),
    ])
    if has_v2:
        rows.extend([
            _validation("rtc_weight_cap_violation", cap_violation <= 1e-8, cap_violation, "<= 1e-8"),
            _validation("rtc_exposure_budget_violation", budget_violation <= 1e-8, budget_violation, "<= 1e-8"),
            _validation("rtc_subprobability_mass", weight_mass <= 1.0 + 1e-8, weight_mass, "<= 1 + 1e-8"),
        ])
    if has_v3:
        rows.extend([
            _validation(
                "rtc_v3_metrics_present", v3_metrics_present,
                v3_metrics_present, "true",
            ),
            _validation(
                "rtc_v3_manifest_hash_match", v3_manifest_match,
                v3_manifest_match, "true",
            ),
            _validation(
                "rtc_v3_constraint_violation",
                v3_constraint_violation <= 1e-8,
                v3_constraint_violation,
                "<= 1e-8",
            ),
            _validation(
                "rtc_v3_subprobability_mass",
                v3_weight_mass <= 1.0 + 1e-8,
                v3_weight_mass,
                "<= 1 + 1e-8",
            ),
        ])
    rows.extend(validate_sampling_manifests(rounds_dir, raw_dir, expected_ids=expected_ids))
    return rows


def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    attacks = _csv_subset(args.attacks, DEFAULT_ATTACKS)
    periods = _csv_subset(args.periods, PERIODS)
    definitions = tuple(
        (*DEFENSES, *ABLATIONS, RTC_V3_DEFENSE, *RTC_V3_SEMANTIC_ABLATIONS)
    )
    defenses = _csv_subset(getattr(args, "defenses", ",".join(DEFAULT_DEFENSES)), definitions)
    rtc_v3_custom_params = None
    if set(defenses).intersection(RTC_V3_FAMILY):
        rtc_v3_custom_params = build_rtc_v3_custom_params(
            manifest_path=getattr(args, "rtc_v3_manifest", ""),
            implementation_phase=getattr(args, "rtc_v3_phase", 6),
            num_clients=args.num_clients,
            principal_map_path=getattr(args, "rtc_v3_principal_map", ""),
            parameter_roles_path=getattr(args, "rtc_v3_parameter_roles", ""),
        )
    attacked = build_comparison_matrix(
        seed=args.seed,
        attacks=attacks,
        periods=periods,
        defenses=defenses,
        malicious_fraction=args.malicious_fraction,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        participation_rate=args.participation_rate,
        num_clients=args.num_clients,
        boost_factor=args.boost_factor,
        label_flip_source_label=args.label_flip_source_label,
        label_flip_target_label=args.label_flip_target_label,
        label_flip_poison_fraction=args.label_flip_poison_fraction,
        attack_start_round=args.attack_start_round,
        attack_end_round=args.attack_end_round,
        rtc_v3_custom_params=rtc_v3_custom_params,
    )
    output = Path(args.output)
    if str(getattr(args, "pairing_mode", "strict")) == "strict":
        attacked = attach_trial_plans(attacked, args, output)
        paired_clean = (
            []
            if bool(getattr(args, "skip_clean", False))
            else paired_fedavg_clean_specs(attacked)
        )
        specs = [*attacked, *paired_clean]
    else:
        specs = (
            attacked
            if bool(getattr(args, "skip_clean", False))
            else add_clean_comparators(attacked)
        )
    rounds_dir = output / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_manifest(specs, output)
    _matrix_frame(specs).to_csv(output / "experiment_matrix.csv", index=False)
    if args.dry_run:
        logger.info(
            "Dry run: %d attacked + %d clean runs, %d rounds each",
            len(attacked), len(specs) - len(attacked), args.rounds,
        )
        return pd.DataFrame()

    summary = _run_specs(specs, args, output, rounds_dir)
    summary = add_accuracy_drop_auc(add_accuracy_drop(summary), rounds_dir)
    summary.to_csv(output / "comparison_runs.csv", index=False)

    validation = validate_execution(specs, rounds_dir, output / "raw", args.rounds)
    pd.DataFrame(validation).to_csv(output / "execution_validation.csv", index=False)
    failed = [item for item in validation if not bool(item["passed"])]
    if failed:
        raise RuntimeError(
            "Experiment execution validation failed: "
            + "; ".join(str(item["gate"]) for item in failed)
        )
    pairwise = build_pairwise_report(summary)
    pairwise.to_csv(output / "fedavg_rtc_pairs.csv", index=False)
    build_category_report(pairwise).to_csv(output / "attack_group_summary.csv", index=False)
    generate_plots(summary, rounds_dir, output / "plots")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="60-round single-seed RTC versus FedAvg continuous-attack comparison"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="logs/rtc_fedavg_60round")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--periods", default=",".join(DEFAULT_PERIODS))
    parser.add_argument("--defenses", default=",".join(DEFAULT_DEFENSES))
    parser.add_argument(
        "--rtc-v3-manifest", default=str(RTC_V3_PROMOTED_MANIFEST),
        help="RTC-v3 manifest (defaults to the engineering-promoted manifest)",
    )
    parser.add_argument(
        "--rtc-v3-phase", type=int, default=6, choices=range(1, 7),
        help="RTC-v3 implementation phase (1-6)",
    )
    parser.add_argument(
        "--rtc-v3-principal-map", default="",
        help="Optional JSON client-to-principal mapping; defaults to one principal per client",
    )
    parser.add_argument(
        "--rtc-v3-parameter-roles", default="",
        help="Optional JSON parameter-index-to-role mapping used by calibration",
    )
    parser.add_argument("--malicious-fraction", type=float, default=0.2)
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
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Run attacked defense cells only; reuse a separately attested shared clean trajectory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write and validate the matrix without training")
    args = parser.parse_args(argv)
    args.smoke = False
    args.mode = "rtc_fedavg_comparison"
    args.skip_clean = bool(args.skip_clean)
    args.clean_only = False
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
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


def build_rtc_v3_custom_params(
    *,
    manifest_path: str | Path,
    implementation_phase: int,
    num_clients: int,
    principal_map_path: str | Path = "",
    parameter_roles_path: str | Path = "",
) -> Dict[str, Any]:
    """Validate and materialize the immutable RTC-v3 experiment contract."""
    if not str(manifest_path).strip():
        manifest_path = RTC_V3_PROMOTED_MANIFEST
    if not 1 <= int(implementation_phase) <= 6:
        raise ValueError("rtc_v3 implementation phase must be between 1 and 6")
    if int(num_clients) <= 0:
        raise ValueError("num_clients must be positive")

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if not resolved_manifest.is_file():
        raise FileNotFoundError(
            f"RTC-v3 calibration manifest not found: {resolved_manifest}. "
            "For a runtime-only bootstrap, run: "
            "python.exe experiments\\rtc_v3_manifest.py "
            f"--config config\\config.yaml --output \"{resolved_manifest}\""
        )
    manifest = CalibrationManifest.load(resolved_manifest)
    metadata = manifest.payload.get("metadata", {})
    if (
        isinstance(metadata, Mapping)
        and metadata.get("artifact_kind") == "rtc_v3_bootstrap"
    ):
        logger.warning(
            "RTC-v3 is using a bootstrap runtime-smoke manifest (%s). "
            "Its permissive limits are not valid for formal defense comparison.",
            manifest.hash,
        )
    principal_map = (
        _load_json_mapping(principal_map_path, wrapper_key="principal_map")
        if str(principal_map_path).strip()
        else {str(client_id): str(client_id) for client_id in range(int(num_clients))}
    )
    expected_clients = {str(client_id) for client_id in range(int(num_clients))}
    missing_clients = sorted(expected_clients.difference(principal_map))
    if missing_clients:
        raise ValueError(
            "RTC-v3 principal map is missing simulated client IDs: "
            f"{missing_clients}"
        )
    if any(not str(principal_map[client_id]).strip() for client_id in expected_clients):
        raise ValueError("RTC-v3 principal IDs must be non-empty strings")

    parameter_roles = (
        _load_json_mapping(parameter_roles_path, wrapper_key="parameter_roles")
        if str(parameter_roles_path).strip()
        else {}
    )
    if any(not str(value).strip() for value in parameter_roles.values()):
        raise ValueError("RTC-v3 parameter roles must be non-empty strings")

    stable_cones = manifest.payload.get("stable_cones", {})
    if isinstance(stable_cones, bool):
        stable_cones = {"enabled": stable_cones}
    return {
        "calibration_path": str(resolved_manifest),
        "sketch_algorithm_version": str(
            stable_cones.get(
                "sketch_algorithm_version", "legacy_blake2b_v1"
            )
        ),
        "implementation_phase": int(implementation_phase),
        "principal_map": {
            str(key): str(value) for key, value in principal_map.items()
        },
        # TrialPlanV1 sets this only after validating the actual plan.
        "principal_first_sampling_verified": False,
        "parameter_roles": {
            str(key): str(value) for key, value in parameter_roles.items()
        },
    }


def _defense_definitions(
    rtc_v3_custom_params: Mapping[str, Any] | None,
) -> Dict[str, tuple[str, Mapping[str, Any]]]:
    definitions: Dict[str, tuple[str, Mapping[str, Any]]] = {
        **DEFENSES,
        **ABLATIONS,
    }
    if rtc_v3_custom_params is not None:
        definitions["rtc_full"] = (
            RTC_V3_DEFENSE_TYPE,
            dict(rtc_v3_custom_params),
        )
        definitions[RTC_V3_DEFENSE] = (
            RTC_V3_DEFENSE_TYPE,
            dict(rtc_v3_custom_params),
        )
        for name, mode in (
            ("semantic_observe", "observe"),
            ("semantic_soft", "soft"),
            ("semantic_temporal", "temporal"),
            ("semantic_exposure", "exposure"),
        ):
            custom = dict(rtc_v3_custom_params)
            custom["semantic_ablation"] = mode
            definitions[name] = (RTC_V3_DEFENSE_TYPE, custom)
    return definitions


def _load_json_mapping(
    source: str | Path,
    *,
    wrapper_key: str,
) -> Dict[str, Any]:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RTC-v3 JSON mapping not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and wrapper_key in payload:
        payload = payload[wrapper_key]
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"RTC-v3 {wrapper_key} JSON must contain an object mapping"
        )
    return {str(key): value for key, value in payload.items()}


def _matrix_frame(specs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {**{key: value for key, value in spec.items() if key != "custom_params"},
         "custom_params": json.dumps(spec["custom_params"], sort_keys=True)}
        for spec in specs
    ])


def _csv_subset(raw: str, allowed: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if "label_flip" in requested:
        raise ValueError(
            "Attack 'label_flip' was removed; use 'label_flip_targeted' or "
            "'label_flip_all_reverse'."
        )
    unknown = set(requested) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported values: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one value is required")
    return requested


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if np.isfinite(result) else math.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    top, bottom = _as_float(numerator), _as_float(denominator)
    if not np.isfinite(top) or not np.isfinite(bottom):
        return math.nan
    return float(top / max(abs(bottom), 1e-12))


def _finite_max(values: Any) -> float:
    if values is None:
        return 0.0
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if len(array) else 0.0


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _validation(gate: str, passed: bool, observed: Any, required: str) -> Dict[str, Any]:
    return {"gate": gate, "passed": bool(passed), "observed": observed, "required": required}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_comparison(parse_args())
