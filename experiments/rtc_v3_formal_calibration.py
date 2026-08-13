"""Collect independent clean RTC-v3 trajectories and freeze a formal manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from defenses.rtc.calibration import (
    CalibrationManifest,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    build_manifest,
    content_hash,
    model_metadata_hash,
)
from defenses.rtc.cone_calibration import calibrate_offline_cones
from experiments.periodic_attack import (
    _deduplicate,
    _run_specs,
    run_id,
    write_experiment_manifest,
)
from experiments.rtc_fedavg_comparison import build_rtc_v3_custom_params
from experiments.trial_plan import attach_trial_plans
from defenses.rtc.semantic import ClassifierHeadLayout, ordered_class_pairs
from models.model_factory import (
    get_model,
    get_parameter_names,
    get_parameter_roles,
    get_parameters,
)


CALIBRATION_SEEDS = (40, 41)
DEVELOPMENT_SEEDS = (43, 44, 45)
EVALUATION_SEEDS = (42, 46, 47, 48, 51)
MARGIN = 1.10
SERVER_WINDOWS = (1, 4, 8)
PRINCIPAL_WINDOWS = (1, 4, 8)


def build_semantic_collection_bootstrap(
    *,
    base_manifest: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Derive a non-promotable V3 clean-collection contract from formal V2."""

    base = CalibrationManifest.load(base_manifest)
    cfg = load_config(str(config_path))
    model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
        dataset_name=cfg.dataset.name,
    )
    params = get_parameters(model)
    roles = get_parameter_roles(model)
    names = get_parameter_names(model)
    layout = ClassifierHeadLayout.build(params, roles)
    phases = base.payload.get("training_phases") or [
        {"start_round": 1, "profile": str(base.payload["profile"])}
    ]
    phase_names = [str(value["profile"]) for value in phases]
    pair_count = layout.num_classes * (layout.num_classes - 1)
    semantic = {
        "enabled": True,
        "num_classes": layout.num_classes,
        "phases": {
            profile: {
                "row_scales": [1.0] * layout.num_classes,
                "pair_centers": [0.0] * pair_count,
                "pair_scales": [1.0] * pair_count,
                "head_exposure_scale": 1.0,
            }
            for profile in phase_names
        },
        "temporal": {
            "decay_rate": 0.0,
            "kappa": 1.0,
            "threshold": 1e12,
            "eta": 1.0,
            "q_min": 0.999999,
            "recovery_threshold": 0.0,
            "recovery_observations": 3,
            "recovery_factor": 1.0,
            "min_history_observations": 3,
            "synchronization": {
                "enabled": False,
                "min_distinct_principals": 2,
                "evidence_threshold": 1e12,
                "similarity_threshold": 1.0,
                "multiplier": 1.0,
            },
        },
        "risk_lambda": 0.0,
        "state_thresholds": {"watch": 0.10, "restricted": 0.50, "quarantined": 0.80},
        "hard_exposure_risk_floor": 0.0,
        "top_k_pairs": min(4, pair_count - 1),
        "global_windows": list(SERVER_WINDOWS),
        "global_budgets": {str(value): 1e12 for value in SERVER_WINDOWS},
        "principal_windows": list(PRINCIPAL_WINDOWS),
        "principal_budgets": {str(value): 1e12 for value in PRINCIPAL_WINDOWS},
        "pair_windows": list(SERVER_WINDOWS),
        "pair_budgets": {str(value): 1e12 for value in SERVER_WINDOWS},
        "performance_budget": {
            "semantic_mean_seconds": 0.15,
            "semantic_p95_seconds": 0.25,
            "defense_mean_increase_fraction": 0.05,
            "defense_p95_increase_fraction": 0.10,
        },
    }
    payload = dict(base.payload)
    payload.pop("content_hash", None)
    payload["schema_version"] = SCHEMA_VERSION_V3
    payload["model_metadata_hash"] = model_metadata_hash(params, roles)
    payload["classifier_head_contract"] = layout.contract(params, names)
    payload["semantic_temporal_exposure"] = semantic
    stable = dict(payload.get("stable_cones") or {})
    stable["online_updates_enabled"] = False
    payload["stable_cones"] = stable
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "artifact_kind": "rtc_v3_semantic_clean_collection_bootstrap",
            "formal_evaluation_ready": False,
            "promotion_ready": False,
            "base_manifest_hash": base.hash,
        }
    )
    payload["metadata"] = metadata
    payload["content_hash"] = content_hash(payload)
    CalibrationManifest.load(payload)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    archive = destination.with_name(
        f"rtc_v3_semantic_collection_bootstrap_{payload['content_hash'][:12]}.json"
    )
    if archive.exists():
        archived = json.loads(archive.read_text(encoding="utf-8"))
        if archived.get("content_hash") != payload["content_hash"]:
            raise ValueError("collection bootstrap archive collision")
    else:
        archive.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return destination


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _collection_observation_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the controls that can change a clean calibration observation.

    Collection manifests can differ in descriptive classifier resolutions or
    reporting-only settings while producing the same full row/pair sidecar.  We
    preserve their distinct content hashes and additionally require this
    observation contract to match across calibration seeds.
    """

    classifier = dict(payload.get("classifier_head_contract") or {})
    classifier.pop("contract_hash", None)
    classifier.pop("semantic_resolutions", None)
    semantic = dict(payload.get("semantic_temporal_exposure") or {})
    temporal = dict(semantic.get("temporal") or {})
    synchronization = dict(temporal.get("synchronization") or {})
    if float(semantic.get("risk_lambda", float("nan"))) != 0.0:
        raise ValueError("clean collection manifest must use neutral semantic risk")
    if float(temporal.get("q_min", float("nan"))) < 0.999999:
        raise ValueError("clean collection manifest must use neutral temporal caps")
    if bool(synchronization.get("enabled", False)):
        raise ValueError("clean collection manifest must disable synchronization caps")
    stable = dict(payload.get("stable_cones") or {})
    if bool(stable.get("online_updates_enabled", False)):
        raise ValueError("clean collection manifest must freeze stable cones")

    # These settings affect reporting/ledger allocation only while risk_lambda
    # is zero and every budget is intentionally non-binding.  The complete 90
    # ordered-pair sidecar is exported independently of Top-K.
    semantic.pop("top_k_pairs", None)
    semantic.pop("state_thresholds", None)
    semantic.pop("hard_exposure_risk_floor", None)
    semantic.pop("performance_budget", None)
    temporal.pop("min_history_observations", None)
    semantic["temporal"] = temporal
    return {
        "schema": "rtc_v3.semantic_collection_observation.v1",
        "model_metadata_hash": payload.get("model_metadata_hash"),
        "server_optimizer": payload.get("server_optimizer"),
        "validated_mass_policy": payload.get("validated_mass_policy"),
        "principal_mapping_version": payload.get("principal_mapping_version"),
        "clip_lower": payload.get("clip_lower"),
        "clip_upper": payload.get("clip_upper"),
        "resolutions": payload.get("resolutions"),
        "residual_scales": payload.get("residual_scales"),
        "server_windows": payload.get("server_windows"),
        "server_budgets": payload.get("server_budgets"),
        "principal_windows": payload.get("principal_windows"),
        "principal_budgets": payload.get("principal_budgets"),
        "training_phases": payload.get("training_phases"),
        "stable_cones": stable,
        "classifier_head_contract": classifier,
        "semantic_temporal_exposure": semantic,
    }


def _profiles(rounds: int) -> list[dict[str, Any]]:
    candidates = [(1, "warmup"), (11, "steady"), (31, "late")]
    return [
        {"start_round": start, "profile": profile}
        for start, profile in candidates if start <= int(rounds)
    ]


def _profile_for_round(round_number: int, phases: Sequence[Mapping[str, Any]]) -> str:
    selected = str(phases[0]["profile"])
    for phase in phases:
        if int(round_number) >= int(phase["start_round"]):
            selected = str(phase["profile"])
    return selected


def build_clean_specs(
    *,
    seeds: Sequence[int],
    bootstrap_manifest: str | Path,
    num_clients: int,
    participation_rate: float,
    partition: str,
    dirichlet_alpha: float,
    parameter_roles: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    custom = build_rtc_v3_custom_params(
        manifest_path=bootstrap_manifest,
        implementation_phase=6,
        num_clients=num_clients,
    )
    custom["export_sketches"] = True
    custom["parameter_roles"] = {
        str(key): str(value) for key, value in (parameter_roles or {}).items()
    }
    return [
        {
            "benchmark_version": 3,
            "attack_group": "calibration_clean",
            "attack": "none",
            "period": "calibration_clean",
            "on_rounds": 1,
            "off_rounds": 0,
            "malicious_fraction": 0.0,
            "seed": int(seed),
            "defense": "rtc_v3",
            "defense_type": "rtc_v3_candidate",
            "custom_params": dict(custom),
            "attack_start_round": 1,
            "attack_end_round": -1,
            "partition": partition,
            "dirichlet_alpha": float(dirichlet_alpha),
            "participation_rate": float(participation_rate),
            "boost_factor": 10.0,
            "calibration_role": "clean_calibration",
        }
        for seed in seeds
    ]


def collect_clean(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
        dataset_name=cfg.dataset.name,
    )
    parameter_roles = get_parameter_roles(model)
    bootstrap_manifest = Path(args.bootstrap_manifest).resolve()
    if CalibrationManifest.load(bootstrap_manifest).payload["schema_version"] != SCHEMA_VERSION_V3:
        bootstrap_manifest = build_semantic_collection_bootstrap(
            base_manifest=bootstrap_manifest,
            config_path=args.config,
            output_path=output / "rtc_v3_semantic_collection_bootstrap.json",
        )
    specs = build_clean_specs(
        seeds=args.calibration_seeds,
        bootstrap_manifest=bootstrap_manifest,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        parameter_roles=parameter_roles,
    )
    runner_args = SimpleNamespace(**vars(args))
    runner_args.smoke = False
    runner_args.pairing_mode = "strict"
    runner_args.sampling_protocol = "principal_uniform"
    runner_args.rerun = bool(args.rerun)
    runner_args.max_spec_retries = int(args.max_spec_retries)
    planned = attach_trial_plans(specs, runner_args, output)
    planned = _deduplicate(planned)
    write_experiment_manifest(planned, output)
    pd.DataFrame([
        {
            **{key: value for key, value in spec.items() if key != "custom_params"},
            "custom_params": json.dumps(spec["custom_params"], sort_keys=True),
        }
        for spec in planned
    ]).to_csv(output / "calibration_matrix.csv", index=False)
    if not args.collect:
        return planned
    rounds_dir = output / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    _run_specs(planned, runner_args, output, rounds_dir)
    return planned


def _load_observations(
    output: Path,
    specs: Sequence[Mapping[str, Any]],
    expected_rounds: int,
    num_classes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    round_frames: list[pd.DataFrame] = []
    client_frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    required_round = {
        "round", "fit_rtc_v3_clip_norm",
        "fit_rtc_v3_full_anchor_exposure",
        "fit_rtc_v3_full_residual_exposure",
        "fit_rtc_v3_full_total_exposure",
    }
    required_client = {
        "round", "cid", "principal_id", "nominal_mass",
        "rtc_v3_residual_norm", "rtc_v3_cone_full",
    }
    sketch_columns = [f"rtc_v3_sketch_full_{index:03d}" for index in range(256)]
    semantic_row_columns = [
        f"rtc_v3_semantic_row_norm_{index:03d}" for index in range(num_classes)
    ]
    semantic_pair_columns = [
        f"rtc_v3_semantic_pair_raw_{index:03d}"
        for index in range(num_classes * (num_classes - 1))
    ]
    required_client.update(
        [
            *sketch_columns,
            *semantic_row_columns,
            *semantic_pair_columns,
            "rtc_v3_semantic_head_norm",
        ]
    )
    for spec in specs:
        identifier = run_id(dict(spec))
        round_path = output / "rounds" / f"{identifier}.csv"
        client_path = output / "raw" / f"{identifier}_clients.csv"
        config_path = output / "raw" / f"{identifier}_config.json"
        data_path = output / "raw" / f"{identifier}_data_manifest.json"
        for path in (round_path, client_path, config_path, data_path):
            if not path.is_file():
                raise FileNotFoundError(f"Calibration source is incomplete: {path}")
        rounds = pd.read_csv(round_path)
        clients = pd.read_csv(client_path)
        missing_round = required_round.difference(rounds.columns)
        missing_client = required_client.difference(clients.columns)
        if missing_round or missing_client:
            raise ValueError(
                f"Calibration diagnostics missing: rounds={sorted(missing_round)} "
                f"clients={sorted(missing_client)}"
            )
        rounds = rounds[pd.to_numeric(rounds["round"], errors="coerce").between(1, expected_rounds)]
        if set(rounds["round"].astype(int)) != set(range(1, expected_rounds + 1)):
            raise ValueError(f"Calibration run {identifier} does not contain all rounds")
        clients = clients[pd.to_numeric(clients["round"], errors="coerce").between(1, expected_rounds)]
        expected_profiles = clients["round"].map(
            lambda value: _profile_for_round(int(value), _profiles(expected_rounds))
        )
        if "profile" in clients:
            observed_profiles = clients["profile"].astype(str)
            if not observed_profiles.equals(expected_profiles.astype(str)):
                raise ValueError(
                    f"Calibration source {identifier} has inconsistent phase labels"
                )
        else:
            # Client metrics intentionally stay narrow at runtime.  Phase is a
            # deterministic function of server round and is materialized only
            # in the calibration sidecar.
            clients = clients.copy()
            clients["profile"] = expected_profiles.to_numpy()
        rounds["calibration_seed"] = int(spec["seed"])
        clients["calibration_seed"] = int(spec["seed"])
        round_frames.append(rounds)
        client_frames.append(clients)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
        if config["security"]["attack"]["enabled"]:
            raise ValueError(f"Calibration source {identifier} is not clean")
        federation = config.get("federation", {})
        if (
            str(federation.get("pairing_mode")) != "strict"
            or str(federation.get("sampling_protocol")) != "principal_uniform"
            or not bool(federation.get("deterministic_client_training", False))
        ):
            raise ValueError(
                f"Calibration source {identifier} is not strict deterministic"
            )
        calibration_hashes = set(
            rounds["fit_rtc_v3_calibration_hash"].dropna().astype(str)
        ) if "fit_rtc_v3_calibration_hash" in rounds else set()
        if len(calibration_hashes) != 1:
            raise ValueError(
                f"Calibration source {identifier} has inconsistent runtime manifest hashes"
            )
        collection_manifest_hash = next(iter(calibration_hashes))
        collection_archive = (
            output
            / f"rtc_v3_semantic_collection_bootstrap_{collection_manifest_hash[:12]}.json"
        )
        if not collection_archive.is_file():
            raise FileNotFoundError(
                "immutable clean-collection manifest archive missing: "
                f"{collection_archive}"
            )
        archived_payload = json.loads(collection_archive.read_text(encoding="utf-8"))
        if str(archived_payload.get("content_hash")) != collection_manifest_hash:
            raise ValueError("clean-collection manifest archive hash contract mismatch")
        observation_contract = _collection_observation_contract(archived_payload)
        observation_contract_hash = content_hash(observation_contract)
        sketch_path = output / "raw" / f"{identifier}_rtc_v3_sketches.parquet"
        sketch_frame = clients[
            [
                "calibration_seed", "round", "profile", "cid", "principal_id", "nominal_mass",
                "rtc_v3_residual_norm", "rtc_v3_semantic_head_norm",
                *sketch_columns, *semantic_row_columns, *semantic_pair_columns,
            ]
        ].copy()
        float32_columns = [
            "rtc_v3_residual_norm",
            "rtc_v3_semantic_head_norm",
            *sketch_columns,
            *semantic_row_columns,
            *semantic_pair_columns,
        ]
        sketch_frame[float32_columns] = sketch_frame[float32_columns].astype(
            np.float32
        )
        sketch_frame.to_parquet(sketch_path, index=False, compression="zstd")
        sources.append({
            "run_id": identifier,
            "seed": int(spec["seed"]),
            "trial_plan_hash": str(spec["trial_plan_hash"]),
            "round_sha256": _sha256_file(round_path),
            "client_sha256": _sha256_file(client_path),
            "config_sha256": _sha256_file(config_path),
            "data_manifest_sha256": str(data["sha256"]),
            "sketch_sidecar": str(sketch_path),
            "sketch_sidecar_sha256": _sha256_file(sketch_path),
            "sketch_dimension": 256,
            "sketch_algorithm_version": "splitmix64_v1",
            "semantic_row_count": num_classes,
            "semantic_ordered_pair_count": num_classes * (num_classes - 1),
            "collection_manifest_hash": collection_manifest_hash,
            "collection_manifest_archive": str(collection_archive),
            "collection_manifest_archive_sha256": _sha256_file(collection_archive),
            "collection_observation_contract_hash": observation_contract_hash,
            "pairing_mode": "strict",
            "sampling_protocol": "principal_uniform",
            "deterministic_client_training": True,
        })
    observation_hashes = {
        str(source["collection_observation_contract_hash"]) for source in sources
    }
    if len(observation_hashes) != 1:
        raise ValueError(
            "calibration sources do not share one clean observation contract: "
            f"{sorted(observation_hashes)}"
        )
    return (
        pd.concat(round_frames, ignore_index=True),
        pd.concat(client_frames, ignore_index=True),
        sources,
    )


def _quantile(values: Sequence[float], q: float, *, floor: float = 1e-12) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("Cannot calibrate an empty statistic")
    return float(max(floor, np.quantile(array, q)))


def _apply_parent_budget_floor(
    candidate: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Prevent inherited parent budgets from being calibrated twice.

    The clean collection trajectory is already constrained by the promoted V2
    parent ledgers. Re-estimating those parents from their effective exposure
    would recursively shrink the feasible set. New semantic/cone children can
    use raw clean observations, but server/principal parents remain at least as
    permissive as their validated V2 values.
    """

    result = json.loads(json.dumps(candidate))
    applied: list[str] = []

    def visit(dst: dict[str, Any], src: Mapping[str, Any], prefix: str) -> None:
        for key, value in src.items():
            name = str(key)
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(value, Mapping):
                child = dst.setdefault(name, {})
                if not isinstance(child, dict):
                    raise ValueError(f"parent budget shape mismatch at {path}")
                visit(child, value, path)
            else:
                floor = float(value)
                current = float(dst.get(name, floor))
                if current < floor:
                    dst[name] = floor
                    applied.append(path)

    visit(result, parent, "")
    return result, applied


def _rolling_sums(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby("calibration_seed", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=1).sum()
    )


def calibrate_semantic_temporal_exposure(
    clients: pd.DataFrame,
    *,
    profile_names: Sequence[str],
    num_classes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-fit clean classifier-row evidence and replay clean exposure."""

    pairs = ordered_class_pairs(num_classes)
    row_columns = [
        f"rtc_v3_semantic_row_norm_{index:03d}" for index in range(num_classes)
    ]
    raw_pair_columns = [
        f"rtc_v3_semantic_pair_raw_{index:03d}" for index in range(len(pairs))
    ]
    frame = clients.sort_values(
        ["calibration_seed", "round", "principal_id", "cid"],
        kind="mergesort",
    ).reset_index(drop=True)
    rows = frame[row_columns].to_numpy(dtype=np.float64)
    exported_pairs = frame[raw_pair_columns].to_numpy(dtype=np.float64)
    sources = np.asarray([source for source, _ in pairs], dtype=np.int64)
    targets = np.asarray([target for _, target in pairs], dtype=np.int64)
    recomputed_raw = rows[:, targets] - rows[:, sources]
    if not np.allclose(exported_pairs, recomputed_raw, rtol=1e-5, atol=1e-7):
        raise ValueError("semantic sidecar ordered-pair contract is inconsistent")
    seeds = tuple(sorted(int(value) for value in frame["calibration_seed"].unique()))
    if len(seeds) < 2:
        raise ValueError("semantic calibration requires at least two clean seeds")

    final_phases: dict[str, Any] = {}
    for profile in profile_names:
        mask = frame["profile"].astype(str).to_numpy() == str(profile)
        profile_rows = rows[mask]
        row_scales = np.maximum(np.quantile(profile_rows, 0.50, axis=0), 1e-8)
        normalized = profile_rows / row_scales.reshape(1, -1)
        pair_values = normalized[:, targets] - normalized[:, sources]
        centers = np.quantile(pair_values, 0.50, axis=0)
        mad = np.quantile(np.abs(pair_values - centers), 0.50, axis=0) * 1.4826
        iqr = (
            np.quantile(pair_values, 0.75, axis=0)
            - np.quantile(pair_values, 0.25, axis=0)
        ) / 1.349
        pair_scales = np.maximum(np.maximum(mad, iqr), 1e-6)
        head_scale = _quantile(
            frame.loc[mask, "rtc_v3_semantic_head_norm"], 0.99
        )
        final_phases[str(profile)] = {
            "row_scales": row_scales.tolist(),
            "pair_centers": centers.tolist(),
            "pair_scales": pair_scales.tolist(),
            "head_exposure_scale": head_scale,
        }

    oof_z = np.zeros((len(frame), len(pairs)), dtype=np.float64)
    for evaluation_seed in seeds:
        train_mask = frame["calibration_seed"].astype(int).to_numpy() != evaluation_seed
        eval_mask = ~train_mask
        for profile in profile_names:
            profile_mask = frame["profile"].astype(str).to_numpy() == str(profile)
            fit = train_mask & profile_mask
            evaluate = eval_mask & profile_mask
            fit_rows = rows[fit]
            row_scales = np.maximum(np.quantile(fit_rows, 0.50, axis=0), 1e-8)
            fit_pairs = (
                fit_rows[:, targets] / row_scales[targets]
                - fit_rows[:, sources] / row_scales[sources]
            )
            centers = np.quantile(fit_pairs, 0.50, axis=0)
            mad = np.quantile(np.abs(fit_pairs - centers), 0.50, axis=0) * 1.4826
            iqr = (
                np.quantile(fit_pairs, 0.75, axis=0)
                - np.quantile(fit_pairs, 0.25, axis=0)
            ) / 1.349
            scales = np.maximum(np.maximum(mad, iqr), 1e-6)
            eval_rows = rows[evaluate]
            eval_pairs = (
                eval_rows[:, targets] / row_scales[targets]
                - eval_rows[:, sources] / row_scales[sources]
            )
            oof_z[evaluate] = np.maximum(0.0, (eval_pairs - centers) / scales)

    decay_rate = math.log(2.0) / 8.0
    kappa = _quantile(oof_z.reshape(-1), 0.95)
    statistics = np.zeros_like(oof_z)
    states: dict[tuple[int, str, int], tuple[float, int]] = {}
    for row_index, row in frame.iterrows():
        seed = int(row["calibration_seed"])
        principal = str(row["principal_id"])
        round_number = int(row["round"])
        for pair_index, evidence in enumerate(oof_z[row_index]):
            key = (seed, principal, pair_index)
            previous, previous_round = states.get(key, (0.0, round_number))
            elapsed = max(0, round_number - previous_round)
            statistic = max(
                0.0,
                previous * math.exp(-decay_rate * elapsed)
                + float(evidence)
                - kappa,
            )
            states[key] = (statistic, round_number)
            statistics[row_index, pair_index] = statistic
    maximum_statistics = np.max(statistics, axis=1)
    threshold = MARGIN * _quantile(maximum_statistics, 0.99)
    eta = math.log(2.0) / max(threshold, 1.0)
    risks = 1.0 - np.exp(
        -eta * np.maximum(0.0, maximum_statistics - threshold)
    )
    head_scales = np.asarray(
        [
            final_phases[str(profile)]["head_exposure_scale"]
            for profile in frame["profile"]
        ],
        dtype=np.float64,
    )
    # Soft risk/q remain continuous, while the hard exposure ledger ignores
    # sub-watch clean jitter. Without this calibrated deadband, a nearly-zero
    # benign risk can consume the last floating-point remainder of a long
    # window and force an otherwise benign round to zero mass.
    hard_exposure_risk_floor = 0.10
    coefficients = np.maximum(0.0, risks - hard_exposure_risk_floor) * (
        frame["rtc_v3_semantic_head_norm"].to_numpy(dtype=np.float64)
        / head_scales
    )
    replay = frame[
        ["calibration_seed", "round", "principal_id", "nominal_mass"]
    ].copy()
    replay["semantic_exposure"] = (
        coefficients * replay["nominal_mass"].to_numpy(dtype=np.float64)
    )
    round_replay = replay.groupby(
        ["calibration_seed", "round"], as_index=False
    )["semantic_exposure"].sum()
    global_budgets = {
        str(window): MARGIN
        * _quantile(_rolling_sums(round_replay, "semantic_exposure", window), 1.0)
        for window in SERVER_WINDOWS
    }
    principal_values: dict[str, float] = {}
    principal_replay = replay.sort_values(
        ["calibration_seed", "principal_id", "round"], kind="mergesort"
    )
    for window in PRINCIPAL_WINDOWS:
        rolling = principal_replay.groupby(
            ["calibration_seed", "principal_id"], sort=False
        )["semantic_exposure"].transform(
            lambda values: values.rolling(window, min_periods=1).sum()
        )
        principal_values[str(window)] = MARGIN * _quantile(rolling, 1.0)
    top_pair_indices = np.argmax(oof_z, axis=1)
    pair_replay = replay.copy()
    pair_replay["pair_index"] = top_pair_indices
    pair_rounds = pair_replay.groupby(
        ["calibration_seed", "round", "pair_index"], as_index=False
    )["semantic_exposure"].sum()
    pair_budgets: dict[str, float] = {}
    for window in SERVER_WINDOWS:
        rolling = pair_rounds.sort_values(
            ["calibration_seed", "pair_index", "round"], kind="mergesort"
        ).groupby(["calibration_seed", "pair_index"], sort=False)[
            "semantic_exposure"
        ].transform(lambda values: values.rolling(window, min_periods=1).sum())
        pair_budgets[str(window)] = MARGIN * _quantile(rolling, 1.0)

    semantic = {
        "enabled": True,
        "num_classes": num_classes,
        "phases": final_phases,
        "temporal": {
            "decay_rate": decay_rate,
            "kappa": kappa,
            "threshold": threshold,
            "eta": eta,
            "q_min": 0.30,
            "recovery_threshold": _quantile(oof_z.reshape(-1), 0.50),
            "recovery_observations": 3,
            "recovery_factor": 0.50,
            "min_history_observations": 3,
            "synchronization": {
                "enabled": True,
                "min_distinct_principals": 2,
                "evidence_threshold": _quantile(np.max(oof_z, axis=1), 0.99),
                "similarity_threshold": 0.90,
                "multiplier": 1.25,
            },
        },
        "risk_lambda": 0.25,
        "state_thresholds": {"watch": 0.10, "restricted": 0.50, "quarantined": 0.80},
        "hard_exposure_risk_floor": hard_exposure_risk_floor,
        "top_k_pairs": min(4, len(pairs) - 1),
        "global_windows": list(SERVER_WINDOWS),
        "global_budgets": global_budgets,
        "principal_windows": list(PRINCIPAL_WINDOWS),
        "principal_budgets": principal_values,
        "pair_windows": list(SERVER_WINDOWS),
        "pair_budgets": pair_budgets,
        "performance_budget": {
            "semantic_mean_seconds": 0.15,
            "semantic_p95_seconds": 0.25,
            "defense_mean_increase_fraction": 0.05,
            "defense_p95_increase_fraction": 0.10,
        },
    }
    diagnostics = {
        "cross_fit_seeds": list(seeds),
        "ordered_pair_count": len(pairs),
        "oof_z_q99": _quantile(oof_z.reshape(-1), 0.99),
        "clean_statistic_q99": _quantile(maximum_statistics, 0.99),
        "clean_risk_max": float(np.max(risks)),
        "clean_positive_risk_fraction": float(np.mean(risks > 0.0)),
        "global_budgets": global_budgets,
        "principal_budgets": principal_values,
        "pair_budgets": pair_budgets,
        "budget_envelopes": {
            **{
                f"global/{window}": {
                    "observed_max": float(budget) / MARGIN,
                    "budget": float(budget),
                }
                for window, budget in global_budgets.items()
            },
            **{
                f"principal/{window}": {
                    "observed_max": float(budget) / MARGIN,
                    "budget": float(budget),
                }
                for window, budget in principal_values.items()
            },
            **{
                f"pair/{window}": {
                    "observed_max": float(budget) / MARGIN,
                    "budget": float(budget),
                }
                for window, budget in pair_budgets.items()
            },
        },
    }
    return semantic, diagnostics


def calibrate_manifest(
    *,
    args: argparse.Namespace,
    specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(args.output).resolve()
    cfg = load_config(args.config)
    parent_manifest = CalibrationManifest.load(args.bootstrap_manifest)
    rounds, clients, sources = _load_observations(
        output, specs, args.rounds, cfg.dataset.num_classes
    )
    phases = _profiles(args.rounds)
    profile_names = [str(value["profile"]) for value in phases]
    rounds["profile"] = rounds["round"].astype(int).map(
        lambda value: _profile_for_round(value, phases)
    )
    clients["profile"] = clients["round"].astype(int).map(
        lambda value: _profile_for_round(value, phases)
    )

    stable_cones, cone_assignments, cone_cluster_diagnostics = (
        calibrate_offline_cones(
            clients,
            profile_names=profile_names,
            dimension=256,
            max_prototypes=16,
            momentum=0.98,
        )
    )
    stable_cones = dict(stable_cones)
    stable_cones["online_updates_enabled"] = False
    semantic, semantic_diagnostics = calibrate_semantic_temporal_exposure(
        clients,
        profile_names=profile_names,
        num_classes=cfg.dataset.num_classes,
    )

    clip_lower: dict[str, float] = {}
    clip_upper: dict[str, float] = {}
    residual_scales: dict[str, dict[str, float]] = {"full": {}}
    for profile in profile_names:
        phase_rounds = rounds[rounds["profile"] == profile]
        phase_clients = clients[clients["profile"] == profile]
        clips = pd.to_numeric(phase_rounds["fit_rtc_v3_clip_norm"], errors="coerce")
        residuals = pd.to_numeric(phase_clients["rtc_v3_residual_norm"], errors="coerce")
        clip_lower[profile] = 0.90 * _quantile(clips, 0.01)
        clip_upper[profile] = MARGIN * _quantile(clips, 0.99)
        residual_scales["full"][profile] = _quantile(residuals, 0.50)

    for exposure_type in ("anchor", "residual", "total"):
        raw_col = f"fit_rtc_v3_full_{exposure_type}_exposure"
        rounds[f"normalized_{exposure_type}"] = [
            float(value) / residual_scales["full"][str(profile)]
            for value, profile in zip(
                pd.to_numeric(rounds[raw_col], errors="raise"), rounds["profile"]
            )
        ]

    # Bootstrap budgets may already suppress the recorded aggregate residual.
    # Recalibration must instead replay the unconstrained clean nominal client
    # exposure, otherwise child cone envelopes can exceed their parent simply
    # because the two levels were calibrated from different measures.
    clients["nominal_mass"] = pd.to_numeric(
        clients["nominal_mass"], errors="raise"
    )
    clients["rtc_v3_residual_norm"] = pd.to_numeric(
        clients["rtc_v3_residual_norm"], errors="raise"
    )
    clients["normalized_nominal_residual"] = [
        float(mass) * float(residual) / residual_scales["full"][str(profile)]
        for mass, residual, profile in zip(
            clients["nominal_mass"],
            clients["rtc_v3_residual_norm"],
            clients["profile"],
        )
    ]
    nominal_residual_by_round = clients.groupby(
        ["calibration_seed", "round"], sort=True
    )["normalized_nominal_residual"].sum()
    rounds["normalized_residual"] = [
        float(nominal_residual_by_round.loc[(int(seed), int(round_number))])
        for seed, round_number in zip(rounds["calibration_seed"], rounds["round"])
    ]
    rounds["normalized_total"] = (
        rounds["normalized_anchor"] + rounds["normalized_residual"]
    )

    server_budgets: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "full": {}
    }
    server_diagnostics: dict[str, Any] = {}
    for window in SERVER_WINDOWS:
        server_budgets["full"][str(window)] = {}
        for exposure_type in ("anchor", "residual", "total"):
            column = f"normalized_{exposure_type}"
            rolled = _rolling_sums(
                rounds.sort_values(["calibration_seed", "round"]), column, window
            )
            values = rounds.sort_values(["calibration_seed", "round"]).assign(
                rolled=rolled.to_numpy()
            )
            # A hard budget must admit every independent clean calibration
            # observation.  P99 remains diagnostic metadata, but using it as
            # the hard envelope contradicts the max-coverage validation gate.
            observed_max = float(values["rolled"].max())
            global_envelope = MARGIN * observed_max
            profile_map = {
                profile: max(
                    global_envelope,
                    MARGIN * _quantile(values.loc[values["profile"] == profile, "rolled"], 0.99),
                )
                for profile in profile_names
            }
            server_budgets["full"][str(window)][exposure_type] = profile_map
            server_diagnostics[f"full/{window}/{exposure_type}"] = {
                "observed_max": observed_max,
                "observed_p99": _quantile(values["rolled"], 0.99),
                "budget_min": float(min(profile_map.values())),
            }

    server_budgets, server_floor_paths = _apply_parent_budget_floor(
        server_budgets, parent_manifest.payload.get("server_budgets") or {}
    )
    for key, diagnostic in server_diagnostics.items():
        resolution, window, exposure_type = key.split("/")
        diagnostic["budget_min"] = float(
            min(server_budgets[resolution][window][exposure_type].values())
        )

    # Replay raw clean client exposure under the new offline assignments.  This
    # calibrates real matched cones and overflow rather than the bootstrap bank.
    cone_assignments["normalized_exposure"] = [
        float(mass) * float(residual) / residual_scales["full"][str(profile)]
        for mass, residual, profile in zip(
            cone_assignments["nominal_mass"],
            cone_assignments["rtc_v3_residual_norm"],
            cone_assignments["profile"],
        )
    ]
    per_cone = cone_assignments.groupby(
        ["calibration_seed", "round", "profile", "cone_id"], as_index=False
    )["normalized_exposure"].sum()
    cone_current = per_cone.groupby(
        ["calibration_seed", "round", "profile"], as_index=False
    )["normalized_exposure"].max().rename(
        columns={"normalized_exposure": "normalized_cone"}
    )
    cone_budgets: dict[str, dict[str, dict[str, dict[str, float]]]] = {"full": {}}
    cone_diagnostics: dict[str, Any] = {}
    for window in SERVER_WINDOWS:
        ordered = cone_current.sort_values(["calibration_seed", "round"]).copy()
        ordered["rolled"] = _rolling_sums(ordered, "normalized_cone", window)
        observed_max = float(ordered["rolled"].max())
        envelope = MARGIN * observed_max
        parent = server_budgets["full"][str(window)]["residual"]
        profile_map = {
            profile: min(float(parent[profile]), float(envelope))
            for profile in profile_names
        }
        cone_budgets["full"][str(window)] = {"residual": profile_map}
        cone_diagnostics[f"full/{window}/residual"] = {
            "observed_max": observed_max,
            "observed_p99": _quantile(ordered["rolled"], 0.99),
            "budget_min": float(min(profile_map.values())),
        }

    clients["nominal_mass"] = pd.to_numeric(clients["nominal_mass"], errors="raise")
    clients["rtc_v3_residual_norm"] = pd.to_numeric(
        clients["rtc_v3_residual_norm"], errors="raise"
    )
    clients["raw_principal_exposure"] = (
        clients["nominal_mass"] * clients["rtc_v3_residual_norm"]
    )
    events = clients.groupby(
        ["calibration_seed", "principal_id", "round", "profile"], as_index=False
    ).agg(
        exposure=("raw_principal_exposure", "sum"),
        mass=("nominal_mass", "sum"),
    )
    events["normalized_exposure"] = [
        float(value) / residual_scales["full"][str(profile)]
        for value, profile in zip(events["exposure"], events["profile"])
    ]
    principal_betas: dict[str, dict[str, dict[str, float]]] = {"full": {}}
    principal_diagnostics: dict[str, Any] = {}
    for window in PRINCIPAL_WINDOWS:
        ratios: list[tuple[str, float]] = []
        for (_, _), group in events.groupby(["calibration_seed", "principal_id"]):
            ordered = group.sort_values("round")
            exposure_roll = ordered["normalized_exposure"].rolling(window, min_periods=1).sum()
            mass_roll = ordered["mass"].rolling(window, min_periods=1).sum()
            ratios.extend(zip(ordered["profile"], exposure_roll / np.maximum(mass_roll, 1e-12)))
        ratio_frame = pd.DataFrame(ratios, columns=["profile", "ratio"])
        observed_max = float(ratio_frame["ratio"].max())
        envelope = MARGIN * observed_max
        profile_map = {
            profile: max(
                envelope,
                MARGIN * _quantile(
                    ratio_frame.loc[ratio_frame["profile"] == profile, "ratio"], 0.99
                ),
            )
            for profile in profile_names
        }
        principal_betas["full"][str(window)] = profile_map
        principal_diagnostics[f"full/{window}"] = {
            "observed_max": observed_max,
            "observed_p99": _quantile(ratio_frame["ratio"], 0.99),
            "beta_min": float(min(profile_map.values())),
        }

    principal_betas, principal_floor_paths = _apply_parent_budget_floor(
        principal_betas, parent_manifest.payload.get("principal_betas") or {}
    )
    for key, diagnostic in principal_diagnostics.items():
        resolution, window = key.split("/")
        diagnostic["beta_min"] = float(
            min(principal_betas[resolution][window].values())
        )

    # Calibrate optional one-sided evidence from clean principal trajectories.
    raw_by_round = events.assign(raw=lambda frame: frame["exposure"] / np.maximum(frame["mass"], 1e-12))
    centers = raw_by_round.groupby(["calibration_seed", "round"])["raw"].median()
    z_values = []
    for _, event in raw_by_round.iterrows():
        center = float(centers.loc[(event["calibration_seed"], event["round"])])
        z_values.append(float(event["raw"]) / max(center, 1e-12))
    raw_by_round["z"] = z_values
    kappa = _quantile(raw_by_round["z"], 0.95)
    statistics: list[float] = []
    for (_, _), group in raw_by_round.groupby(["calibration_seed", "principal_id"]):
        statistic = 0.0
        for z in group.sort_values("round")["z"]:
            statistic = max(0.0, statistic + float(z) - kappa)
            statistics.append(statistic)
    threshold = MARGIN * _quantile(statistics, 0.99)
    cumulative = {
        "enabled": True,
        "resolutions": {
            "full": {
                "scale_center": _quantile(raw_by_round["raw"], 0.50),
                "scale_lower": _quantile(raw_by_round["raw"], 0.01),
                "scale_upper": MARGIN * _quantile(raw_by_round["raw"], 0.99),
                "kappa": kappa,
                "threshold": threshold,
                "eta": math.log(2.0) / max(threshold, 1.0),
                "q_min": 0.50,
            }
        },
    }
    direction = {
        "enabled": True,
        "resolutions": {
            "full": {
                "z_threshold": MARGIN * _quantile(raw_by_round["z"], 0.99),
                "history_length": 4,
                "min_matches": 3,
                "server_window": 8,
                "principal_window": 4,
                "server_usage_threshold": 0.80,
                "principal_usage_threshold": 0.80,
                "q_value": 0.70,
            }
        },
    }

    model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
        dataset_name=cfg.dataset.name,
    )
    params = get_parameters(model)
    roles = get_parameter_roles(model)
    names = get_parameter_names(model)
    head_layout = ClassifierHeadLayout.build(params, roles)
    payload = build_manifest(
        params=params,
        roles=roles,
        profile=profile_names[0],
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        resolutions=("full",),
        residual_scales=residual_scales,
        server_windows=SERVER_WINDOWS,
        server_budgets=server_budgets,
        principal_windows=PRINCIPAL_WINDOWS,
        principal_betas=principal_betas,
        training_phases=phases,
        schema_version=SCHEMA_VERSION_V3,
        stable_cones=stable_cones,
        cone_budgets=cone_budgets,
        cumulative=cumulative,
        direction_persistence=direction,
        classifier_head_contract=head_layout.contract(params, names),
        semantic_temporal_exposure=semantic,
        metadata={
            "artifact_kind": "rtc_v3_semantic_temporal_exposure_candidate",
            # Calibration alone does not authorize held-out evaluation.  The
            # candidate must first pass development-only strict gates and be
            # frozen unchanged by rtc_v3_semantic_development.py.
            "formal_evaluation_ready": False,
            "promotion_ready": False,
            "calibration_seeds": [int(value) for value in args.calibration_seeds],
            "reserved_evaluation_seeds": [int(value) for value in EVALUATION_SEEDS],
            "development_seeds": [int(value) for value in DEVELOPMENT_SEEDS],
            "seed_sets_disjoint": not bool(set(args.calibration_seeds) & set(EVALUATION_SEEDS)),
            "margin_multiplier": MARGIN,
            "quantile": 0.99,
            "partition": args.partition,
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "participation_rate": float(args.participation_rate),
            "num_clients": int(args.num_clients),
            "rounds": int(args.rounds),
            "batch_size": int(args.batch_size),
            "sketch_algorithm_version": "splitmix64_v1",
            "source_runs": sources,
            "semantic_calibration": semantic_diagnostics,
            "parent_budget_floor": {
                "manifest_hash": parent_manifest.hash,
                "server_paths_raised": server_floor_paths,
                "principal_paths_raised": principal_floor_paths,
                "policy": "inherited_parent_budgets_never_tighten",
            },
            "warning": "Promotion still requires held-out multi-seed clean/attack gates.",
        },
    )
    manifest = CalibrationManifest.load(payload)
    manifest.validate_runtime(
        model_hash=payload["model_metadata_hash"],
        server_optimizer="fedavg",
        validated_mass_policy="unit_principal",
        principal_mapping_version="v1",
    )
    validation = {
        "passed": True,
        "content_hash": manifest.hash,
        "calibration_seeds": list(args.calibration_seeds),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "seed_sets_disjoint": not bool(set(args.calibration_seeds) & set(EVALUATION_SEEDS)),
        "all_seed_roles_disjoint": not bool(
            set(args.calibration_seeds) & set(DEVELOPMENT_SEEDS)
            or set(args.calibration_seeds) & set(EVALUATION_SEEDS)
            or set(DEVELOPMENT_SEEDS) & set(EVALUATION_SEEDS)
        ),
        "q_min_less_than_one": float(cumulative["resolutions"]["full"]["q_min"]) < 1.0,
        "q_value_less_than_one": float(direction["resolutions"]["full"]["q_value"]) < 1.0,
        "semantic_q_min_less_than_one": float(
            semantic["temporal"]["q_min"]
        ) < 1.0,
        "stable_cones_frozen": not bool(
            stable_cones.get("online_updates_enabled", True)
        ),
        "clip_profiles": {
            profile: {"lower": clip_lower[profile], "upper": clip_upper[profile]}
            for profile in profile_names
        },
        "server_budget_envelopes": server_diagnostics,
        "principal_budget_envelopes": principal_diagnostics,
        "cone_budget_envelopes": cone_diagnostics,
        "cone_clustering": cone_cluster_diagnostics,
        "semantic_budget_envelopes": semantic_diagnostics["budget_envelopes"],
        "cone_sketch_dimension": 256,
        "source_runs": sources,
    }
    validation["passed"] = bool(
        validation["seed_sets_disjoint"]
        and validation["all_seed_roles_disjoint"]
        and validation["q_min_less_than_one"]
        and validation["q_value_less_than_one"]
        and validation["semantic_q_min_less_than_one"]
        and validation["stable_cones_frozen"]
        and all(value["lower"] <= value["upper"] for value in validation["clip_profiles"].values())
        and all(value["observed_max"] <= value["budget_min"] + 1e-12 for value in server_diagnostics.values())
        and all(value["observed_max"] <= value["beta_min"] + 1e-12 for value in principal_diagnostics.values())
        and all(value["observed_max"] <= value["budget_min"] + 1e-12 for value in cone_diagnostics.values())
        and all(value["coverage"] >= 0.95 for value in cone_cluster_diagnostics.values())
        and all(
            value["observed_max"] <= value["budget"] + 1e-12
            for value in semantic_diagnostics["budget_envelopes"].values()
        )
    )
    return payload, validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--bootstrap-manifest", default="config/rtc_v3_manifest.json")
    parser.add_argument("--output", default="logs/rtc_v3_formal_calibration_iid")
    parser.add_argument(
        "--manifest-output",
        default="config/rtc_v3_manifest_formal_iid_semantic_candidate.json",
    )
    parser.add_argument(
        "--validation-output",
        default="config/rtc_v3_manifest_formal_iid_semantic_candidate.validation.json",
    )
    parser.add_argument("--calibration-seeds", default="40,41")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--partition", choices=("iid", "dirichlet", "non_iid"), default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-client-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--ray-client-num-cpus", type=float, default=1.0)
    parser.add_argument("--ray-client-num-gpus", type=float, default=0.25)
    parser.add_argument("--ray-object-store-memory-mb", type=int, default=3072)
    parser.add_argument("--ray-min-available-memory-mb", type=int, default=10240)
    parser.add_argument("--ray-memory-wait-seconds", type=float, default=120.0)
    parser.add_argument("--max-spec-retries", type=int, default=1)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args(argv)
    args.calibration_seeds = tuple(
        int(value.strip()) for value in args.calibration_seeds.split(",") if value.strip()
    )
    if len(args.calibration_seeds) < 2:
        parser.error("formal calibration requires at least two independent clean seeds")
    if set(args.calibration_seeds) & set(EVALUATION_SEEDS):
        parser.error("calibration seeds must be disjoint from reserved evaluation seeds")
    if set(args.calibration_seeds) & set(DEVELOPMENT_SEEDS):
        parser.error("calibration seeds must be disjoint from development seeds")
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    specs = collect_clean(args)
    if not args.generate:
        print(f"Calibration collection prepared: {Path(args.output).resolve()}")
        return Path(args.output).resolve()
    payload, validation = calibrate_manifest(args=args, specs=specs)
    manifest_path = Path(args.manifest_output).resolve()
    validation_path = Path(args.validation_output).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError(f"Formal manifest validation failed: {validation_path}")
    print(f"RTC-v3 formal manifest written: {manifest_path}")
    print(f"content_hash: {payload['content_hash']}")
    print(f"validation: {validation_path}")
    return manifest_path


if __name__ == "__main__":
    main()
