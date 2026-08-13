"""Audit RTC-v3 cone V2 held-out runs and emit promotion or rejection evidence."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import (
    CalibrationManifest,
    content_hash,
    model_metadata_hash,
)
from experiments.trial_plan import TrialPlanV1
from experiments.periodic_attack import run_id, validate_sampling_manifests
from models.model_factory import get_model, get_parameters


REQUIRED_GATES = {
    "matrix_complete",
    "rounds_complete",
    "critical_metrics_finite",
    "rtc_v3_manifest_hash_match",
    "rtc_v3_constraint_violation",
    "rtc_v3_subprobability_mass",
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
RAW_PAIRING_GATES = {
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
RAW_EXECUTION_GATES = {
    "matrix_complete",
    "rounds_complete",
    "critical_metrics_finite",
    "rtc_v3_manifest_hash_match",
    "rtc_v3_constraint_violation",
    "rtc_v3_subprobability_mass",
}
PHASES = ("warmup", "steady", "late")
ATTACK_JOIN_KEYS = (
    "trial_plan_hash",
    "seed",
    "attack_group",
    "attack",
    "period",
    "on_rounds",
    "off_rounds",
    "malicious_fraction",
    "attack_start_round",
    "attack_end_round",
    "partition",
    "dirichlet_alpha",
    "participation_rate",
    "label_flip_poison_fraction",
    "label_flip_source_label",
    "label_flip_target_label",
    "pairing_mode",
    "sampling_protocol",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) != 0.0)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _execution_gates(root: Path) -> tuple[dict[str, bool], list[str]]:
    path = root / "execution_validation.csv"
    if not path.is_file():
        return {}, [f"missing execution validation: {path}"]
    frame = pd.read_csv(path)
    failures: list[str] = []
    required_columns = {"gate", "passed"}
    if not required_columns.issubset(frame.columns):
        return {}, [
            f"execution validation missing columns: {sorted(required_columns - set(frame.columns))}"
        ]
    names = frame["gate"].astype(str)
    duplicates = sorted(names[names.duplicated(keep=False)].unique())
    if duplicates:
        failures.append(f"duplicate execution gate rows: {duplicates}")
    values: dict[str, bool] = {}
    for _, row in frame.iterrows():
        gate = str(row["gate"])
        raw = str(row["passed"]).strip().lower()
        if raw not in {"true", "false", "1", "0"}:
            failures.append(
                f"invalid execution gate boolean for {gate}: {row['passed']!r}"
            )
            continue
        values[gate] = raw in {"true", "1"}
    failures.extend(
        gate for gate in sorted(REQUIRED_GATES) if not values.get(gate, False)
    )
    return values, failures


def _raw_pairing_audit(
    root: Path,
    saved_gates: Mapping[str, bool],
) -> tuple[dict[str, bool], list[str]]:
    """Recompute strict pairing gates from immutable plans and raw manifests."""
    failures: list[str] = []
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        return {}, [f"missing experiment manifest: {manifest_path}"]
    experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_hash = str(experiment.get("sha256", ""))
    unhashed = dict(experiment)
    unhashed.pop("sha256", None)
    expected_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_hash != expected_hash:
        failures.append(
            f"experiment manifest hash mismatch: observed={observed_hash}, expected={expected_hash}"
        )
    specs = experiment.get("specs", [])
    expected_ids = {run_id(spec) for spec in specs}
    recomputed_rows = validate_sampling_manifests(
        root / "rounds", root / "raw", expected_ids=expected_ids
    )
    recomputed = {
        str(row["gate"]): bool(row["passed"])
        for row in recomputed_rows
        if str(row["gate"]) in RAW_PAIRING_GATES
    }
    for gate in sorted(RAW_PAIRING_GATES):
        if not recomputed.get(gate, False):
            failures.append(f"raw recomputation failed: {gate}")
        if saved_gates.get(gate) != recomputed.get(gate):
            failures.append(
                f"saved/raw strict gate mismatch for {gate}: "
                f"saved={saved_gates.get(gate)}, recomputed={recomputed.get(gate)}"
            )
    return recomputed, failures


def _raw_execution_audit(
    root: Path,
    saved_gates: Mapping[str, bool],
) -> tuple[dict[str, bool], list[str]]:
    """Recompute coverage, numerical, manifest, and RTC hard-constraint gates."""
    def numeric(data: pd.DataFrame, column: str) -> pd.Series:
        if column not in data:
            return pd.Series(np.nan, index=data.index, dtype=float)
        return pd.to_numeric(data[column], errors="coerce")

    failures: list[str] = []
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        empty = {gate: False for gate in RAW_EXECUTION_GATES}
        return empty, [f"missing experiment manifest: {manifest_path}"]
    try:
        experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        empty = {gate: False for gate in RAW_EXECUTION_GATES}
        return empty, [f"invalid experiment manifest {manifest_path}: {exc}"]
    specs = experiment.get("specs", [])
    expected = {run_id(spec): spec for spec in specs}
    paths = {
        path.stem: path
        for path in (root / "rounds").glob("*.csv")
        if path.stem in expected
    }
    recomputed = {
        "matrix_complete": set(paths) == set(expected),
        "rounds_complete": True,
        "critical_metrics_finite": True,
        "rtc_v3_manifest_hash_match": True,
        "rtc_v3_constraint_violation": True,
        "rtc_v3_subprobability_mass": True,
    }
    for identifier, spec in expected.items():
        path = paths.get(identifier)
        if path is None:
            recomputed["rounds_complete"] = False
            recomputed["critical_metrics_finite"] = False
            continue
        frame = pd.read_csv(path)
        numeric_round = numeric(frame, "round")
        observed = set(numeric_round.dropna().astype(int))
        if observed not in (set(range(1, 61)), set(range(0, 61))):
            recomputed["rounds_complete"] = False
        status_path = root / "status" / f"{identifier}.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file() else {}
        )
        last_round_value = status.get("last_round")
        exit_code_value = status.get("exit_code")
        try:
            status_last_round = (
                int(last_round_value) if last_round_value is not None else 0
            )
            status_exit_code = (
                int(exit_code_value) if exit_code_value is not None else -1
            )
        except (TypeError, ValueError):
            status_last_round = 0
            status_exit_code = -1
        if (
            status.get("state") not in {"completed", "completed_cached"}
            or status_last_round != 60
            or status_exit_code != 0
        ):
            recomputed["rounds_complete"] = False
        fit = frame[numeric_round.between(1, 60)].copy()
        accuracy = numeric(fit, "server_accuracy")
        if len(accuracy) != 60 or accuracy.isna().any() or not np.isfinite(accuracy).all():
            recomputed["critical_metrics_finite"] = False
        attack = str(spec.get("attack"))
        if attack == "label_flip_targeted":
            asr = numeric(fit, "server_asr")
            if len(asr) != 60 or asr.isna().any() or not np.isfinite(asr).all():
                recomputed["critical_metrics_finite"] = False
        elif attack == "label_flip_all_reverse":
            try:
                if "server_asr" in fit and pd.to_numeric(
                    fit["server_asr"], errors="coerce"
                ).notna().all():
                    reverse_asr = pd.to_numeric(fit["server_asr"], errors="coerce")
                else:
                    reverse_asr = fit["server_confusion_matrix_json"].map(
                        _reverse_mapping_asr
                    )
                if len(reverse_asr) != 60 or not np.isfinite(reverse_asr).all():
                    recomputed["critical_metrics_finite"] = False
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                recomputed["critical_metrics_finite"] = False
        if str(spec.get("defense")) != "rtc_v3":
            continue
        config_path = root / "raw" / f"{identifier}_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            calibration_path = config["security"]["defense"]["custom_params"]["calibration_path"]
            expected_hash = CalibrationManifest.load(calibration_path).hash
            observed_hashes = set(fit["fit_rtc_v3_calibration_hash"].dropna().astype(str))
            if observed_hashes != {expected_hash}:
                recomputed["rtc_v3_manifest_hash_match"] = False
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
            recomputed["rtc_v3_manifest_hash_match"] = False
        violation = numeric(fit, "fit_rtc_v3_max_constraint_violation")
        if len(violation) != 60 or violation.isna().any() or (violation > 1e-8).any():
            recomputed["rtc_v3_constraint_violation"] = False
        mass = numeric(fit, "fit_rtc_v3_weight_sum")
        if (
            len(mass) != 60
            or mass.isna().any()
            or (mass < -1e-12).any()
            or (mass > 1.0 + 1e-8).any()
        ):
            recomputed["rtc_v3_subprobability_mass"] = False
    for gate in sorted(RAW_EXECUTION_GATES):
        if not recomputed.get(gate, False):
            failures.append(f"raw recomputation failed: {gate}")
        if saved_gates.get(gate) != recomputed.get(gate):
            failures.append(
                f"saved/raw execution gate mismatch for {gate}: "
                f"saved={saved_gates.get(gate)}, recomputed={recomputed.get(gate)}"
            )
    return recomputed, failures


def _calibration_runtime_audit(
    root: Path,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expected_seeds = {40, 41}
    observed_seeds: set[int] = set()
    contracts: list[dict[str, Any]] = []
    model_contracts: set[tuple[str, int, bool, str]] = set()
    status_by_run: dict[str, Mapping[str, Any]] = {}
    for path in sorted((root / "status").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        status_by_run[str(value.get("run_id"))] = value
    sources = {
        str(value["run_id"]): value for value in validation.get("source_runs", [])
    }
    manifest_sources = {
        str(value["run_id"]): value
        for value in manifest.get("metadata", {}).get("source_runs", [])
    }
    for config_path in sorted((root / "raw").glob("*_config.json")):
        run_id = config_path.name.removesuffix("_config.json")
        if "calibration_clean" not in run_id:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(config["project"]["seed"])
        observed_seeds.add(seed)
        federation = config["federation"]
        client = config["client"]
        dataset = config["dataset"]
        ray = config["ray"]
        attack = config["security"]["attack"]
        model_config = config["model"]
        model_contracts.add((
            str(model_config["architecture"]),
            int(model_config["num_classes"]),
            bool(model_config["pretrained"]),
            str(dataset["name"]),
        ))
        contract = {
            "run_id": run_id,
            "seed": seed,
            "num_rounds": int(federation["num_rounds"]),
            "num_clients": int(federation["num_clients"]),
            "clients_per_round": int(federation["clients_per_round"]),
            "max_client_samples": int(federation["max_client_samples"]),
            "max_test_samples": int(config["evaluation"]["max_test_samples"]),
            "batch_size": int(client["batch_size"]),
            "partition": str(dataset["partition"]),
            "pairing_mode": str(federation["pairing_mode"]),
            "sampling_protocol": str(federation["sampling_protocol"]),
            "deterministic_client_training": bool(
                federation["deterministic_client_training"]
            ),
            "client_num_gpus": float(ray["client_num_gpus"]),
            "attack_enabled": bool(attack["enabled"]),
            "attack_type": str(attack["type"]),
            "trial_plan_hash": str(federation["trial_plan_hash"]),
        }
        contracts.append(contract)
        expected = {
            "num_rounds": 60,
            "num_clients": 20,
            "clients_per_round": 10,
            "max_client_samples": 0,
            "batch_size": 48,
            "partition": "iid",
            "pairing_mode": "strict",
            "sampling_protocol": "principal_uniform",
            "deterministic_client_training": True,
            "client_num_gpus": 0.25,
            "attack_enabled": False,
            "attack_type": "none",
        }
        for field, required in expected.items():
            if contract[field] != required:
                failures.append(
                    f"calibration runtime mismatch {run_id}/{field}: "
                    f"observed={contract[field]!r}, required={required!r}"
                )
        status = status_by_run.get(run_id)
        if not status or status.get("state") != "completed" or int(
            status.get("last_round", 0)
        ) != 60 or int(status.get("exit_code", -1)) != 0:
            failures.append(f"calibration run is incomplete: {run_id}")
        source = sources.get(run_id)
        if not source:
            failures.append(f"calibration provenance missing source run: {run_id}")
        elif str(source.get("trial_plan_hash")) != contract["trial_plan_hash"]:
            failures.append(f"calibration TrialPlan hash mismatch: {run_id}")
        manifest_source = manifest_sources.get(run_id)
        if manifest_source != source:
            failures.append(
                f"candidate/validation calibration provenance mismatch: {run_id}"
            )
        plan_path = Path(str(federation.get("trial_plan_path", "")))
        if not plan_path.is_absolute():
            plan_path = (root.parent.parent / plan_path).resolve()
        if not plan_path.is_file():
            failures.append(f"missing calibration TrialPlan file: {run_id}")
        else:
            try:
                plan = TrialPlanV1.load(plan_path)
                if plan.trial_plan_hash != contract["trial_plan_hash"]:
                    failures.append(f"calibration TrialPlan content mismatch: {run_id}")
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                failures.append(f"invalid calibration TrialPlan {run_id}: {exc}")
        data_path = root / "raw" / f"{run_id}_data_manifest.json"
        if not data_path.is_file():
            failures.append(f"missing calibration data manifest: {run_id}")
        else:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            stored = str(data.pop("sha256", ""))
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
            actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if stored != actual:
                failures.append(f"calibration data manifest hash mismatch: {run_id}")
            if int(data.get("max_client_samples", -1)) != 0:
                failures.append(f"calibration client data was capped: {run_id}")
            if source and str(source.get("data_manifest_sha256")) != stored:
                failures.append(f"calibration data manifest provenance mismatch: {run_id}")
        round_path = root / "rounds" / f"{run_id}.csv"
        client_path = root / "raw" / f"{run_id}_clients.csv"
        round_complete = False
        if round_path.is_file():
            round_frame = pd.read_csv(round_path)
            if "round" in round_frame:
                round_values = pd.to_numeric(
                    round_frame["round"], errors="coerce"
                )
                training_rounds = round_values[round_values.between(1, 60)]
                round_complete = (
                    len(training_rounds) == 60
                    and set(training_rounds.astype(int)) == set(range(1, 61))
                    and not training_rounds.duplicated().any()
                    and round_values.dropna().isin(range(0, 61)).all()
                )
        if not round_complete:
            failures.append(f"calibration round rows are incomplete: {run_id}")
        if not client_path.is_file() or len(pd.read_csv(client_path)) != 600:
            failures.append(f"calibration client rows are incomplete: {run_id}")
        if source and round_path.is_file() and str(source.get("round_sha256")) != _sha256(round_path):
            failures.append(f"calibration round provenance mismatch: {run_id}")
        if source and client_path.is_file() and str(source.get("client_sha256")) != _sha256(client_path):
            failures.append(f"calibration client provenance mismatch: {run_id}")
        if source and str(source.get("config_sha256")) != _sha256(config_path):
            failures.append(f"calibration config provenance mismatch: {run_id}")
        sidecar = root / "raw" / f"{run_id}_rtc_v3_sketches.parquet"
        if not sidecar.is_file():
            failures.append(f"missing calibration sketch sidecar: {run_id}")
        else:
            sidecar_frame = pd.read_parquet(sidecar)
            sketch_columns = [
                column for column in sidecar_frame if column.startswith("rtc_v3_sketch_full_")
            ]
            if len(sidecar_frame) != 600 or len(sketch_columns) != 256:
                failures.append(
                    f"invalid calibration sketch sidecar shape: {run_id}, "
                    f"rows={len(sidecar_frame)}, dimension={len(sketch_columns)}"
                )
            if source and str(source.get("sketch_sidecar_sha256")) != _sha256(sidecar):
                failures.append(f"calibration sketch sidecar hash mismatch: {run_id}")
            if source and int(source.get("sketch_dimension", -1)) != 256:
                failures.append(f"calibration sketch dimension provenance mismatch: {run_id}")
            if source and str(source.get("sketch_algorithm_version")) != "splitmix64_v1":
                failures.append(f"calibration sketch algorithm provenance mismatch: {run_id}")
    if observed_seeds != expected_seeds:
        failures.append(
            f"calibration seed set mismatch: observed={sorted(observed_seeds)}, "
            f"required={sorted(expected_seeds)}"
        )
    reserved = set(int(value) for value in manifest["metadata"]["reserved_evaluation_seeds"])
    if observed_seeds & reserved or reserved != {42}:
        failures.append("calibration and held-out seed sets are not exactly disjoint")
    if len(contracts) != 2:
        failures.append(f"expected two calibration runtime contracts, found {len(contracts)}")
    if set(manifest_sources) != set(sources) or set(sources) != set(status_by_run):
        failures.append(
            "calibration source/status run-id sets do not match exactly: "
            f"manifest={sorted(manifest_sources)}, validation={sorted(sources)}, "
            f"status={sorted(status_by_run)}"
        )
    runtime_model_hash = ""
    if len(model_contracts) != 1:
        failures.append(
            f"calibration model contracts are inconsistent: {sorted(model_contracts)}"
        )
    else:
        architecture, num_classes, pretrained, dataset_name = next(iter(model_contracts))
        model = get_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dataset_name=dataset_name,
        )
        runtime_model_hash = model_metadata_hash(get_parameters(model))
        if runtime_model_hash != str(manifest.get("model_metadata_hash", "")):
            failures.append(
                "calibration model metadata hash mismatch: "
                f"runtime={runtime_model_hash}, manifest={manifest.get('model_metadata_hash')}"
            )
    return {
        "contracts": contracts,
        "model_contracts": [list(value) for value in sorted(model_contracts)],
        "runtime_model_metadata_hash": runtime_model_hash,
        "observed_seeds": sorted(observed_seeds),
        "reserved_evaluation_seeds": sorted(reserved),
        "source_run_count": len(sources),
    }, failures


def _calibration_validation_audit(
    validation: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Independently verify every formal-calibration promotion gate."""
    failures: list[str] = []
    if not bool(validation.get("passed", False)):
        failures.append("formal clean calibration validation failed")
    if str(validation.get("content_hash", "")) != str(manifest.get("content_hash", "")):
        failures.append("calibration validation content hash differs from candidate")
    calibration_seeds = {int(value) for value in validation.get("calibration_seeds", [])}
    evaluation_seeds = {int(value) for value in validation.get("evaluation_seeds", [])}
    if calibration_seeds != {40, 41} or evaluation_seeds != {42}:
        failures.append(
            "calibration/evaluation seed sets are not exactly {40,41}/{42}: "
            f"calibration={sorted(calibration_seeds)}, evaluation={sorted(evaluation_seeds)}"
        )
    if calibration_seeds & evaluation_seeds or not bool(validation.get("seed_sets_disjoint", False)):
        failures.append("calibration validation seed sets are not disjoint")
    if not bool(validation.get("q_min_less_than_one", False)):
        failures.append("calibration validation q_min gate failed")
    if not bool(validation.get("q_value_less_than_one", False)):
        failures.append("calibration validation q_value gate failed")

    envelope_evidence: dict[str, Any] = {}
    for family, key, budget_field in (
        ("server", "server_budget_envelopes", "budget_min"),
        ("principal", "principal_budget_envelopes", "beta_min"),
        ("cone", "cone_budget_envelopes", "budget_min"),
    ):
        entries = validation.get(key, {})
        violations: list[dict[str, Any]] = []
        for name, item in entries.items():
            observed = float(item["observed_max"])
            budget = float(item[budget_field])
            if not np.isfinite(observed) or not np.isfinite(budget) or observed > budget + 1e-12:
                violations.append({"name": str(name), "observed": observed, "budget": budget})
        if not entries:
            failures.append(f"missing {family} budget envelope evidence")
        if violations:
            failures.append(f"{family} budget envelope violations: {violations}")
        envelope_evidence[family] = {
            "entry_count": len(entries),
            "violation_count": len(violations),
            "violations": violations,
        }

    clustering = validation.get("cone_clustering", {})
    phase_evidence: dict[str, Any] = {}
    manifest_phases = manifest.get("stable_cones", {}).get("phases", {})
    if set(clustering) != set(PHASES):
        failures.append(
            f"calibration cone phases mismatch: observed={sorted(clustering)}, required={list(PHASES)}"
        )
    for phase in PHASES:
        item = clustering.get(phase, {})
        coverage = float(item.get("coverage", np.nan))
        prototype_count = int(item.get("prototype_count", -1))
        expected_count = len(manifest_phases.get(phase, {}).get("prototypes", []))
        drift = float(item.get("max_angular_drift", np.nan))
        expected_drift = float(manifest_phases.get(phase, {}).get("max_angular_drift", np.nan))
        if not np.isfinite(coverage) or coverage < 0.95 - 1e-12:
            failures.append(f"calibration cone coverage below 95% in {phase}: {coverage}")
        if prototype_count < 1 or prototype_count > 16 or prototype_count != expected_count:
            failures.append(
                f"calibration prototype count mismatch in {phase}: "
                f"validation={prototype_count}, manifest={expected_count}"
            )
        if not np.isfinite(drift) or not np.isclose(drift, expected_drift, rtol=0.0, atol=1e-15):
            failures.append(
                f"calibration drift provenance mismatch in {phase}: "
                f"validation={drift}, manifest={expected_drift}"
            )
        phase_evidence[phase] = {
            "coverage": coverage,
            "prototype_count": prototype_count,
            "max_angular_drift": drift,
        }
    sketch_dimension = int(validation.get("cone_sketch_dimension", -1))
    if sketch_dimension != 256:
        failures.append(f"calibration cone sketch dimension is not 256: {sketch_dimension}")
    return {
        "budget_envelopes": envelope_evidence,
        "phases": phase_evidence,
        "calibration_seeds": sorted(calibration_seeds),
        "evaluation_seeds": sorted(evaluation_seeds),
        "sketch_dimension": sketch_dimension,
    }, failures


def _rtc_summary(root: Path, *, attacks: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(root / "comparison_runs.csv")
    return frame[
        (frame["defense"].astype(str) == "rtc_v3")
        & frame["attack"].astype(str).isin(set(attacks))
    ].copy()


def _reverse_mapping_asr(confusion_json: Any) -> float:
    """Return y -> C-1-y success rate from a serialized confusion matrix."""
    matrix = np.asarray(json.loads(str(confusion_json)), dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"invalid all-reverse confusion-matrix shape: {matrix.shape}")
    total = int(matrix.sum())
    if total <= 0:
        raise ValueError("all-reverse confusion matrix has no observations")
    labels = np.arange(matrix.shape[0])
    success = int(matrix[labels, matrix.shape[0] - 1 - labels].sum())
    return float(success / total)


def _attack_objective_metrics(
    summary: Mapping[str, Any],
    rounds: pd.DataFrame,
    *,
    attack: str,
) -> dict[str, Any]:
    """Calculate promotion metrics from immutable per-round evidence.

    Older all-reverse runs predate the explicit ``server_asr`` column.  Their
    full confusion matrices are sufficient to recover the exact attack
    objective without retraining or modifying experimental artifacts.
    """
    ordered = rounds.copy()
    ordered["round"] = pd.to_numeric(ordered["round"], errors="raise").astype(int)
    ordered = ordered.sort_values("round")
    active = ordered[ordered["planned_attack_active"].map(_as_bool)]
    if active.empty:
        raise ValueError(f"no active attack rounds for {attack}")

    asr = (
        pd.to_numeric(active["server_asr"], errors="coerce")
        if "server_asr" in active
        else pd.Series(np.nan, index=active.index, dtype=float)
    )
    definition = "reported server_asr"
    if asr.isna().any():
        if attack != "label_flip_all_reverse":
            raise ValueError(f"non-finite server_asr for {attack}")
        if "server_confusion_matrix_json" not in active:
            raise ValueError("all-reverse run lacks server_asr and confusion matrices")
        asr = active["server_confusion_matrix_json"].map(_reverse_mapping_asr)
        definition = "derived from confusion matrix as P(prediction=C-1-true_label)"
    values = asr.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"non-finite attack-objective ASR for {attack}")
    auc = float(values[0]) if len(values) == 1 else float(np.trapezoid(values, dx=1.0))
    accuracy = pd.to_numeric(ordered["server_accuracy"], errors="coerce").dropna()
    if accuracy.empty or not np.isfinite(accuracy.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite server accuracy for {attack}")
    return {
        "active_asr": float(values.mean()),
        "active_asr_auc": auc,
        "peak_asr": float(values.max()),
        "final_accuracy": float(accuracy.iloc[-1]),
        "best_accuracy": float(accuracy.max()),
        "asr_definition": definition,
    }


def _round_frames(root: Path, *, attack: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "rounds").glob("*.csv")):
        frame = pd.read_csv(path)
        if "defense" not in frame or not (frame["defense"].astype(str) == "rtc_v3").any():
            continue
        if attack is not None and not (frame["attack"].astype(str) == attack).any():
            continue
        # The round table has hundreds of metrics; compact it before adding
        # provenance to avoid pandas fragmentation warnings in the final audit.
        frame = frame.copy()
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError(f"no RTC-v3 round files found under {root}")
    return pd.concat(frames, ignore_index=True)


def _client_frames(root: Path, *, attack: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "raw").glob("*_clients.csv")):
        frame = pd.read_csv(path)
        if "rtc_v3_cone_full" not in frame:
            continue
        config_path = path.with_name(path.name.replace("_clients.csv", "_config.json"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        observed_attack = str(config["security"]["attack"]["type"])
        if attack is not None and observed_attack != attack:
            continue
        frame["attack"] = observed_attack
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError(f"no RTC-v3 client files found under {root}")
    return pd.concat(frames, ignore_index=True)


def _cone_budget_activation(rounds: pd.DataFrame) -> dict[str, Any]:
    """Summarize positive cone-budget constraints that reach their envelope."""
    pattern = re.compile(r"^(fit_rtc_v3_.+_cone_.+)_budget_w(1|4|8)$")
    active_by_window = {"1": 0, "4": 0, "8": 0}
    eligible_by_window = {"1": 0, "4": 0, "8": 0}
    maximum = 0.0
    for budget_column in rounds.columns:
        match = pattern.match(str(budget_column))
        if not match:
            continue
        prefix, window = match.groups()
        used_column = f"{prefix}_used_w{window}"
        if used_column not in rounds:
            continue
        budget = pd.to_numeric(rounds[budget_column], errors="coerce")
        used = pd.to_numeric(rounds[used_column], errors="coerce")
        eligible = budget.notna() & used.notna() & budget.gt(1e-12)
        if not eligible.any():
            continue
        ratios = used[eligible] / budget[eligible]
        maximum = max(maximum, float(ratios.max()))
        eligible_by_window[window] += int(eligible.sum())
        active_by_window[window] += int(
            (used[eligible] >= budget[eligible] - 1e-8).sum()
        )
    eligible_total = sum(eligible_by_window.values())
    active_total = sum(active_by_window.values())
    return {
        "rate": float(active_total / eligible_total) if eligible_total else 0.0,
        "active_cells": active_total,
        "eligible_cells": eligible_total,
        "rate_by_window": {
            window: (
                float(active_by_window[window] / eligible_by_window[window])
                if eligible_by_window[window] else 0.0
            )
            for window in ("1", "4", "8")
        },
        "active_cells_by_window": active_by_window,
        "eligible_cells_by_window": eligible_by_window,
        "max_used_over_budget": maximum,
    }


def _clean_cone_audit(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rounds = _round_frames(root)
    rounds = rounds[
        pd.to_numeric(rounds["round"], errors="coerce").ge(1)
    ].copy()
    clients = _client_frames(root)
    failures: list[str] = []
    modes = set(rounds.get("fit_rtc_v3_cone_mode", pd.Series(dtype=str)).dropna().astype(str))
    if modes != {"offline_controlled_v1"}:
        failures.append(f"held-out clean used unexpected cone modes: {sorted(modes)}")
    selected = pd.to_numeric(
        rounds["fit_selected_malicious_clients"], errors="coerce"
    ).fillna(0.0) + pd.to_numeric(
        rounds["fit_selected_benign_clients"], errors="coerce"
    ).fillna(0.0)
    overflow = pd.to_numeric(
        rounds["fit_rtc_v3_cone_overflow_assignments"], errors="coerce"
    ).fillna(0.0)
    invalid_selected = rounds[
        ~np.isclose(selected.to_numpy(dtype=float), 10.0, rtol=0.0, atol=0.0)
    ]
    if not invalid_selected.empty:
        failures.append(
            "held-out clean selected-client count differs from 10: "
            + str(invalid_selected[["source_file", "round"]].head(10).to_dict(orient="records"))
        )
    invalid_overflow = rounds[(overflow < 0.0) | (overflow > selected)]
    if not invalid_overflow.empty:
        failures.append(
            "held-out clean overflow count is outside [0, selected]: "
            + str(invalid_overflow[["source_file", "round"]].head(10).to_dict(orient="records"))
        )
    expected_phase = pd.to_numeric(rounds["round"], errors="coerce").map(
        lambda value: "warmup" if value <= 10 else ("steady" if value <= 30 else "late")
    )
    observed_phase = rounds["fit_rtc_v3_cone_profile"].astype(str)
    phase_mismatch = rounds[observed_phase != expected_phase]
    if not phase_mismatch.empty:
        failures.append(
            "held-out clean phase boundary mismatch: "
            + str(
                phase_mismatch[["source_file", "round", "fit_rtc_v3_cone_profile"]]
                .head(10).to_dict(orient="records")
            )
        )
    overall = float(overflow.sum() / max(float(selected.sum()), 1.0))
    if overall > 0.05 + 1e-12:
        failures.append(f"clean overflow {overall:.6f} exceeds 0.05")
    per_phase: dict[str, float] = {}
    expected_ids: dict[str, list[str]] = {}
    observed_ids: dict[str, list[list[str]]] = {}
    drift: dict[str, dict[str, float]] = {}
    phases = manifest["stable_cones"]["phases"]
    for phase in PHASES:
        phase_rounds = rounds[rounds["fit_rtc_v3_cone_profile"].astype(str) == phase]
        phase_selected = (
            pd.to_numeric(phase_rounds["fit_selected_malicious_clients"], errors="coerce").fillna(0.0)
            + pd.to_numeric(phase_rounds["fit_selected_benign_clients"], errors="coerce").fillna(0.0)
        )
        phase_overflow = pd.to_numeric(
            phase_rounds["fit_rtc_v3_cone_overflow_assignments"], errors="coerce"
        ).fillna(0.0)
        per_phase[phase] = float(
            phase_overflow.sum() / max(float(phase_selected.sum()), 1.0)
        )
        expected = sorted(
            str(item["cone_id"]) for item in phases[phase]["prototypes"]
        )
        expected_ids[phase] = expected
        id_column = "fit_rtc_v3_cone_prototype_ids_full"
        if id_column not in phase_rounds:
            failures.append(f"missing runtime prototype IDs for phase {phase}")
            observed_ids[phase] = []
        else:
            observed = [
                sorted(value for value in str(raw).split(",") if value)
                for raw in phase_rounds[id_column].fillna("")
            ]
            observed_ids[phase] = observed
            mismatched = [
                (int(row["round"]), str(row["source_file"]), value)
                for (_, row), value in zip(phase_rounds.iterrows(), observed)
                if value != expected
            ]
            if not observed or mismatched:
                detail = mismatched[0] if mismatched else "no observations"
                failures.append(
                    f"runtime prototype IDs changed in phase {phase}: {detail}"
                )
        count = pd.to_numeric(
            phase_rounds.get("fit_rtc_v3_cone_prototype_count"), errors="coerce"
        )
        bad_count = count[(count > 16) | (count != len(expected))]
        if count.empty or not bad_count.empty:
            detail = (
                "no observations"
                if count.empty
                else f"round={int(phase_rounds.loc[bad_count.index[0], 'round'])}, value={float(bad_count.iloc[0])}"
            )
            failures.append(
                f"runtime prototype count mismatch in phase {phase}: {detail}"
            )
        observed_drift = float(
            pd.to_numeric(
                phase_rounds.get("fit_rtc_v3_cone_update_max_angular_drift"),
                errors="coerce",
            ).fillna(0.0).max()
        )
        allowed_drift = float(phases[phase]["max_angular_drift"])
        drift[phase] = {"observed": observed_drift, "allowed": allowed_drift}
        if observed_drift > allowed_drift + 1e-10:
            worst = pd.to_numeric(
                phase_rounds["fit_rtc_v3_cone_update_max_angular_drift"],
                errors="coerce",
            ).idxmax()
            failures.append(
                "prototype angular drift exceeded in phase "
                f"{phase}: round={int(phase_rounds.loc[worst, 'round'])}, "
                f"observed={observed_drift}, allowed={allowed_drift}"
            )
    invalid_update_eligible = 0
    invalid_q_eligible = 0
    eligible_column = "rtc_v3_cone_update_eligible_full"
    if eligible_column not in clients:
        failures.append("missing per-client controlled-update eligibility")
    else:
        eligible = clients[eligible_column].map(_as_bool)
        invalid = eligible & (
            clients["clipped"].map(_as_bool)
            | clients["capped"].map(_as_bool)
            | clients["quarantined"].map(_as_bool)
        )
        invalid_update_eligible = int(invalid.sum())
        if invalid_update_eligible:
            example_columns = [
                column for column in ("source_file", "round", "cid", "principal_id")
                if column in clients
            ]
            examples = clients.loc[invalid, example_columns].head(5)
            failures.append(
                f"{invalid_update_eligible} clipped/capped/quarantined clients were "
                f"update-eligible: {examples.to_dict(orient='records')}"
            )
        q_columns = ("rtc_v3_cumulative_q_full", "rtc_v3_direction_q_full")
        missing_q = [column for column in q_columns if column not in clients]
        if missing_q:
            failures.append(
                f"missing per-client controlled-update q evidence: {missing_q}"
            )
        else:
            cumulative_q = pd.to_numeric(clients[q_columns[0]], errors="coerce")
            direction_q = pd.to_numeric(clients[q_columns[1]], errors="coerce")
            invalid_q = eligible & (
                cumulative_q.lt(1.0 - 1e-12)
                | direction_q.lt(1.0 - 1e-12)
                | cumulative_q.isna()
                | direction_q.isna()
            )
            invalid_q_eligible = int(invalid_q.sum())
            if invalid_q_eligible:
                example_columns = [
                    column
                    for column in (
                        "source_file",
                        "round",
                        "cid",
                        "principal_id",
                        *q_columns,
                    )
                    if column in clients
                ]
                examples = clients.loc[invalid_q, example_columns].head(5)
                failures.append(
                    f"{invalid_q_eligible} cumulative/direction q<1 clients were "
                    f"update-eligible: {examples.to_dict(orient='records')}"
                )
    invalid_single_principal_updates = 0
    support_column = "fit_rtc_v3_cone_update_principal_support_json"
    applied_column = "fit_rtc_v3_cone_update_applied_ids_json"
    if support_column not in rounds or applied_column not in rounds:
        failures.append("missing per-cone update principal-support evidence")
    else:
        for _, row in rounds.iterrows():
            try:
                support = json.loads(str(row[support_column]))
                applied = json.loads(str(row[applied_column]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    f"invalid cone update support evidence at round={int(row['round'])}: {exc}"
                )
                continue
            for resolution, cone_ids in applied.items():
                for cone_id in cone_ids:
                    principal_count = int(support.get(resolution, {}).get(cone_id, 0))
                    if principal_count < 2:
                        invalid_single_principal_updates += 1
                        failures.append(
                            "prototype updated without two-principal support: "
                            f"source={row.get('source_file')}, round={int(row['round'])}, "
                            f"resolution={resolution}, cone={cone_id}, "
                            f"principal_count={principal_count}"
                        )
    cone_budget_activation = _cone_budget_activation(rounds)
    if cone_budget_activation["eligible_cells"] == 0:
        failures.append("missing positive cone-budget activation evidence")
    return {
        "overflow_rate": overall,
        "overflow_rate_by_phase": per_phase,
        "expected_prototype_ids": expected_ids,
        "runtime_prototype_ids": observed_ids,
        "prototype_drift": drift,
        "invalid_update_eligible_clients": invalid_update_eligible,
        "invalid_q_update_eligible_clients": invalid_q_eligible,
        "invalid_single_principal_updates": invalid_single_principal_updates,
        "update_candidates": float(
            pd.to_numeric(rounds["fit_rtc_v3_cone_update_candidate_count"], errors="coerce").fillna(0).sum()
        ),
        "updates_applied": float(
            pd.to_numeric(rounds["fit_rtc_v3_cone_update_applied_count"], errors="coerce").fillna(0).sum()
        ),
        "drift_clipped": float(
            pd.to_numeric(rounds["fit_rtc_v3_cone_update_drift_clipped_count"], errors="coerce").fillna(0).sum()
        ),
        "cone_budget_activation": cone_budget_activation,
    }, failures


def _attack_client_diagnostics(
    root: Path,
    *,
    attack: str,
    rounds: pd.DataFrame,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    """Collect active-window diagnostics and reject incomplete evidence."""
    failures: list[str] = []
    active_mask = rounds["planned_attack_active"].map(_as_bool)
    active_rounds = sorted(
        pd.to_numeric(rounds.loc[active_mask, "round"], errors="coerce")
        .dropna().astype(int).tolist()
    )
    if not active_rounds:
        failures.append(f"{attack} {label} has no active diagnostic rounds")
    clients = _client_frames(root, attack=attack)
    if "round" not in clients:
        failures.append(f"{attack} {label} client evidence has no round column")
        clients = clients.iloc[0:0]
    else:
        client_rounds = pd.to_numeric(clients["round"], errors="coerce")
        clients = clients[client_rounds.isin(active_rounds)].copy()
    malicious_mask = clients["is_malicious"].map(_as_bool)
    malicious = clients[malicious_mask]
    benign = clients[~malicious_mask]
    if malicious.empty:
        failures.append(f"{attack} {label} has no malicious client observations")
    if benign.empty:
        failures.append(f"{attack} {label} has no benign client observations")

    weight_column = "fit_malicious_aggregation_weight_share"
    active_weight = (
        pd.to_numeric(rounds.loc[active_mask, weight_column], errors="coerce")
        if weight_column in rounds
        else pd.Series(dtype=float)
    )
    weight_share = (
        float(active_weight.mean())
        if len(active_weight) and active_weight.notna().all()
        else np.nan
    )
    if not np.isfinite(weight_share):
        failures.append(f"{attack} {label} malicious weight share is non-finite")
    benign_clipping_rate = (
        float(benign["clipped"].map(_as_bool).mean()) if not benign.empty else np.nan
    )
    if not np.isfinite(benign_clipping_rate):
        failures.append(f"{attack} {label} benign clipping rate is non-finite")

    cone_values = malicious["rtc_v3_cone_full"] if "rtc_v3_cone_full" in malicious else pd.Series(dtype=object)
    missing_cones = int(cone_values.isna().sum()) if len(cone_values) else 0
    if malicious.shape[0] and (cone_values.empty or missing_cones):
        failures.append(
            f"{attack} {label} missing cone assignments for {missing_cones or malicious.shape[0]} malicious observations"
        )
    normalized_cones = cone_values.fillna("missing").astype(str)
    distribution = {
        str(key): int(value)
        for key, value in normalized_cones.value_counts().items()
    }
    total = int(len(normalized_cones))
    overflow_count = int((normalized_cones == "overflow").sum())
    missing_count = int((normalized_cones == "missing").sum())
    matched_count = total - overflow_count - missing_count
    return {
        "malicious_weight_share": weight_share,
        "diagnostic_rounds": active_rounds,
        "benign_clipping_rate": benign_clipping_rate,
        "malicious_cone_distribution": distribution,
        "malicious_cone_total": total,
        "malicious_overflow_count": overflow_count,
        "malicious_overflow_rate": (
            float(overflow_count / total) if total else np.nan
        ),
        "malicious_matched_count": matched_count,
        "malicious_matched_rate": (
            float(matched_count / total) if total else np.nan
        ),
        "cone_budget_activation": _cone_budget_activation(rounds),
    }, failures


def _attack_runtime_contract(
    root: Path,
    *,
    attack: str,
    label: str,
    reference_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    matches: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted((root / "raw").glob("*_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if (
            str(config.get("security", {}).get("attack", {}).get("type")) == attack
            and str(config.get("security", {}).get("defense", {}).get("type"))
            == "rtc_v3_candidate"
        ):
            matches.append((path, config))
    if len(matches) != 1:
        return {}, [
            f"{attack} {label} expected one RTC-v3 config, found {len(matches)}"
        ]
    path, config = matches[0]
    federation = config["federation"]
    client = config["client"]
    dataset = config["dataset"]
    evaluation = config["evaluation"]
    ray = config["ray"]
    attack_config = config["security"]["attack"]
    contract = {
        "config_path": str(path.resolve()),
        "config_sha256": _sha256(path),
        "seed": int(config["project"]["seed"]),
        "rounds": int(federation["num_rounds"]),
        "num_clients": int(federation["num_clients"]),
        "clients_per_round": int(federation["clients_per_round"]),
        "max_client_samples": int(federation["max_client_samples"]),
        "pairing_mode": str(federation["pairing_mode"]),
        "sampling_protocol": str(federation["sampling_protocol"]),
        "deterministic_client_training": bool(federation["deterministic_client_training"]),
        "batch_size": int(client["batch_size"]),
        "partition": str(dataset["partition"]),
        "max_test_samples": int(evaluation["max_test_samples"]),
        "client_num_gpus": float(ray["client_num_gpus"]),
        "attack_enabled": bool(attack_config["enabled"]),
        "attack_type": str(attack_config["type"]),
        "malicious_fraction": float(attack_config["malicious_fraction"]),
        "attack_start_round": int(attack_config["attack_start_round"]),
        "attack_end_round": int(attack_config["attack_end_round"]),
        "attack_on_rounds": int(attack_config["attack_on_rounds"]),
        "attack_off_rounds": int(attack_config["attack_off_rounds"]),
        "label_flip_poison_fraction": float(attack_config["label_flip_poison_fraction"]),
        "source_label": int(attack_config["source_label"]),
        "target_label": int(attack_config["target_label"]),
    }
    expected = {
        "seed": 42,
        "rounds": 60,
        "num_clients": 20,
        "clients_per_round": 10,
        "max_client_samples": 0,
        "pairing_mode": "strict",
        "sampling_protocol": "principal_uniform",
        "deterministic_client_training": True,
        "batch_size": 48,
        "partition": "iid",
        "max_test_samples": 0,
        "client_num_gpus": 0.25,
        "attack_enabled": True,
        "attack_type": attack,
        "malicious_fraction": 0.2,
        "attack_start_round": 11,
        "attack_end_round": -1,
        "attack_on_rounds": 1,
        "attack_off_rounds": 0,
        "label_flip_poison_fraction": 1.0,
    }
    if attack == "label_flip_targeted":
        expected.update({"source_label": 5, "target_label": 3})
    for key, value in expected.items():
        if contract[key] != value:
            failures.append(
                f"{attack} {label} runtime mismatch {key}: "
                f"observed={contract[key]!r}, required={value!r}"
            )
    if reference_config is not None:
        for section in ("client", "model"):
            if config.get(section) != reference_config.get(section):
                failures.append(
                    f"{attack} {label} {section} config differs from formal calibration"
                )
        for section, keys in (
            ("dataset", ("name", "partition", "dirichlet_alpha", "num_classes", "val_split")),
            ("ray", ("client_num_cpus", "client_num_gpus")),
        ):
            observed = {key: config.get(section, {}).get(key) for key in keys}
            reference = {key: reference_config.get(section, {}).get(key) for key in keys}
            if observed != reference:
                failures.append(
                    f"{attack} {label} {section} config differs from formal calibration: "
                    f"observed={observed}, reference={reference}"
                )
    return contract, failures


def _heldout_clean_contract_audit(
    root: Path,
    *,
    reference_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Verify RTC validation-only clean and its paired FedAvg counterfactual."""
    failures: list[str] = []
    observed: dict[str, tuple[str, Path, Mapping[str, Any]]] = {}
    for path in sorted((root / "raw").glob("*_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        defense_type = str(config["security"]["defense"]["type"])
        role = "rtc_validation_only" if defense_type == "rtc_v3_candidate" else (
            "fedavg_counterfactual" if defense_type == "none" else ""
        )
        if role:
            observed[role] = (
                path.name.removesuffix("_config.json"), path, config
            )
    required_roles = {"rtc_validation_only", "fedavg_counterfactual"}
    if set(observed) != required_roles:
        failures.append(
            f"held-out clean role set mismatch: observed={sorted(observed)}, required={sorted(required_roles)}"
        )
    contracts: dict[str, Any] = {}
    plan_hashes: set[str] = set()
    for role in sorted(required_roles):
        if role not in observed:
            continue
        identifier, path, config = observed[role]
        federation = config["federation"]
        client = config["client"]
        dataset = config["dataset"]
        evaluation = config["evaluation"]
        ray = config["ray"]
        attack = config["security"]["attack"]
        contract = {
            "run_id": identifier,
            "config_path": str(path.resolve()),
            "config_sha256": _sha256(path),
            "seed": int(config["project"]["seed"]),
            "rounds": int(federation["num_rounds"]),
            "num_clients": int(federation["num_clients"]),
            "clients_per_round": int(federation["clients_per_round"]),
            "max_client_samples": int(federation["max_client_samples"]),
            "pairing_mode": str(federation["pairing_mode"]),
            "sampling_protocol": str(federation["sampling_protocol"]),
            "deterministic_client_training": bool(federation["deterministic_client_training"]),
            "trial_plan_hash": str(federation["trial_plan_hash"]),
            "batch_size": int(client["batch_size"]),
            "partition": str(dataset["partition"]),
            "max_test_samples": int(evaluation["max_test_samples"]),
            "client_num_gpus": float(ray["client_num_gpus"]),
            "attack_type": str(attack["type"]),
            "malicious_fraction": float(attack["malicious_fraction"]),
            "attack_start_round": int(attack["attack_start_round"]),
            "attack_end_round": int(attack["attack_end_round"]),
        }
        contracts[role] = contract
        plan_hashes.add(contract["trial_plan_hash"])
        expected = {
            "seed": 42,
            "rounds": 60,
            "num_clients": 20,
            "clients_per_round": 10,
            "max_client_samples": 0,
            "pairing_mode": "strict",
            "sampling_protocol": "principal_uniform",
            "deterministic_client_training": True,
            "batch_size": 48,
            "partition": "iid",
            "max_test_samples": 0,
            "client_num_gpus": 0.25,
            "attack_start_round": 61,
            "attack_end_round": -1,
        }
        if role == "rtc_validation_only":
            expected.update({
                "attack_type": "label_flip_targeted",
                "malicious_fraction": 0.2,
            })
        else:
            expected.update({"attack_type": "none", "malicious_fraction": 0.0})
        for key, value in expected.items():
            if contract[key] != value:
                failures.append(
                    f"held-out clean {role} mismatch {key}: "
                    f"observed={contract[key]!r}, required={value!r}"
                )
        round_path = root / "rounds" / f"{identifier}.csv"
        if not round_path.is_file():
            failures.append(f"held-out clean missing round file: {identifier}")
        else:
            rounds = pd.read_csv(round_path)
            planned = (
                pd.to_numeric(rounds["planned_attack_active"], errors="coerce")
                if "planned_attack_active" in rounds
                else pd.Series(np.nan, index=rounds.index, dtype=float)
            )
            if planned.isna().any() or float(planned.fillna(0).sum()) != 0.0:
                failures.append(
                    f"held-out clean {role} contains active attack rounds"
                )
        if reference_config is not None:
            for section in ("client", "model"):
                if config.get(section) != reference_config.get(section):
                    failures.append(
                        f"held-out clean {role} {section} differs from formal calibration"
                    )
    if len(plan_hashes) != 1 or "" in plan_hashes:
        failures.append(
            f"held-out clean RTC/FedAvg TrialPlan mismatch: {sorted(plan_hashes)}"
        )
    return {
        "contracts": contracts,
        "shared_trial_plan_hash": next(iter(plan_hashes), "") if len(plan_hashes) == 1 else "",
    }, failures


def _attack_audit(
    candidate_root: Path,
    baseline_root: Path,
    *,
    reference_config: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    attacks = {"label_flip_targeted", "label_flip_all_reverse"}
    candidate = _rtc_summary(candidate_root, attacks=attacks)
    baseline = _rtc_summary(baseline_root, attacks=attacks)
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for attack in sorted(attacks):
        left = candidate[candidate["attack"].astype(str) == attack]
        right = baseline[baseline["attack"].astype(str) == attack]
        if len(left) != 1 or len(right) != 1:
            failures.append(f"expected one candidate and baseline row for {attack}")
            continue
        new = left.iloc[0]
        old = right.iloc[0]
        join_key: dict[str, Any] = {}
        mismatched_join_keys: list[str] = []
        for key in ATTACK_JOIN_KEYS:
            if key not in new or key not in old:
                mismatched_join_keys.append(f"{key}=missing")
                continue
            left_value = _json_safe(new[key])
            right_value = _json_safe(old[key])
            join_key[key] = left_value
            if left_value != right_value:
                mismatched_join_keys.append(
                    f"{key}: candidate={left_value!r}, baseline={right_value!r}"
                )
        if mismatched_join_keys:
            failures.append(
                f"strict attack join-key mismatch for {attack}: "
                + "; ".join(mismatched_join_keys)
            )
            # Never calculate or publish a candidate/baseline delta for rows
            # that cannot be joined on the preregistered strict condition.
            continue
        candidate_rounds = _round_frames(candidate_root, attack=attack)
        baseline_rounds = _round_frames(baseline_root, attack=attack)
        candidate_metrics = _attack_objective_metrics(new, candidate_rounds, attack=attack)
        baseline_metrics = _attack_objective_metrics(old, baseline_rounds, attack=attack)
        active_delta = (
            float(candidate_metrics["active_asr"])
            - float(baseline_metrics["active_asr"])
        )
        accuracy_delta = (
            float(candidate_metrics["final_accuracy"])
            - float(baseline_metrics["final_accuracy"])
        )
        if active_delta > 0.02 + 1e-12:
            failures.append(f"{attack} active ASR regressed by {active_delta:.6f}")
        if accuracy_delta < -0.01 - 1e-12:
            failures.append(f"{attack} final accuracy regressed by {accuracy_delta:.6f}")
        candidate_modes = set(
            candidate_rounds.get("fit_rtc_v3_cone_mode", pd.Series(dtype=str))
            .dropna().astype(str)
        )
        baseline_modes = set(
            baseline_rounds.get("fit_rtc_v3_cone_mode", pd.Series(dtype=str))
            .dropna().astype(str)
        )
        if candidate_modes != {"offline_controlled_v1"}:
            failures.append(
                f"{attack} candidate used unexpected cone modes: {sorted(candidate_modes)}"
            )
        if baseline_modes != {"legacy_online_v1"}:
            failures.append(
                f"{attack} baseline used unexpected cone modes: {sorted(baseline_modes)}"
            )
        candidate_diagnostics, candidate_diagnostic_failures = _attack_client_diagnostics(
            candidate_root,
            attack=attack,
            rounds=candidate_rounds,
            label="candidate",
        )
        baseline_diagnostics, baseline_diagnostic_failures = _attack_client_diagnostics(
            baseline_root,
            attack=attack,
            rounds=baseline_rounds,
            label="baseline",
        )
        failures.extend(candidate_diagnostic_failures)
        failures.extend(baseline_diagnostic_failures)
        candidate_contract, candidate_contract_failures = _attack_runtime_contract(
            candidate_root,
            attack=attack,
            label="candidate",
            reference_config=reference_config,
        )
        baseline_contract, baseline_contract_failures = _attack_runtime_contract(
            baseline_root,
            attack=attack,
            label="baseline",
            reference_config=reference_config,
        )
        failures.extend(candidate_contract_failures)
        failures.extend(baseline_contract_failures)
        rows.append({
            "attack": attack,
            "trial_plan_hash": str(new["trial_plan_hash"]),
            "join_key": join_key,
            "candidate": {
                **candidate_metrics,
                **candidate_diagnostics,
                "runtime_contract": candidate_contract,
            },
            "baseline": {
                **baseline_metrics,
                **baseline_diagnostics,
                "runtime_contract": baseline_contract,
            },
            "delta": {
                "active_asr": active_delta,
                "final_accuracy": accuracy_delta,
            },
        })
    return rows, failures


def _artifact_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in _artifact_files(paths)}


def _artifact_files(paths: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            for pattern in (
                "execution_validation.csv", "comparison_runs.csv",
                "experiment_manifest.json", "calibration_matrix.csv",
                "trial_plans/*.json", "rounds/*.csv", "status/*.json",
                "raw/*_clients.csv", "raw/*_config.json", "raw/*_data_manifest.json",
                "raw/*_pairing_manifest.json",
                "raw/*_rtc_v3_sketches.parquet",
            ):
                files.update(item.resolve() for item in path.glob(pattern))
    return sorted(files)


def _source_directory_attestations(roots: Mapping[str, Path]) -> dict[str, Any]:
    attestations: dict[str, Any] = {}
    for name, root in roots.items():
        files = _artifact_files([root])
        entries = [
            {
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in files
        ]
        canonical = json.dumps(
            [(item["relative_path"], item["sha256"]) for item in entries],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        attestations[name] = {
            "path": str(root),
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "file_count": len(entries),
            "files": entries,
        }
    return attestations


def _git_summary(workspace: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=workspace, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def _recommended_actions(failures: Sequence[str]) -> list[str]:
    actions: list[str] = []
    text = "\n".join(failures).lower()
    if "overflow" in text or "prototype" in text or "angular drift" in text:
        actions.append(
            "Revisit phase clustering coverage or controlled-update drift calibration; "
            "do not relax held-out thresholds in place."
        )
    if "asr" in text:
        actions.append(
            "Diagnose the failed attack by round/cone and recalibrate cone budgets on "
            "new independent clean seeds before another candidate evaluation."
        )
    if "accuracy" in text:
        actions.append(
            "Inspect clean clipping and active server/principal/cone budgets before "
            "changing any security threshold."
        )
    if "trial plan" in text or "strict" in text or "match" in text or "sequence" in text:
        actions.append(
            "Discard the invalid comparison and rerun the unchanged condition under "
            "one strict TrialPlan; never join unpaired rows."
        )
    if "pytest" in text:
        actions.append("Fix the regression and rerun the complete test suite.")
    if failures and not actions:
        actions.append(
            "Resolve the recorded gate failure and rerun the unchanged held-out protocol."
        )
    return actions


def _report(path: Path, evidence: Mapping[str, Any]) -> None:
    decision = str(evidence["decision"])
    attacks = evidence.get("attacks", [])
    clean = evidence.get("clean", {})
    lines = [
        "# RTC-v3 Stable Cone V2 Promotion Report",
        "",
        f"Decision: **{decision}**",
        "",
        f"Candidate hash: `{evidence.get('candidate_hash', '')}`",
        f"Baseline manifest hash: `{evidence.get('baseline_manifest_hash', '')}`",
        f"Baseline manifest SHA-256: `{evidence.get('baseline_manifest_sha256', '')}`",
        f"Baseline schema/mode: `{evidence.get('baseline_manifest_schema', '')}` / `{evidence.get('baseline_cone_mode', '')}`",
        f"Promoted hash: `{evidence.get('promoted_hash', '')}`",
        f"Promoted file SHA-256: `{evidence.get('promoted_file_sha256', '')}`",
        f"Candidate/promoted semantic identity: `{evidence.get('promotion_semantic_identity', False)}`",
        f"Clean overflow: `{clean.get('overflow_rate', 'n/a')}`",
        f"Calibration seeds: `{evidence.get('calibration_seeds', [])}`",
        f"Held-out seed: `{evidence.get('held_out_seed')}`",
        f"Calibration schema: `{evidence.get('schema_version', '')}`",
        f"Model metadata hash: `{evidence.get('model_metadata_hash', '')}`",
        "",
        "## Phase overflow",
        "",
    ]
    for phase, value in clean.get("overflow_rate_by_phase", {}).items():
        lines.append(f"- {phase}: `{value}`")
    lines.extend(["", "## Held-out clean runtime contract", ""])
    clean_contract = evidence.get("clean_runtime_contract", {})
    lines.append(
        f"- shared TrialPlan: `{clean_contract.get('shared_trial_plan_hash', '')}`"
    )
    for role, contract in clean_contract.get("contracts", {}).items():
        lines.append(f"- {role}: `{contract}`")
    lines.extend([
        "",
        "## Controlled prototype updates",
        "",
        f"- update candidates: `{clean.get('update_candidates', 'n/a')}`",
        f"- updates applied: `{clean.get('updates_applied', 'n/a')}`",
        f"- drift-clipped updates: `{clean.get('drift_clipped', 'n/a')}`",
        f"- cone budget activation: `{clean.get('cone_budget_activation', 'n/a')}`",
        f"- clipped/capped/quarantined eligibility violations: `{clean.get('invalid_update_eligible_clients', 'n/a')}`",
        f"- cumulative/direction q eligibility violations: `{clean.get('invalid_q_update_eligible_clients', 'n/a')}`",
        f"- single-principal update violations: `{clean.get('invalid_single_principal_updates', 'n/a')}`",
    ])
    lines.extend(["", "## Attack gates", ""])
    for item in attacks:
        candidate = item["candidate"]
        baseline = item["baseline"]
        lines.extend([
            f"### {item['attack']}",
            "",
            f"- TrialPlan: `{item['trial_plan_hash']}`",
            f"- ASR definition: candidate `{candidate['asr_definition']}`, baseline `{baseline['asr_definition']}`",
            f"- active ASR: candidate `{candidate['active_asr']}`, baseline `{baseline['active_asr']}`, Δ `{item['delta']['active_asr']}`",
            f"- ASR AUC: candidate `{candidate['active_asr_auc']}`, baseline `{baseline['active_asr_auc']}`",
            f"- peak ASR: candidate `{candidate['peak_asr']}`, baseline `{baseline['peak_asr']}`",
            f"- final accuracy: candidate `{candidate['final_accuracy']}`, baseline `{baseline['final_accuracy']}`, Δ `{item['delta']['final_accuracy']}`",
            f"- best accuracy: candidate `{candidate['best_accuracy']}`, baseline `{baseline['best_accuracy']}`",
            f"- active-window malicious weight share: candidate `{candidate['malicious_weight_share']}`, baseline `{baseline['malicious_weight_share']}`",
            f"- active-window benign clipping rate: candidate `{candidate['benign_clipping_rate']}`, baseline `{baseline['benign_clipping_rate']}`",
            f"- candidate malicious overflow/matched: `{candidate['malicious_overflow_count']}`/`{candidate['malicious_matched_count']}` "
            f"(rates `{candidate['malicious_overflow_rate']}`/`{candidate['malicious_matched_rate']}`)",
            f"- baseline malicious overflow/matched: `{baseline['malicious_overflow_count']}`/`{baseline['malicious_matched_count']}` "
            f"(rates `{baseline['malicious_overflow_rate']}`/`{baseline['malicious_matched_rate']}`)",
            f"- candidate malicious cone distribution: `{candidate['malicious_cone_distribution']}`",
            f"- baseline malicious cone distribution: `{baseline['malicious_cone_distribution']}`",
            f"- cone budget activation: candidate `{candidate['cone_budget_activation']}`, baseline `{baseline['cone_budget_activation']}`",
            f"- candidate runtime contract: `{candidate['runtime_contract']}`",
            f"- baseline runtime contract: `{baseline['runtime_contract']}`",
            "",
        ])
    lines.extend(["", "## Calibration and phased prototypes", ""])
    for phase, value in evidence.get("stable_cones", {}).get("phases", {}).items():
        prototype_ids = [str(item.get("cone_id")) for item in value.get("prototypes", [])]
        drift = clean.get("prototype_drift", {}).get(phase, {})
        lines.append(
            f"- {phase}: prototypes `{len(value.get('prototypes', []))}`, "
            f"IDs `{prototype_ids}`, "
            f"match threshold `{value.get('match_threshold')}`, "
            f"update threshold `{value.get('update_threshold')}`, "
            f"drift observed/allowed `{drift.get('observed', 'n/a')}`/"
            f"`{drift.get('allowed', value.get('max_angular_drift'))}`"
        )
    lines.extend([
        "",
        f"- q_min: `{evidence.get('q_min')}`",
        f"- q_value: `{evidence.get('q_value')}`",
        f"- TrialPlan hashes: `{', '.join(evidence.get('trial_plan_hashes', []))}`",
    ])
    calibration = evidence.get("calibration_validation", {})
    lines.extend(["", "## Calibration provenance and budgets", ""])
    lines.append(
        "- runtime model metadata hash: "
        f"`{evidence.get('calibration_runtime', {}).get('runtime_model_metadata_hash', '')}`"
    )
    for source in calibration.get("source_runs", []):
        lines.append(
            f"- seed `{source.get('seed')}` / run `{source.get('run_id')}` / "
            f"TrialPlan `{source.get('trial_plan_hash')}` / sketch SHA-256 "
            f"`{source.get('sketch_sidecar_sha256')}`"
        )
    for contract in evidence.get("calibration_runtime", {}).get("contracts", []):
        lines.append(
            f"- runtime seed `{contract.get('seed')}`: batch `{contract.get('batch_size')}`, "
            f"client samples cap `{contract.get('max_client_samples')}`, "
            f"server test cap `{contract.get('max_test_samples')}`, GPU/client "
            f"`{contract.get('client_num_gpus')}`"
        )
    for family, key in (
        ("server", "server_budget_envelopes"),
        ("principal", "principal_budget_envelopes"),
        ("cone", "cone_budget_envelopes"),
    ):
        values = calibration.get(key, {})
        maximum = max(
            (
                float(item["observed_max"])
                / max(
                    float(item.get("budget_min", item.get("beta_min", 0.0))),
                    1e-12,
                )
                for item in values.values()
            ),
            default=0.0,
        )
        lines.append(
            f"- {family} envelopes: `{len(values)}` entries, max observed/budget ratio `{maximum}`"
        )
    for title, key in (
        ("Server budgets", "server_budgets"),
        ("Principal budgets", "principal_betas"),
        ("Cone budgets", "cone_budgets"),
    ):
        lines.extend([
            "",
            f"### {title}",
            "",
            "```json",
            json.dumps(evidence.get(key, {}), indent=2, ensure_ascii=False),
            "```",
        ])
    lines.extend(["", "## Strict gates", ""])
    for root, gates in evidence.get("strict_gates", {}).items():
        lines.append(f"- `{root}`: `{all(gates.values())}`")
        for gate, passed in gates.items():
            lines.append(f"  - {gate}: `{passed}`")
    lines.extend(["", "### Raw pairing gate recomputation", ""])
    for root, gates in evidence.get("strict_raw_recomputed_gates", {}).items():
        lines.append(f"- `{root}`: `{all(gates.values())}`")
        for gate, passed in gates.items():
            lines.append(f"  - {gate}: `{passed}`")
    lines.extend(["", "### Raw execution gate recomputation", ""])
    for root, gates in evidence.get("execution_raw_recomputed_gates", {}).items():
        lines.append(f"- `{root}`: `{all(gates.values())}`")
        for gate, passed in gates.items():
            lines.append(f"  - {gate}: `{passed}`")
    lines.extend(["", "## Verification", ""])
    tests = evidence.get("tests", {})
    lines.append(f"- full pytest passed: `{tests.get('passed', False)}`")
    lines.extend(["- pytest summary:", "", "```text", str(tests.get("summary", "not run")), "```"])
    git = evidence.get("git", {})
    lines.append(f"- git commit: `{git.get('commit', '')}`")
    lines.extend(["- working-tree summary:", "", "```text", str(git.get("status", "")), "```"])
    lines.extend(["", "## Source directory attestations", ""])
    for name, source in evidence.get("source_directories", {}).items():
        lines.append(
            f"- {name}: `{source.get('path')}` / SHA-256 "
            f"`{source.get('sha256')}` / files `{source.get('file_count')}`"
        )
    lines.extend(["", "## Failed gates", ""])
    failures = evidence.get("failed_gates", [])
    lines.extend([f"- {value}" for value in failures] or ["- None"])
    lines.extend(["", "## Recommended actions", ""])
    lines.extend(
        [f"- {value}" for value in evidence.get("recommended_actions", [])]
        or ["- None; all promotion gates passed."]
    )
    lines.extend([
        "",
        "## Limitations",
        "",
        "Seed 42 is a held-out promotion gate only; it does not establish statistical significance.",
        "Calibration server evaluation may be capped for monitoring; it is not an input to prototype, threshold, or budget calibration. Held-out promotion metrics use the complete test set.",
        "Cone budgets did not activate in the held-out clean or attack trajectories, and candidate attack metrics matched the V1 baseline exactly. This gate therefore establishes safe non-regression and controlled offline-cone behavior, not incremental defensive efficacy from cone budgeting.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    candidate_path = (workspace / args.candidate).resolve()
    baseline_manifest_path = (workspace / args.baseline_manifest).resolve()
    validation_path = (workspace / args.calibration_validation).resolve()
    roots = {
        "calibration": (workspace / args.calibration_root).resolve(),
        "clean": (workspace / args.clean).resolve(),
        "candidate_attacks": (workspace / args.candidate_attacks).resolve(),
        "baseline_attacks": (workspace / args.baseline_attacks).resolve(),
    }
    manifest = CalibrationManifest.load(candidate_path)
    baseline_manifest = CalibrationManifest.load(baseline_manifest_path)
    payload = copy.deepcopy(dict(manifest.payload))
    baseline_payload = copy.deepcopy(dict(baseline_manifest.payload))
    failures: list[str] = []
    if payload["schema_version"] != "rtc_v3.calibration.v2":
        failures.append("candidate schema is not rtc_v3.calibration.v2")
    if payload["stable_cones"].get("mode") != "offline_controlled_v1":
        failures.append("candidate cone mode is not offline_controlled_v1")
    baseline_mode = str(
        baseline_payload.get("stable_cones", {}).get("mode", "legacy_online_v1")
    )
    if baseline_payload.get("schema_version") != "rtc_v3.calibration.v1":
        failures.append("baseline manifest schema is not rtc_v3.calibration.v1")
    if baseline_mode != "legacy_online_v1":
        failures.append(f"baseline cone mode is not legacy_online_v1: {baseline_mode}")
    if baseline_manifest.hash == manifest.hash:
        failures.append("baseline manifest is identical to the candidate manifest")
    calibration = json.loads(validation_path.read_text(encoding="utf-8"))
    calibration_gates, calibration_gate_failures = _calibration_validation_audit(
        calibration, payload
    )
    failures.extend(calibration_gate_failures)
    calibration_runtime, calibration_runtime_failures = _calibration_runtime_audit(
        roots["calibration"], payload, calibration
    )
    failures.extend(calibration_runtime_failures)
    calibration_config_paths = sorted(
        (roots["calibration"] / "raw").glob("*calibration_clean*_config.json")
    )
    calibration_reference_config = (
        json.loads(calibration_config_paths[0].read_text(encoding="utf-8"))
        if calibration_config_paths else None
    )
    q_min = float(payload["cumulative"]["resolutions"]["full"]["q_min"])
    q_value = float(payload["direction_persistence"]["resolutions"]["full"]["q_value"])
    if q_min >= 1.0 or q_value >= 1.0:
        failures.append("q_min or q_value is not below one")
    strict: dict[str, dict[str, bool]] = {}
    strict_raw: dict[str, dict[str, bool]] = {}
    execution_raw: dict[str, dict[str, bool]] = {}
    clean: dict[str, Any] = {}
    clean_runtime_contract: dict[str, Any] = {}
    attacks: list[dict[str, Any]] = []
    clean_trial_plan_hashes: set[str] = set()
    if args.calibration_only:
        failures.append(
            "held-out pipeline was not run because formal calibration failed"
        )
    else:
        strict_attack_comparison_ready = True
        for name, root in roots.items():
            if name == "calibration":
                continue
            gates, gate_failures = _execution_gates(root)
            strict[str(root)] = gates
            failures.extend(f"{name}: {value}" for value in gate_failures)
            raw_gates, raw_gate_failures = _raw_pairing_audit(root, gates)
            strict_raw[str(root)] = raw_gates
            failures.extend(f"{name}: {value}" for value in raw_gate_failures)
            raw_execution_gates, raw_execution_failures = _raw_execution_audit(
                root, gates
            )
            execution_raw[str(root)] = raw_execution_gates
            failures.extend(f"{name}: {value}" for value in raw_execution_failures)
            if name in {"candidate_attacks", "baseline_attacks"} and (
                gate_failures or raw_gate_failures or raw_execution_failures
            ):
                strict_attack_comparison_ready = False
        try:
            clean_runtime_contract, clean_contract_failures = (
                _heldout_clean_contract_audit(
                    roots["clean"],
                    reference_config=calibration_reference_config,
                )
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            clean_runtime_contract, clean_contract_failures = {}, [
                f"held-out clean runtime contract unavailable: {exc}"
            ]
        failures.extend(clean_contract_failures)
        try:
            clean, clean_failures = _clean_cone_audit(roots["clean"], payload)
            clean_trial_plan_hashes = {
                str(value)
                for value in _round_frames(roots["clean"])[
                    "trial_plan_hash"
                ].dropna().unique()
            }
        except (FileNotFoundError, KeyError, ValueError) as exc:
            clean, clean_failures = {}, [f"held-out clean audit unavailable: {exc}"]
        failures.extend(clean_failures)
        if strict_attack_comparison_ready:
            try:
                attacks, attack_failures = _attack_audit(
                    roots["candidate_attacks"],
                    roots["baseline_attacks"],
                    reference_config=calibration_reference_config,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                attacks, attack_failures = [], [f"held-out attack audit unavailable: {exc}"]
        else:
            attacks, attack_failures = [], [
                "attack metrics excluded because candidate or baseline strict gates failed"
            ]
        failures.extend(attack_failures)

    tests = {"passed": False, "summary": "not run"}
    if args.run_tests:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        tests = {
            "passed": result.returncode == 0,
            "summary": (result.stdout + result.stderr).strip()[-4000:],
        }
        if result.returncode != 0:
            failures.append("full pytest failed")
    else:
        failures.append("full pytest was not run by the promotion audit")

    candidate_hash = manifest.hash
    promoted_hash = ""
    promoted_file_sha256 = ""
    promotion_semantic_identity = False
    decision = "rejected" if failures else "promoted"
    promoted_path = (workspace / args.promoted).resolve()
    attestation_path = (workspace / args.attestation).resolve()
    rejection_path = (workspace / args.rejection).resolve()
    if not failures:
        promoted = copy.deepcopy(payload)
        promoted["metadata"]["promotion_ready"] = True
        promoted["content_hash"] = content_hash(promoted)
        CalibrationManifest.load(promoted)
        promoted_hash = str(promoted["content_hash"])
        candidate_semantics = copy.deepcopy(payload)
        promoted_semantics = copy.deepcopy(promoted)
        candidate_semantics.pop("content_hash", None)
        promoted_semantics.pop("content_hash", None)
        candidate_semantics["metadata"].pop("promotion_ready", None)
        promoted_semantics["metadata"].pop("promotion_ready", None)
        promotion_semantic_identity = candidate_semantics == promoted_semantics
        if not promotion_semantic_identity:
            raise RuntimeError(
                "promoted manifest differs from candidate beyond promotion metadata"
            )
        _atomic_json(promoted_path, promoted)
        promoted_file_sha256 = _sha256(promoted_path)
        # A previous preflight or incomplete-run rejection is no longer the
        # current decision once every formal gate passes.  Leaving both files
        # behind makes the promotion state ambiguous to automation and humans.
        rejection_path.unlink(missing_ok=True)
    else:
        # Symmetrically, a newly rejected audit must not leave stale success
        # artifacts from an earlier run in place.
        promoted_path.unlink(missing_ok=True)
        attestation_path.unlink(missing_ok=True)
    source_directories = _source_directory_attestations(roots)
    evidence: dict[str, Any] = {
        "schema": "rtc_v3.cone_v2.promotion.v1",
        "schema_version": payload["schema_version"],
        "decision": decision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_hash": candidate_hash,
        "baseline_manifest_path": str(baseline_manifest_path),
        "baseline_manifest_hash": baseline_manifest.hash,
        "baseline_manifest_schema": baseline_payload.get("schema_version"),
        "baseline_cone_mode": baseline_mode,
        "baseline_manifest_sha256": _sha256(baseline_manifest_path),
        "promoted_hash": promoted_hash,
        "promoted_file_sha256": promoted_file_sha256,
        "promotion_semantic_identity": promotion_semantic_identity,
        "calibration_seeds": payload["metadata"]["calibration_seeds"],
        "held_out_seed": 42,
        "q_min": q_min,
        "q_value": q_value,
        "model_metadata_hash": payload["model_metadata_hash"],
        "calibration_validation": calibration,
        "calibration_gates": calibration_gates,
        "calibration_runtime": calibration_runtime,
        "stable_cones": payload["stable_cones"],
        "server_budgets": payload["server_budgets"],
        "principal_betas": payload["principal_betas"],
        "cone_budgets": payload["cone_budgets"],
        "strict_gates": strict,
        "strict_raw_recomputed_gates": strict_raw,
        "execution_raw_recomputed_gates": execution_raw,
        "clean": clean,
        "clean_runtime_contract": clean_runtime_contract,
        "attacks": attacks,
        "failed_gates": failures,
        "recommended_actions": _recommended_actions(failures),
        "tests": tests,
        "trial_plan_hashes": sorted(
            {str(item["trial_plan_hash"]) for item in attacks}
            | clean_trial_plan_hashes
        ),
        "artifact_sha256": _artifact_hashes(
            [candidate_path, baseline_manifest_path, validation_path, *roots.values()]
        ),
        "source_directories": source_directories,
        "git": _git_summary(workspace),
    }
    evidence_path = attestation_path if decision == "promoted" else rejection_path
    _atomic_json(evidence_path, evidence)
    _report((workspace / args.report).resolve(), evidence)
    print(json.dumps({
        "decision": decision,
        "candidate_hash": candidate_hash,
        "promoted_hash": promoted_hash,
        "evidence": str(evidence_path),
        "report": str((workspace / args.report).resolve()),
        "failed_gates": failures,
    }, indent=2, ensure_ascii=False))
    return evidence


def early_rejection(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    """Persist auditable evidence when calibration cannot produce a candidate."""

    workspace = Path(args.workspace).resolve()
    candidate_path = (workspace / args.candidate).resolve()
    validation_path = (workspace / args.calibration_validation).resolve()
    calibration_root = (workspace / args.calibration_root).resolve()
    validation: dict[str, Any] = {}
    if validation_path.is_file():
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            validation = {}
    statuses: list[dict[str, Any]] = []
    for path in sorted((calibration_root / "status").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            statuses.append(
                {
                    "path": str(path.resolve()),
                    "run_id": value.get("run_id"),
                    "state": value.get("state"),
                    "last_round": value.get("last_round"),
                    "exit_code": value.get("exit_code"),
                }
            )
        except (OSError, ValueError, json.JSONDecodeError):
            statuses.append({"path": str(path.resolve()), "state": "unreadable"})
    tests = {"passed": False, "summary": "not run"}
    if args.run_tests:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        tests = {
            "passed": result.returncode == 0,
            "summary": (result.stdout + result.stderr).strip()[-4000:],
        }
    failures = [
        "formal calibration did not produce a valid candidate: "
        f"{type(error).__name__}: {error}"
    ]
    if not tests["passed"]:
        failures.append("full pytest failed" if args.run_tests else "full pytest was not run")
    evidence: dict[str, Any] = {
        "schema": "rtc_v3.cone_v2.promotion.v1",
        "decision": "rejected",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_hash": None,
        "promoted_hash": "",
        "calibration_seeds": [40, 41],
        "held_out_seed": 42,
        "calibration_validation": validation,
        "calibration_runtime": {"statuses": statuses},
        "strict_gates": {},
        "clean": {},
        "attacks": [],
        "failed_gates": failures,
        "recommended_actions": _recommended_actions(failures),
        "tests": tests,
        "trial_plan_hashes": sorted(
            {
                str(value.get("trial_plan_hash"))
                for value in validation.get("source_runs", [])
                if value.get("trial_plan_hash")
            }
        ),
        "artifact_sha256": _artifact_hashes(
            [candidate_path, validation_path, calibration_root]
        ),
        "git": _git_summary(workspace),
    }
    evidence_path = (workspace / args.rejection).resolve()
    (workspace / args.promoted).resolve().unlink(missing_ok=True)
    (workspace / args.attestation).resolve().unlink(missing_ok=True)
    _atomic_json(evidence_path, evidence)
    _report((workspace / args.report).resolve(), evidence)
    print(
        json.dumps(
            {
                "decision": "rejected",
                "candidate_hash": None,
                "evidence": str(evidence_path),
                "report": str((workspace / args.report).resolve()),
                "failed_gates": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return evidence


def calibration_preflight(args: argparse.Namespace) -> list[str]:
    """Validate every calibration gate before any held-out GPU work starts."""

    workspace = Path(args.workspace).resolve()
    candidate_path = (workspace / args.candidate).resolve()
    validation_path = (workspace / args.calibration_validation).resolve()
    calibration_root = (workspace / args.calibration_root).resolve()
    manifest = CalibrationManifest.load(candidate_path)
    payload = manifest.payload
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("schema_version") != "rtc_v3.calibration.v2":
        failures.append("candidate schema is not rtc_v3.calibration.v2")
    if payload.get("stable_cones", {}).get("mode") != "offline_controlled_v1":
        failures.append("candidate cone mode is not offline_controlled_v1")
    _, validation_failures = _calibration_validation_audit(validation, payload)
    failures.extend(validation_failures)
    _, runtime_failures = _calibration_runtime_audit(
        calibration_root, payload, validation
    )
    failures.extend(runtime_failures)
    try:
        q_min = float(payload["cumulative"]["resolutions"]["full"]["q_min"])
        q_value = float(
            payload["direction_persistence"]["resolutions"]["full"]["q_value"]
        )
        if q_min >= 1.0 or q_value >= 1.0:
            failures.append("q_min or q_value is not below one")
    except (KeyError, TypeError, ValueError):
        failures.append("candidate q_min/q_value provenance is invalid")
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--candidate", default="config/rtc_v3_manifest_formal_iid_cone_v2_candidate.json")
    parser.add_argument("--baseline-manifest", default="config/rtc_v3_manifest_formal_iid.json")
    parser.add_argument("--calibration-validation", default="config/rtc_v3_manifest_formal_iid_cone_v2_candidate.validation.json")
    parser.add_argument("--calibration-root", default="logs/rtc_v3_cone_v2_formal_iid_v2")
    parser.add_argument("--clean", default="logs/rtc_v3_cone_v2_heldout_clean")
    parser.add_argument("--candidate-attacks", default="logs/rtc_v3_cone_v2_heldout_attacks")
    parser.add_argument("--baseline-attacks", default="logs/rtc_v3_cone_v2_baseline_attacks")
    parser.add_argument("--promoted", default="config/rtc_v3_manifest_formal_iid_cone_v2.json")
    parser.add_argument("--attestation", default="config/rtc_v3_manifest_formal_iid_cone_v2.promotion.json")
    parser.add_argument("--rejection", default="config/rtc_v3_manifest_formal_iid_cone_v2.rejection.json")
    parser.add_argument("--report", default="docs/rtc_v3_cone_v2_promotion_report.md")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="emit a rejection without held-out audits after calibration failure",
    )
    parser.add_argument(
        "--calibration-preflight",
        action="store_true",
        help="audit calibration gates before launching held-out runs",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        if parsed_args.calibration_preflight:
            preflight_failures = calibration_preflight(parsed_args)
            if preflight_failures:
                parsed_args.calibration_only = True
                audit(parsed_args)
                raise SystemExit(1)
            print("RTC-v3 cone V2 calibration preflight passed.")
        else:
            audit(parsed_args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        if not (parsed_args.calibration_only or parsed_args.calibration_preflight):
            raise
        early_rejection(parsed_args, exc)
        if parsed_args.calibration_preflight:
            raise SystemExit(1)
