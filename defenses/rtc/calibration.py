"""RTC-v3 calibration manifest loading and fail-fast validation.

The manifest is deliberately independent from the RTC-v2 state.  It is a
content-addressed contract: changing any field changes ``content_hash`` and a
runtime mismatch is rejected before aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "rtc_v3.calibration.v1"
SCHEMA_VERSION_V2 = "rtc_v3.calibration.v2"
SCHEMA_VERSION_V3 = "rtc_v3.calibration.v3"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION, SCHEMA_VERSION_V2, SCHEMA_VERSION_V3}
RTC_VERSION = "rtc_v3_candidate"

_REQUIRED_FIELDS = {
    "schema_version",
    "rtc_version",
    "profile",
    "model_metadata_hash",
    "server_optimizer",
    "validated_mass_policy",
    "principal_mapping_version",
    "clip_lower",
    "clip_upper",
    "resolutions",
    "residual_scales",
    "server_windows",
    "server_budgets",
    "principal_windows",
    "principal_betas",
}
_OPTIONAL_FIELDS = {
    "training_phases",
    "stable_cones",
    "cone_budgets",
    "cumulative",
    "direction_persistence",
    "metadata",
    "classifier_head_contract",
    "semantic_temporal_exposure",
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(payload: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "content_hash"}
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _validate_nonnegative_scalar_tree(
    value: Any,
    *,
    field: str,
    strictly_positive: bool,
) -> None:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"RTC-v3 {field} profile mapping cannot be empty")
        for key, nested in value.items():
            _validate_nonnegative_scalar_tree(
                nested,
                field=f"{field}.{key}",
                strictly_positive=strictly_positive,
            )
        return
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0 or (
        strictly_positive and scalar <= 0.0
    ):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"RTC-v3 {field} must be finite and {qualifier}")


def _scalar_profiles(value: Any, *, field: str) -> dict[str, float]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"RTC-v3 {field} profile mapping cannot be empty")
        result: dict[str, float] = {}
        for key, nested in value.items():
            if isinstance(nested, Mapping):
                raise ValueError(
                    f"RTC-v3 {field}.{key} must resolve directly to a scalar"
                )
            result[str(key)] = float(nested)
        return result
    return {"default": float(value)}


def model_metadata(
    params: Sequence[np.ndarray],
    roles: Mapping[int | str, str] | None = None,
) -> list[dict[str, Any]]:
    role_map = {str(key): str(value) for key, value in (roles or {}).items()}
    return [
        {
            "index": index,
            "shape": list(np.asarray(param).shape),
            "dtype": str(np.asarray(param).dtype),
            "floating": bool(np.issubdtype(np.asarray(param).dtype, np.floating)),
            "role": role_map.get(str(index), "unassigned"),
        }
        for index, param in enumerate(params)
    ]


def model_metadata_hash(
    params: Sequence[np.ndarray],
    roles: Mapping[int | str, str] | None = None,
) -> str:
    metadata = model_metadata(params, roles)
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CalibrationManifest:
    """Validated, immutable RTC-v3 calibration contract."""

    payload: Mapping[str, Any]

    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any]) -> "CalibrationManifest":
        if isinstance(source, Mapping):
            payload = dict(source)
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"RTC-v3 calibration manifest not found: {path}")
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        manifest = cls(payload=payload)
        manifest.validate_static()
        return manifest

    @property
    def hash(self) -> str:
        return str(self.payload["content_hash"])

    def validate_static(self) -> None:
        missing = sorted(_REQUIRED_FIELDS.difference(self.payload))
        if missing:
            raise ValueError(f"RTC-v3 calibration manifest missing fields: {missing}")
        unknown = sorted(
            set(self.payload).difference(
                _REQUIRED_FIELDS | _OPTIONAL_FIELDS | {"content_hash"}
            )
        )
        if unknown:
            raise ValueError(
                f"RTC-v3 calibration manifest has unknown fields: {unknown}"
            )
        if self.payload["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                "RTC-v3 calibration schema mismatch: "
                f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}, "
                f"got {self.payload['schema_version']!r}"
            )
        if self.payload["rtc_version"] != RTC_VERSION:
            raise ValueError(
                "RTC-v3 calibration version mismatch: "
                f"expected {RTC_VERSION!r}, got {self.payload['rtc_version']!r}"
            )
        expected_hash = content_hash(self.payload)
        if self.payload.get("content_hash") != expected_hash:
            raise ValueError(
                "RTC-v3 calibration content hash mismatch: "
                f"expected {expected_hash}, got {self.payload.get('content_hash')}"
            )

        lower_profiles = _scalar_profiles(
            self.payload["clip_lower"], field="clip_lower"
        )
        upper_profiles = _scalar_profiles(
            self.payload["clip_upper"], field="clip_upper"
        )
        profile_keys = set(lower_profiles) | set(upper_profiles)
        for profile in profile_keys:
            lower = lower_profiles.get(profile, lower_profiles.get("default"))
            upper = upper_profiles.get(profile, upper_profiles.get("default"))
            if lower is None or upper is None:
                raise ValueError(
                    f"RTC-v3 clip bounds are not paired for profile {profile!r}"
                )
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower < 0.0
                or upper <= 0.0
                or lower > upper
            ):
                raise ValueError(
                    f"RTC-v3 invalid clip bounds for {profile!r}: "
                    f"lower={lower}, upper={upper}"
                )

        resolutions = tuple(str(value) for value in self.payload["resolutions"])
        if not resolutions or "full" not in resolutions or len(set(resolutions)) != len(resolutions):
            raise ValueError("RTC-v3 resolutions must be unique and include 'full'")
        scales = self.payload["residual_scales"]
        if not isinstance(scales, Mapping):
            raise ValueError("RTC-v3 residual_scales must be a mapping")
        for resolution in resolutions:
            if resolution not in scales:
                raise ValueError(
                    f"RTC-v3 residual scale missing for {resolution!r}"
                )
            _validate_nonnegative_scalar_tree(
                scales[resolution],
                field=f"residual_scales.{resolution}",
                strictly_positive=True,
            )

        server_windows = tuple(int(value) for value in self.payload["server_windows"])
        principal_windows = tuple(int(value) for value in self.payload["principal_windows"])
        if any(value <= 0 for value in (*server_windows, *principal_windows)):
            raise ValueError("RTC-v3 windows must contain positive integers")
        if tuple(sorted(set(server_windows))) != server_windows:
            raise ValueError("RTC-v3 server_windows must be sorted and unique")
        if tuple(sorted(set(principal_windows))) != principal_windows:
            raise ValueError("RTC-v3 principal_windows must be sorted and unique")

        server_budgets = self.payload["server_budgets"]
        if not isinstance(server_budgets, Mapping):
            raise ValueError("RTC-v3 server_budgets must be a mapping")
        for resolution in resolutions:
            resolution_budgets = server_budgets.get(resolution)
            if not isinstance(resolution_budgets, Mapping):
                raise ValueError(
                    f"RTC-v3 server budgets missing for {resolution!r}"
                )
            for window in server_windows:
                budget = resolution_budgets.get(str(window))
                if not isinstance(budget, Mapping):
                    raise ValueError(
                        f"RTC-v3 server budget missing for {resolution!r}, W={window}"
                    )
                for exposure_type in ("anchor", "residual", "total"):
                    if exposure_type not in budget:
                        raise ValueError(
                            "RTC-v3 server budget missing "
                            f"{resolution!r}/{window}/{exposure_type}"
                        )
                    _validate_nonnegative_scalar_tree(
                        budget[exposure_type],
                        field=(
                            f"server_budgets.{resolution}.{window}."
                            f"{exposure_type}"
                        ),
                        strictly_positive=False,
                    )

        principal_betas = self.payload["principal_betas"]
        if not isinstance(principal_betas, Mapping):
            raise ValueError("RTC-v3 principal_betas must be a mapping")
        for resolution in resolutions:
            resolution_betas = principal_betas.get(resolution)
            if not isinstance(resolution_betas, Mapping):
                raise ValueError(
                    f"RTC-v3 principal betas missing for {resolution!r}"
                )
            for window in principal_windows:
                if str(window) not in resolution_betas:
                    raise ValueError(
                        f"RTC-v3 principal beta missing for {resolution!r}, M={window}"
                    )
                _validate_nonnegative_scalar_tree(
                    resolution_betas[str(window)],
                    field=f"principal_betas.{resolution}.{window}",
                    strictly_positive=False,
                )

        schedule = self.payload.get("training_phases")
        if schedule is not None:
            if not isinstance(schedule, Sequence) or isinstance(
                schedule, (str, bytes)
            ):
                raise ValueError("RTC-v3 training_phases must be a sequence")
            starts: list[int] = []
            for entry in schedule:
                if not isinstance(entry, Mapping):
                    raise ValueError("RTC-v3 training phase entries must be mappings")
                start = int(entry.get("start_round", 0))
                profile = str(entry.get("profile", ""))
                if start <= 0 or not profile:
                    raise ValueError(
                        "RTC-v3 training phases require positive start_round and profile"
                    )
                starts.append(start)
            if starts != sorted(set(starts)):
                raise ValueError(
                    "RTC-v3 training phase starts must be sorted and unique"
                )

        cones = self.payload.get("stable_cones")
        if isinstance(cones, bool):
            cones = {"enabled": cones}
        if cones is not None:
            if not isinstance(cones, Mapping):
                raise ValueError("RTC-v3 stable_cones must be a mapping or false")
            if bool(cones.get("enabled", False)):
                dimension = int(cones.get("dimension", 256))
                if dimension <= 0:
                    raise ValueError("RTC-v3 stable cone dimension must be positive")
                block_size = int(cones.get("coordinate_median_block_size", 262_144))
                if block_size <= 0:
                    raise ValueError(
                        "RTC-v3 coordinate_median_block_size must be positive"
                    )
                sketch_algorithm = str(
                    cones.get("sketch_algorithm_version", "legacy_blake2b_v1")
                )
                if sketch_algorithm not in {
                    "legacy_blake2b_v1",
                    "splitmix64_v1",
                }:
                    raise ValueError(
                        "RTC-v3 unsupported sketch_algorithm_version: "
                        f"{sketch_algorithm!r}"
                    )
                mode = str(cones.get("mode", "legacy_online_v1"))
                if mode == "offline_controlled_v1":
                    if self.payload["schema_version"] not in {
                        SCHEMA_VERSION_V2,
                        SCHEMA_VERSION_V3,
                    }:
                        raise ValueError(
                            "offline controlled cones require rtc_v3.calibration.v2"
                        )
                    phases = cones.get("phases")
                    if not isinstance(phases, Mapping) or not phases:
                        raise ValueError(
                            "offline controlled cones require phase prototypes"
                        )
                    expected_profiles = {
                        str(item["profile"])
                        for item in (schedule or [])
                    }
                    if expected_profiles and set(phases) != expected_profiles:
                        raise ValueError(
                            "offline cone phases must match training_phases"
                        )
                    for profile, raw_phase in phases.items():
                        if not isinstance(raw_phase, Mapping):
                            raise ValueError(
                                f"stable_cones.phases.{profile} must be a mapping"
                            )
                        match = float(raw_phase.get("match_threshold", float("nan")))
                        update = float(raw_phase.get("update_threshold", float("nan")))
                        drift = float(raw_phase.get("max_angular_drift", float("nan")))
                        if not (
                            np.isfinite(match)
                            and np.isfinite(update)
                            and -1.0 <= match <= update <= 1.0
                        ):
                            raise ValueError(
                                f"invalid offline cone thresholds for {profile!r}"
                            )
                        if not np.isfinite(drift) or not 0.0 <= drift <= np.pi:
                            raise ValueError(
                                f"invalid offline cone drift for {profile!r}"
                            )
                        prototypes = raw_phase.get("prototypes")
                        if not isinstance(prototypes, Sequence) or isinstance(
                            prototypes, (str, bytes)
                        ) or not prototypes:
                            raise ValueError(
                                f"offline cone phase {profile!r} has no prototypes"
                            )
                        if len(prototypes) > int(cones.get("max_prototypes", 16)):
                            raise ValueError(
                                f"offline cone phase {profile!r} exceeds capacity"
                            )
                        ids: set[str] = set()
                        for item in prototypes:
                            if not isinstance(item, Mapping):
                                raise ValueError("offline prototypes must be mappings")
                            cone_id = str(item.get("cone_id", ""))
                            vector = np.asarray(item.get("vector", []), dtype=np.float64)
                            if (
                                not cone_id
                                or cone_id == "overflow"
                                or cone_id in ids
                                or vector.shape != (dimension,)
                                or not np.all(np.isfinite(vector))
                                or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
                            ):
                                raise ValueError(
                                    f"invalid offline prototype in phase {profile!r}"
                                )
                            ids.add(cone_id)
                elif mode != "legacy_online_v1":
                    raise ValueError(f"unsupported stable cone mode {mode!r}")

        if self.payload["schema_version"] == SCHEMA_VERSION_V3:
            self._validate_semantic_v3(schedule=schedule, cones=cones)

    def _validate_semantic_v3(
        self,
        *,
        schedule: Any,
        cones: Any,
    ) -> None:
        from defenses.rtc.semantic import classifier_contract_hash

        contract = self.payload.get("classifier_head_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("RTC-v3 schema V3 requires classifier_head_contract")
        expected_contract_hash = classifier_contract_hash(
            {key: value for key, value in contract.items() if key != "contract_hash"}
        )
        if contract.get("contract_hash") != expected_contract_hash:
            raise ValueError("RTC-v3 classifier-head contract hash mismatch")
        num_classes = int(contract.get("num_classes", 0))
        if int(contract.get("class_axis", -1)) != 0 or num_classes < 2:
            raise ValueError("RTC-v3 classifier-head contract has invalid class rows")
        parameters = contract.get("parameters")
        if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
            raise ValueError("RTC-v3 classifier-head parameters must be a sequence")
        roles = [str(item.get("role", "")) for item in parameters if isinstance(item, Mapping)]
        names = [str(item.get("name", "")) for item in parameters if isinstance(item, Mapping)]
        if roles.count("classifier_weight") != 1 or roles.count("classifier_bias") > 1:
            raise ValueError("RTC-v3 classifier-head roles are ambiguous")
        if len(names) != len(roles) or any(not value for value in names) or len(set(names)) != len(names):
            raise ValueError("RTC-v3 classifier-head parameter names are incomplete")
        semantic_resolutions = contract.get("semantic_resolutions")
        if not isinstance(semantic_resolutions, Mapping):
            raise ValueError("RTC-v3 classifier semantic resolutions are missing")
        weight_rows = semantic_resolutions.get("classifier_weight_rows")
        bias_entries = semantic_resolutions.get("classifier_bias_entries")
        head = semantic_resolutions.get("head")
        if (
            not isinstance(head, Sequence)
            or isinstance(head, (str, bytes))
            or not isinstance(weight_rows, Sequence)
            or isinstance(weight_rows, (str, bytes))
            or len(weight_rows) != num_classes
            or not isinstance(bias_entries, Sequence)
            or isinstance(bias_entries, (str, bytes))
            or len(bias_entries) not in {0, num_classes}
        ):
            raise ValueError("RTC-v3 classifier semantic resolution shape is invalid")
        if [int(item.get("class_index", -1)) for item in weight_rows] != list(
            range(num_classes)
        ) or (
            bias_entries
            and [int(item.get("class_index", -1)) for item in bias_entries]
            != list(range(num_classes))
        ):
            raise ValueError("RTC-v3 classifier row ordering is not canonical")

        semantic = self.payload.get("semantic_temporal_exposure")
        if not isinstance(semantic, Mapping) or not bool(semantic.get("enabled", False)):
            raise ValueError("RTC-v3 schema V3 requires enabled semantic exposure")
        if int(semantic.get("num_classes", 0)) != num_classes:
            raise ValueError("semantic num_classes does not match classifier contract")
        pair_count = num_classes * (num_classes - 1)
        phases = semantic.get("phases")
        if not isinstance(phases, Mapping) or not phases:
            raise ValueError("semantic calibration phases are required")
        expected_profiles = {
            str(item["profile"]) for item in (schedule or [])
        }
        if expected_profiles and set(phases) != expected_profiles:
            raise ValueError("semantic phases must match training_phases")
        for profile, phase in phases.items():
            if not isinstance(phase, Mapping):
                raise ValueError(f"semantic phase {profile!r} must be a mapping")
            row_scales = np.asarray(phase.get("row_scales", []), dtype=np.float64)
            centers = np.asarray(phase.get("pair_centers", []), dtype=np.float64)
            scales = np.asarray(phase.get("pair_scales", []), dtype=np.float64)
            head_scale = float(phase.get("head_exposure_scale", float("nan")))
            if (
                row_scales.shape != (num_classes,)
                or centers.shape != (pair_count,)
                or scales.shape != (pair_count,)
                or not np.all(np.isfinite(row_scales))
                or not np.all(np.isfinite(centers))
                or not np.all(np.isfinite(scales))
                or np.any(row_scales <= 0.0)
                or np.any(scales <= 0.0)
                or not np.isfinite(head_scale)
                or head_scale <= 0.0
            ):
                raise ValueError(f"invalid semantic calibration phase {profile!r}")
        temporal = semantic.get("temporal")
        if not isinstance(temporal, Mapping):
            raise ValueError("semantic temporal config is required")
        # Importing the state object here makes static and runtime validation use
        # the exact same range checks.
        from defenses.rtc.semantic_temporal import SemanticTemporalEvidence

        SemanticTemporalEvidence(temporal, pair_count)
        risk_lambda = float(semantic.get("risk_lambda", float("nan")))
        top_k = int(semantic.get("top_k_pairs", 0))
        if not np.isfinite(risk_lambda) or risk_lambda < 0.0:
            raise ValueError("semantic risk_lambda must be finite and non-negative")
        state_thresholds = semantic.get("state_thresholds")
        if not isinstance(state_thresholds, Mapping):
            raise ValueError("semantic state_thresholds are required")
        watch = float(state_thresholds.get("watch", float("nan")))
        restricted = float(state_thresholds.get("restricted", float("nan")))
        quarantined = float(state_thresholds.get("quarantined", float("nan")))
        if not (
            np.isfinite(watch)
            and np.isfinite(restricted)
            and np.isfinite(quarantined)
            and 0.0 <= watch < restricted < quarantined <= 1.0
        ):
            raise ValueError("semantic state thresholds must be ordered within [0,1]")
        hard_exposure_risk_floor = float(
            semantic.get("hard_exposure_risk_floor", float("nan"))
        )
        if not (
            np.isfinite(hard_exposure_risk_floor)
            and 0.0 <= hard_exposure_risk_floor <= watch
        ):
            raise ValueError(
                "semantic hard_exposure_risk_floor must be within [0, watch]"
            )
        if not 1 <= top_k < pair_count:
            raise ValueError("semantic top_k_pairs must be within the pair set")
        performance = semantic.get("performance_budget")
        if not isinstance(performance, Mapping):
            raise ValueError("semantic performance_budget is required")
        performance_limits = {
            "semantic_mean_seconds": 0.15,
            "semantic_p95_seconds": 0.25,
            "defense_mean_increase_fraction": 0.05,
            "defense_p95_increase_fraction": 0.10,
        }
        for key, upper in performance_limits.items():
            value = float(performance.get(key, float("nan")))
            if not np.isfinite(value) or value < 0.0 or value > upper:
                raise ValueError(
                    f"semantic performance budget {key} must be within [0, {upper}]"
                )
        for prefix in ("global", "principal", "pair"):
            windows = tuple(int(value) for value in semantic.get(f"{prefix}_windows", ()))
            budgets = semantic.get(f"{prefix}_budgets")
            if not windows or windows != tuple(sorted(set(windows))) or 1 not in windows:
                raise ValueError(f"semantic {prefix}_windows must be sorted and include 1")
            if not isinstance(budgets, Mapping):
                raise ValueError(f"semantic {prefix}_budgets must be a mapping")
            for window in windows:
                if str(window) not in budgets:
                    raise ValueError(f"semantic {prefix} budget missing window {window}")
                _validate_nonnegative_scalar_tree(
                    budgets[str(window)],
                    field=f"semantic_temporal_exposure.{prefix}_budgets.{window}",
                    strictly_positive=False,
                )
        if not isinstance(cones, Mapping) or bool(
            cones.get("online_updates_enabled", True)
        ):
            raise ValueError("RTC-v3 semantic candidate requires frozen online cones")

    def validate_runtime(
        self,
        *,
        model_hash: str,
        server_optimizer: str,
        validated_mass_policy: str,
        principal_mapping_version: str,
    ) -> None:
        checks = {
            "model_metadata_hash": model_hash,
            "server_optimizer": str(server_optimizer).lower(),
            "validated_mass_policy": validated_mass_policy,
            "principal_mapping_version": principal_mapping_version,
        }
        for field, actual in checks.items():
            expected = str(self.payload[field])
            if expected != str(actual):
                raise ValueError(
                    f"RTC-v3 calibration runtime mismatch for {field}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        if str(server_optimizer).lower() != "fedavg":
            raise ValueError(
                "RTC-v3 candidate currently supports only FedAvg server update semantics"
            )


def build_manifest(
    *,
    params: Sequence[np.ndarray],
    profile: str = "test",
    roles: Mapping[int | str, str] | None = None,
    server_optimizer: str = "fedavg",
    validated_mass_policy: str = "unit_principal",
    principal_mapping_version: str = "v1",
    clip_lower: float | Mapping[str, float] = 0.0,
    clip_upper: float | Mapping[str, float] = 10.0,
    resolutions: Sequence[str] = ("full",),
    residual_scales: Mapping[str, float] | None = None,
    server_windows: Sequence[int] = (1,),
    server_budgets: Mapping[str, Any] | None = None,
    principal_windows: Sequence[int] = (1,),
    principal_betas: Mapping[str, Any] | None = None,
    schema_version: str = SCHEMA_VERSION,
    **extra: Any,
) -> dict[str, Any]:
    """Build a content-addressed manifest for tests and calibration tooling."""

    resolution_values = tuple(str(value) for value in resolutions)
    scales = dict(residual_scales or {value: 1.0 for value in resolution_values})
    budgets = dict(
        server_budgets
        or {
            value: {
                str(window): {"anchor": 1e9, "residual": 1e9, "total": 1e9}
                for window in server_windows
            }
            for value in resolution_values
        }
    )
    betas = dict(
        principal_betas
        or {
            value: {str(window): 1e9 for window in principal_windows}
            for value in resolution_values
        }
    )
    payload: dict[str, Any] = {
        "schema_version": str(schema_version),
        "rtc_version": RTC_VERSION,
        "profile": str(profile),
        "model_metadata_hash": model_metadata_hash(params, roles),
        "server_optimizer": str(server_optimizer).lower(),
        "validated_mass_policy": str(validated_mass_policy),
        "principal_mapping_version": str(principal_mapping_version),
        "clip_lower": (
            dict(clip_lower) if isinstance(clip_lower, Mapping) else float(clip_lower)
        ),
        "clip_upper": (
            dict(clip_upper) if isinstance(clip_upper, Mapping) else float(clip_upper)
        ),
        "resolutions": list(resolution_values),
        "residual_scales": scales,
        "server_windows": [int(value) for value in server_windows],
        "server_budgets": budgets,
        "principal_windows": [int(value) for value in principal_windows],
        "principal_betas": betas,
        **extra,
    }
    payload["content_hash"] = content_hash(payload)
    return payload
