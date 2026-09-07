"""Promoted RTC-v3 semantic-temporal-exposure defense.

The implementation remains state- and manifest-isolated from RTC-v2. Phase
components are enabled through one invariant-checked pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList
from defenses.rtc.calibration import (
    CalibrationManifest,
    model_metadata_hash,
)
from defenses.rtc.cumulative import (
    CumulativeTransition,
    OneSidedCumulativeEvidence,
)
from defenses.rtc.direction import (
    DirectionPersistenceEvidence,
    DirectionTransition,
)
from defenses.rtc.server_ledger import ServerExposureLedger
from defenses.rtc.solver import LinearBudget, max_violation, solve_subprobability_qp
from defenses.rtc.prototypes import Prototype, PrototypeTransition, StablePrototypeBank
from defenses.rtc.principal_ledger import (
    PrincipalExposureLedger,
    PrincipalParticipation,
)
from defenses.rtc.resolutions import ResolutionLayout, squared_norm
from defenses.rtc.semantic import (
    ClassifierHeadLayout,
    SemanticBatch,
    extract_semantic_batch,
    gate_semantic_interventions,
    hard_exposure_coefficients,
    select_top_pair_indices,
)
from defenses.rtc.semantic_temporal import (
    SemanticTemporalEvidence,
    SemanticTemporalTransition,
)
from defenses.rtc.sketch import (
    LEGACY_BLAKE2B_V1,
    signed_count_sketch,
    sketch_cache_stats,
)
from defenses.rtc.weighted_stats import (
    weighted_coordinate_median,
    weighted_median_and_mad,
)

logger = logging.getLogger(__name__)

_ALLOWED_CUSTOM_PARAMS = {
    "calibration_manifest",
    "calibration_path",
    "implementation_phase",
    "validated_mass_policy",
    "principal_mapping_version",
    "server_optimizer",
    "parameter_roles",
    "principal_map",
    "principal_first_sampling_verified",
    "client_mass_caps",
    "principal_mass_caps",
    "coordinate_median_block_size",
    "anchor_recycle_fraction",
    "anchor_recycle_weighting",
    "sketch_algorithm_version",
    "export_sketches",
    "semantic_ablation",
    "semantic_intervention_risk_floor",
    "cumulative_q_cap_power",
    "norm_clip_mad_k",
    "residual_rank_cap_top_k",
    "residual_rank_cap_factor",
    "residual_rank_recycle_fraction",
}


@dataclass(frozen=True)
class RTCv3ClientRecord:
    client_id: str
    principal_id: str
    validated_mass: float
    nominal_mass: float
    aggregation_weight: float
    norm: float
    clipped: bool
    residual_norm: float
    state: str = "normal"
    magnitude_risk: float = 0.0
    direction_risk: float = 0.0
    temporal_risk: float = 0.0
    influence_risk: float = 0.0
    total_risk: float = 0.0
    event_risk: float = 0.0


class RTCv3Defense(BaseDefense):
    """Manifest-bound RTC-v3 implementation.

    Phase 0 establishes the manifest contract; Phases 1–6 are enabled
    explicitly by ``implementation_phase`` while sharing one fixed pipeline.
    """

    version = "rtc_v3"

    def __init__(self, cfg: DefenseConfig, num_clients: int = 100):
        super().__init__(cfg)
        params = dict(cfg.custom_params or {})
        unknown_params = sorted(set(params).difference(_ALLOWED_CUSTOM_PARAMS))
        if unknown_params:
            raise ValueError(
                f"RTC-v3 has unknown custom parameters: {unknown_params}"
            )
        source = params.get("calibration_manifest")
        if source is None:
            source = params.get("calibration_path")
        if source is None:
            raise ValueError(
                "rtc_v3 requires calibration_manifest or calibration_path"
            )
        if isinstance(source, (str, Path)):
            self.manifest = CalibrationManifest.load(source)
        elif isinstance(source, Mapping):
            self.manifest = CalibrationManifest.load(source)
        else:
            raise TypeError(
                "RTC-v3 calibration source must be a mapping or filesystem path"
            )

        self.params = params
        self.num_clients = int(num_clients)
        self.implementation_phase = int(params.get("implementation_phase", 0))
        if self.implementation_phase < 0 or self.implementation_phase > 6:
            raise ValueError("RTC-v3 implementation_phase must be between 0 and 6")

        self.validated_mass_policy = str(
            params.get(
                "validated_mass_policy",
                self.manifest.payload["validated_mass_policy"],
            )
        )
        self.principal_mapping_version = str(
            params.get(
                "principal_mapping_version",
                self.manifest.payload["principal_mapping_version"],
            )
        )
        self.server_optimizer = str(
            params.get("server_optimizer", self.manifest.payload["server_optimizer"])
        ).lower()
        self.parameter_roles = {
            str(key): str(value)
            for key, value in dict(params.get("parameter_roles", {})).items()
        }
        self.principal_map = {
            str(key): str(value)
            for key, value in dict(params.get("principal_map", {})).items()
        }
        self.requires_principal_first_sampling = self.implementation_phase >= 4
        if self.requires_principal_first_sampling:
            if not self.principal_map:
                raise ValueError(
                    "RTC-v3 Phase 4+ requires an explicit stable principal_map"
                )
            if not bool(params.get("principal_first_sampling_verified", False)):
                raise ValueError(
                    "RTC-v3 Phase 4+ requires principal_first_sampling_verified=true"
                )
            if 1 not in {
                int(value) for value in self.manifest.payload["principal_windows"]
            }:
                raise ValueError("RTC-v3 principal windows must include M=1")
        self._principal_ids: list[str] = []
        self._validated_masses: list[float] | None = None
        self._runtime_validated = False
        self._server_ledger = ServerExposureLedger()
        self._principal_ledger = PrincipalExposureLedger()
        cumulative_config = self.manifest.payload.get(
            "cumulative", {"enabled": False}
        )
        if not isinstance(cumulative_config, Mapping):
            raise ValueError("RTC-v3 cumulative config must be a mapping")
        self._cumulative = OneSidedCumulativeEvidence(cumulative_config)
        if self._cumulative.enabled and self.implementation_phase < 5:
            raise ValueError(
                "RTC-v3 cumulative evidence can only be enabled in Phase 5+"
            )
        raw_cumulative_q_cap_power = float(
            params.get("cumulative_q_cap_power", 0)
        )
        if (
            not np.isfinite(raw_cumulative_q_cap_power)
            or raw_cumulative_q_cap_power not in {0.0, 1.0, 2.0}
        ):
            raise ValueError("cumulative_q_cap_power must be 0, 1, or 2")
        self.cumulative_q_cap_power = int(raw_cumulative_q_cap_power)
        if self.cumulative_q_cap_power and not self._cumulative.enabled:
            raise ValueError(
                "cumulative_q_cap_power requires cumulative evidence"
            )
        self.norm_clip_mad_k = float(params.get("norm_clip_mad_k", 2.5))
        if not np.isfinite(self.norm_clip_mad_k) or self.norm_clip_mad_k <= 0.0:
            raise ValueError("norm_clip_mad_k must be finite and positive")
        raw_residual_rank_cap_top_k = float(
            params.get("residual_rank_cap_top_k", 0)
        )
        if (
            not np.isfinite(raw_residual_rank_cap_top_k)
            or raw_residual_rank_cap_top_k < 0.0
            or not raw_residual_rank_cap_top_k.is_integer()
        ):
            raise ValueError(
                "residual_rank_cap_top_k must be a finite non-negative integer"
            )
        self.residual_rank_cap_top_k = int(raw_residual_rank_cap_top_k)
        self.residual_rank_cap_factor = float(
            params.get("residual_rank_cap_factor", 1.0)
        )
        if (
            not np.isfinite(self.residual_rank_cap_factor)
            or not 0.0 <= self.residual_rank_cap_factor <= 1.0
        ):
            raise ValueError(
                "residual_rank_cap_factor must be finite and in [0, 1]"
            )
        self.residual_rank_recycle_fraction = float(
            params.get("residual_rank_recycle_fraction", 0.0)
        )
        if (
            not np.isfinite(self.residual_rank_recycle_fraction)
            or not 0.0 <= self.residual_rank_recycle_fraction <= 1.0
        ):
            raise ValueError(
                "residual_rank_recycle_fraction must be finite and in [0, 1]"
            )
        if self.residual_rank_cap_top_k > 0:
            if self.residual_rank_cap_factor >= 1.0:
                raise ValueError(
                    "residual_rank_cap_factor must be below 1 when rank cap is enabled"
                )
            if self.residual_rank_recycle_fraction > 0.0 and str(
                params.get("anchor_recycle_weighting", "nominal")
            ).lower() != "accepted":
                raise ValueError(
                    "residual rank recycle requires anchor_recycle_weighting='accepted'"
                )
        elif self.residual_rank_recycle_fraction > 0.0:
            raise ValueError(
                "residual_rank_recycle_fraction requires residual rank cap"
            )
        cone_config = self.manifest.payload.get("stable_cones", {})
        if isinstance(cone_config, bool):
            cone_config = {"enabled": cone_config}
        if not isinstance(cone_config, Mapping):
            raise ValueError("RTC-v3 stable_cones must be a mapping or boolean")
        self._cones_enabled = bool(cone_config.get("enabled", False))
        self._cone_config = dict(cone_config)
        self._cone_mode = str(
            self._cone_config.get("mode", "legacy_online_v1")
        )
        self._controlled_cones = self._cone_mode == "offline_controlled_v1"
        self._cone_online_updates_enabled = bool(
            self._cone_config.get("online_updates_enabled", True)
        )
        self._export_sketches = bool(params.get("export_sketches", False))
        self.coordinate_median_block_size = int(
            params.get(
                "coordinate_median_block_size",
                self._cone_config.get("coordinate_median_block_size", 262_144),
            )
        )
        if self.coordinate_median_block_size <= 0:
            raise ValueError("coordinate_median_block_size must be positive")
        self.anchor_recycle_fraction = float(
            params.get("anchor_recycle_fraction", 0.0)
        )
        if (
            not np.isfinite(self.anchor_recycle_fraction)
            or not 0.0 <= self.anchor_recycle_fraction <= 1.0
        ):
            raise ValueError("anchor_recycle_fraction must be finite and in [0, 1]")
        self.anchor_recycle_weighting = str(
            params.get("anchor_recycle_weighting", "nominal")
        ).lower()
        if self.anchor_recycle_weighting not in {"nominal", "accepted"}:
            raise ValueError(
                "anchor_recycle_weighting must be 'nominal' or 'accepted'"
            )
        manifest_sketch_algorithm = str(
            self._cone_config.get("sketch_algorithm_version", LEGACY_BLAKE2B_V1)
        )
        configured_sketch_algorithm = str(
            params.get("sketch_algorithm_version", manifest_sketch_algorithm)
        )
        if configured_sketch_algorithm != manifest_sketch_algorithm:
            raise ValueError(
                "RTC-v3 sketch algorithm does not match calibration manifest: "
                f"expected {manifest_sketch_algorithm!r}, "
                f"got {configured_sketch_algorithm!r}"
            )
        self.sketch_algorithm_version = manifest_sketch_algorithm
        direction_config = self.manifest.payload.get(
            "direction_persistence", {"enabled": False}
        )
        if not isinstance(direction_config, Mapping):
            raise ValueError("RTC-v3 direction_persistence must be a mapping")
        self._direction = DirectionPersistenceEvidence(direction_config)
        if self._direction.enabled:
            if self.implementation_phase < 6:
                raise ValueError(
                    "RTC-v3 direction persistence can only be enabled in Phase 6"
                )
            if not self._cones_enabled:
                raise ValueError(
                    "RTC-v3 direction persistence requires stable_cones"
                )
        semantic_config = self.manifest.payload.get(
            "semantic_temporal_exposure", {"enabled": False}
        )
        if not isinstance(semantic_config, Mapping):
            raise ValueError("RTC-v3 semantic config must be a mapping")
        self._semantic_config = dict(semantic_config)
        self._semantic_enabled = bool(self._semantic_config.get("enabled", False))
        self._semantic_state_thresholds = dict(
            self._semantic_config.get(
                "state_thresholds",
                {"watch": 0.10, "restricted": 0.50, "quarantined": 0.80},
            )
        )
        self._semantic_ablation = str(
            params.get("semantic_ablation", "exposure")
        ).lower()
        if self._semantic_ablation not in {
            "observe",
            "soft",
            "temporal",
            "exposure",
        }:
            raise ValueError(
                "semantic_ablation must be observe, soft, temporal, or exposure"
            )
        if not self._semantic_enabled and self._semantic_ablation != "exposure":
            raise ValueError("semantic ablations require a schema V3 semantic manifest")
        self._semantic_intervention_risk_floor = float(
            params.get("semantic_intervention_risk_floor", 0.0)
        )
        if (
            not np.isfinite(self._semantic_intervention_risk_floor)
            or not 0.0 <= self._semantic_intervention_risk_floor <= 1.0
        ):
            raise ValueError(
                "semantic_intervention_risk_floor must be finite and in [0, 1]"
            )
        if (
            not self._semantic_enabled
            and self._semantic_intervention_risk_floor > 0.0
        ):
            raise ValueError(
                "semantic intervention gating requires a schema V3 semantic manifest"
            )
        self._semantic_hard_exposure_risk_floor = max(
            float(self._semantic_config.get("hard_exposure_risk_floor", 0.0)),
            self._semantic_intervention_risk_floor,
        )
        self.requires_parameter_roles = self._semantic_enabled
        self._semantic_layout: ClassifierHeadLayout | None = None
        self._semantic_temporal: SemanticTemporalEvidence | None = None
        if self._semantic_enabled:
            num_classes = int(self._semantic_config.get("num_classes", 0))
            self._semantic_temporal = SemanticTemporalEvidence(
                self._semantic_config["temporal"],
                num_classes * (num_classes - 1),
            )
        self._prototype_banks: dict[str, StablePrototypeBank] = {}
        self._active_cone_profile: str | None = None
        self._layout: ResolutionLayout | None = None
        self._last_records: list[RTCv3ClientRecord] = []
        self._last_clipped_mask: list[bool] = []
        self._last_constraint_tags: list[str] = []
        self._last_cone_assignments: dict[str, tuple[str, ...]] = {}
        self._last_cone_similarities: dict[str, tuple[float, ...]] = {}
        self._last_sketches: dict[str, tuple[np.ndarray, ...]] = {}
        self._last_cone_update_eligible: dict[str, tuple[bool, ...]] = {}
        self._last_cumulative_q: dict[str, tuple[float, ...]] = {}
        self._last_direction_q: dict[str, tuple[float, ...]] = {}
        self._last_semantic_batch: SemanticBatch | None = None
        self._last_semantic_risks: tuple[float, ...] = ()
        self._last_semantic_q: tuple[float, ...] = ()
        self._last_client_q_cap: tuple[float, ...] = ()
        self._last_clip_norm = float("inf")
        self.last_round_metrics = {
            "time_consistency_version": self.version,
            "rtc_v3_phase": float(self.implementation_phase),
            "rtc_v3_calibration_hash": self.manifest.hash,
        }

    def set_context(
        self,
        server_round: int,
        client_ids: Sequence[str],
        global_params: Sequence[np.ndarray],
        *,
        principal_ids: Sequence[str] | None = None,
        validated_masses: Sequence[float] | None = None,
        server_optimizer: str | None = None,
        parameter_roles: Mapping[int | str, str] | None = None,
        parameter_names: Mapping[int | str, str] | None = None,
        **_: Any,
    ) -> None:
        self._server_round = int(server_round)
        self._client_ids = [str(value) for value in client_ids]
        # Read-only views: RTC-v3 never mutates the arrays held by the strategy.
        self._global_params = [np.asarray(value) for value in global_params]
        if principal_ids is None:
            self._principal_ids = [
                self.principal_map.get(client_id, client_id)
                for client_id in self._client_ids
            ]
        else:
            if len(principal_ids) != len(self._client_ids):
                raise ValueError("principal_ids length must match client_ids")
            self._principal_ids = [str(value) for value in principal_ids]
        if validated_masses is None:
            self._validated_masses = None
        else:
            if len(validated_masses) != len(self._client_ids):
                raise ValueError("validated_masses length must match client_ids")
            masses = np.asarray(validated_masses, dtype=np.float64)
            if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
                raise ValueError("validated_masses must be finite and non-negative")
            self._validated_masses = [float(value) for value in masses]

        roles = dict(self.parameter_roles)
        if parameter_roles is not None:
            roles.update({str(key): str(value) for key, value in parameter_roles.items()})
        actual_optimizer = str(server_optimizer or self.server_optimizer).lower()
        self.manifest.validate_runtime(
            model_hash=model_metadata_hash(self._global_params, roles),
            server_optimizer=actual_optimizer,
            validated_mass_policy=self.validated_mass_policy,
            principal_mapping_version=self.principal_mapping_version,
        )
        if self._semantic_enabled:
            self._semantic_layout = ClassifierHeadLayout.build(
                self._global_params, roles
            )
            contract = self.manifest.payload.get("classifier_head_contract")
            if not isinstance(contract, Mapping):
                raise ValueError("semantic RTC-v3 manifest has no head contract")
            self._semantic_layout.validate_contract(
                self._global_params, contract, parameter_names
            )
        enabled_resolutions = (
            ("full",)
            if self.implementation_phase <= 2
            else tuple(str(value) for value in self.manifest.payload["resolutions"])
        )
        self._layout = ResolutionLayout.build(
            self._global_params,
            roles,
            enabled_resolutions,
        )
        if self._cones_enabled and self.implementation_phase >= 3:
            profile = self._profile_for_round(self._server_round)
            if self._controlled_cones and profile != self._active_cone_profile:
                self._prototype_banks = {
                    resolution: self._new_prototype_bank(profile)
                    for resolution in enabled_resolutions
                }
                self._active_cone_profile = profile
            for resolution in enabled_resolutions:
                self._prototype_banks.setdefault(
                    resolution,
                    self._new_prototype_bank(profile),
                )
        self.server_optimizer = actual_optimizer
        self._runtime_validated = True

    def aggregate(self, updates: UpdateList) -> list[np.ndarray]:
        if not self._runtime_validated or self._global_params is None:
            raise RuntimeError(
                "RTC-v3 requires a validated runtime context before aggregation"
            )
        if self.implementation_phase == 0:
            raise RuntimeError(
                "RTC-v3 Phase 0 is a contract-only scaffold; "
                "set implementation_phase>=1 to enable aggregation"
            )
        return self._aggregate_phase1_to_6(updates)

    def principal_id_for(self, client_id: str) -> str:
        client = str(client_id)
        return self.principal_map.get(client, client)

    def _new_prototype_bank(self, profile: str | None = None) -> StablePrototypeBank:
        if self._controlled_cones:
            phases = self._cone_config.get("phases", {})
            raw_phase = phases.get(str(profile)) if isinstance(phases, Mapping) else None
            if not isinstance(raw_phase, Mapping):
                raise ValueError(
                    f"RTC-v3 has no offline cone prototypes for profile {profile!r}"
                )
            raw_prototypes = raw_phase.get("prototypes", ())
            prototypes = [
                Prototype(
                    str(item["cone_id"]),
                    np.asarray(item["vector"], dtype=np.float64),
                    0,
                )
                for item in raw_prototypes
            ]
            return StablePrototypeBank(
                max_prototypes=int(self._cone_config.get("max_prototypes", 16)),
                match_threshold=float(raw_phase["match_threshold"]),
                update_threshold=float(raw_phase["update_threshold"]),
                momentum=float(self._cone_config.get("momentum", 0.98)),
                retirement_rounds=1,
                initial_prototypes=prototypes,
                controlled=True,
                max_angular_drift=float(raw_phase["max_angular_drift"]),
                min_update_principals=int(
                    self._cone_config.get("min_update_principals", 2)
                ),
            )
        return StablePrototypeBank(
            max_prototypes=int(self._cone_config.get("max_prototypes", 16)),
            match_threshold=float(
                self._cone_config.get("match_threshold", 0.85)
            ),
            momentum=float(self._cone_config.get("momentum", 0.8)),
            retirement_rounds=int(
                self._cone_config.get("retirement_rounds", 16)
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return versioned RTC-v3 state suitable for a server checkpoint."""

        return {
            "schema": (
                "rtc_v3.state.v3"
                if self._semantic_enabled
                else ("rtc_v3.state.v2" if self._controlled_cones else "rtc_v3.state.v1")
            ),
            "rtc_version": self.version,
            "implementation_phase": self.implementation_phase,
            "calibration_hash": self.manifest.hash,
            "principal_mapping_version": self.principal_mapping_version,
            "active_cone_profile": self._active_cone_profile,
            "server_ledger": self._server_ledger.snapshot(),
            "principal_ledger": self._principal_ledger.snapshot(),
            "prototype_banks": {
                resolution: bank.state_dict()
                for resolution, bank in self._prototype_banks.items()
            },
            "cumulative": self._cumulative.state_dict(),
            "direction": self._direction.state_dict(),
            "semantic_temporal": (
                self._semantic_temporal.state_dict()
                if self._semantic_temporal is not None
                else {}
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate a complete checkpoint before replacing live RTC-v3 state."""

        expected = {
            "schema": (
                "rtc_v3.state.v3"
                if self._semantic_enabled
                else ("rtc_v3.state.v2" if self._controlled_cones else "rtc_v3.state.v1")
            ),
            "rtc_version": self.version,
            "implementation_phase": self.implementation_phase,
            "calibration_hash": self.manifest.hash,
            "principal_mapping_version": self.principal_mapping_version,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"RTC-v3 checkpoint mismatch for {key}: "
                    f"expected {value!r}, got {state.get(key)!r}"
                )

        server_ledger = ServerExposureLedger()
        server_ledger.load_snapshot(state.get("server_ledger", {}))
        principal_ledger = PrincipalExposureLedger()
        principal_ledger.load_snapshot(state.get("principal_ledger", {}))
        active_profile = state.get("active_cone_profile")
        if self._controlled_cones and not isinstance(active_profile, str):
            raise ValueError("controlled cone checkpoint is missing its active phase")
        raw_banks = state.get("prototype_banks", {})
        if not isinstance(raw_banks, Mapping):
            raise ValueError("RTC-v3 prototype checkpoint must be a mapping")
        prototype_banks: dict[str, StablePrototypeBank] = {}
        for resolution, bank_state in raw_banks.items():
            if str(resolution) not in {
                str(value) for value in self.manifest.payload["resolutions"]
            }:
                raise ValueError(
                    f"RTC-v3 checkpoint has unknown resolution {resolution!r}"
                )
            if not isinstance(bank_state, Mapping):
                raise ValueError("RTC-v3 prototype bank state must be a mapping")
            bank = self._new_prototype_bank(
                str(active_profile) if active_profile is not None else None
            )
            bank.load_state_dict(bank_state)
            prototype_banks[str(resolution)] = bank
        cumulative = OneSidedCumulativeEvidence(self._cumulative.config)
        cumulative.load_state_dict(state.get("cumulative", {}))
        direction = DirectionPersistenceEvidence(self._direction.config)
        direction.load_state_dict(state.get("direction", {}))
        semantic_temporal = None
        if self._semantic_temporal is not None:
            semantic_temporal = SemanticTemporalEvidence(
                self._semantic_temporal.config,
                self._semantic_temporal.pair_count,
            )
            semantic_temporal.load_state_dict(state.get("semantic_temporal", {}))

        self._server_ledger = server_ledger
        self._principal_ledger = principal_ledger
        self._prototype_banks = prototype_banks
        self._active_cone_profile = (
            str(active_profile) if active_profile is not None else None
        )
        self._cumulative = cumulative
        self._direction = direction
        if semantic_temporal is not None:
            self._semantic_temporal = semantic_temporal

    def _aggregate_phase1_to_6(self, updates: UpdateList) -> list[np.ndarray]:
        started = time.perf_counter()
        self._validate_updates(updates)
        validated = self._validated_mass(updates)
        total_validated = float(validated.sum(dtype=np.float64))
        if total_validated <= 0.0:
            return self._commit_zero_round(
                updates, validated, "zero_validated_mass", started
            )
        nominal = validated / total_validated

        scoring_started = time.perf_counter()
        floating_indices = [
            index
            for index, parameter in enumerate(self._global_params or [])
            if np.issubdtype(parameter.dtype, np.floating)
        ]
        if not floating_indices:
            raise ValueError("RTC-v3 requires at least one floating model parameter")
        deltas: list[list[np.ndarray]] = []
        norms = np.empty(len(updates), dtype=np.float64)
        for client_index, (parameters, _) in enumerate(updates):
            client_deltas: list[np.ndarray] = []
            client_squared_norm = 0.0
            for parameter_index in floating_indices:
                delta = (
                    np.asarray(parameters[parameter_index], dtype=np.float64)
                    - np.asarray(
                        (self._global_params or [])[parameter_index],
                        dtype=np.float64,
                    )
                )
                client_deltas.append(delta)
                client_squared_norm += float(
                    np.dot(delta.reshape(-1), delta.reshape(-1))
                )
            deltas.append(client_deltas)
            norms[client_index] = np.sqrt(client_squared_norm)

        positive_count = int(np.count_nonzero(nominal > 0.0))
        positive = nominal > 0.0
        median, mad = weighted_median_and_mad(
            norms[positive],
            nominal[positive],
            principal_ids=[
                principal
                for principal, keep in zip(self._principal_ids, positive)
                if keep
            ],
            client_ids=[
                client for client, keep in zip(self._client_ids, positive) if keep
            ],
        )
        current_profile = self._profile_for_round(self._server_round)
        clip_lower = self._profile_scalar(
            self.manifest.payload["clip_lower"], profile=current_profile
        )
        clip_upper = self._profile_scalar(
            self.manifest.payload["clip_upper"], profile=current_profile
        )
        if positive_count < 3 or mad <= np.finfo(np.float64).eps:
            clip_norm = clip_lower
        else:
            clip_norm = median + self.norm_clip_mad_k * 1.4826 * mad
        clip_norm = float(np.clip(clip_norm, clip_lower, clip_upper))
        factors = np.ones(len(updates), dtype=np.float64)
        nonzero_norm = norms > 0.0
        factors[nonzero_norm] = np.minimum(
            1.0, clip_norm / (norms[nonzero_norm] + np.finfo(np.float64).eps)
        )
        clipped = [
            [delta * factors[client_index] for delta in client_deltas]
            for client_index, client_deltas in enumerate(deltas)
        ]
        clipped_mask = factors < 1.0 - 1e-15
        scoring_seconds = time.perf_counter() - scoring_started

        anchor_started = time.perf_counter()
        anchors: list[np.ndarray] = []
        positive_client_ids = [
            client for client, keep in zip(self._client_ids, positive) if keep
        ]
        positive_principal_ids = [
            principal
            for principal, keep in zip(self._principal_ids, positive)
            if keep
        ]
        for tensor_index in range(len(floating_indices)):
            anchors.append(
                weighted_coordinate_median(
                    [
                        client_deltas[tensor_index]
                        for client_deltas, keep in zip(clipped, positive)
                        if keep
                    ],
                    nominal[positive],
                    principal_ids=positive_principal_ids,
                    client_ids=positive_client_ids,
                    block_size=self.coordinate_median_block_size,
                )
            )
        anchor_seconds = time.perf_counter() - anchor_started

        residual_started = time.perf_counter()
        if self._layout is None:
            raise RuntimeError("RTC-v3 resolution metadata was not validated")
        enabled_resolutions = tuple(self._layout.indices)
        positions = {
            resolution: self._layout.tensor_positions(
                resolution, floating_indices
            )
            for resolution in enabled_resolutions
        }
        residual_arrays = [
            [
                delta - anchor
                for delta, anchor in zip(client_deltas, anchors)
            ]
            for client_deltas in clipped
        ]
        gamma_anchors: dict[str, float] = {}
        gamma_residuals: dict[str, np.ndarray] = {}
        for resolution in enabled_resolutions:
            scale = self._profile_scalar(
                self.manifest.payload["residual_scales"][resolution],
                profile=current_profile,
            )
            gamma_anchors[resolution] = (
                np.sqrt(squared_norm(anchors, positions[resolution])) / scale
            )
            gamma_residuals[resolution] = np.asarray(
                [
                    np.sqrt(
                        squared_norm(client_residuals, positions[resolution])
                    )
                    / scale
                    for client_residuals in residual_arrays
                ],
                dtype=np.float64,
            )
        residual_seconds = time.perf_counter() - residual_started

        semantic_started = time.perf_counter()
        semantic_extraction_seconds = 0.0
        semantic_temporal_seconds = 0.0
        semantic_batch: SemanticBatch | None = None
        semantic_transition: SemanticTemporalTransition | None = None
        semantic_risks = np.zeros(len(updates), dtype=np.float64)
        semantic_q = np.ones(len(updates), dtype=np.float64)
        semantic_optimization_risks = np.zeros(len(updates), dtype=np.float64)
        semantic_optimization_q = np.ones(len(updates), dtype=np.float64)
        semantic_intervention_active = np.zeros(len(updates), dtype=bool)
        semantic_coefficients = np.zeros(len(updates), dtype=np.float64)
        if self._semantic_enabled:
            if self._semantic_layout is None or self._semantic_temporal is None:
                raise RuntimeError("semantic RTC-v3 runtime contract was not initialized")
            phase_configs = self._semantic_config.get("phases")
            phase_config = (
                phase_configs.get(current_profile)
                if isinstance(phase_configs, Mapping)
                else None
            )
            if not isinstance(phase_config, Mapping):
                raise ValueError(
                    f"semantic RTC-v3 has no calibration for {current_profile!r}"
                )
            extraction_started = time.perf_counter()
            semantic_batch = extract_semantic_batch(
                layout=self._semantic_layout,
                residual_arrays=residual_arrays,
                floating_indices=floating_indices,
                phase_config=phase_config,
            )
            semantic_extraction_seconds = time.perf_counter() - extraction_started
            temporal_started = time.perf_counter()
            semantic_transition = self._semantic_temporal.prepare(
                server_round=self._server_round,
                principal_ids=self._principal_ids,
                z_values=semantic_batch.z_values,
                top_pair_indices=semantic_batch.top_pair_indices,
                top_pair_signatures=semantic_batch.top_pair_signatures,
                client_ids=self._client_ids,
            )
            semantic_temporal_seconds = time.perf_counter() - temporal_started
            semantic_risks = semantic_transition.client_risks
            semantic_q = semantic_transition.client_q
            (
                semantic_optimization_risks,
                semantic_optimization_q,
                semantic_intervention_active,
            ) = gate_semantic_interventions(
                semantic_risks,
                semantic_q,
                self._semantic_intervention_risk_floor,
            )
            semantic_coefficients = hard_exposure_coefficients(
                semantic_risks,
                semantic_batch.normalized_head_norms,
                self._semantic_hard_exposure_risk_floor,
            )
        semantic_seconds = time.perf_counter() - semantic_started

        cumulative_transition = CumulativeTransition(observations={})
        if self.implementation_phase >= 5 and (
            self._cumulative.enabled or self._direction.enabled
        ):
            cumulative_transition = self._cumulative.prepare(
                principal_ids=self._principal_ids,
                nominal_masses=nominal,
                raw_residuals={
                    resolution: (
                        gamma_residuals[resolution]
                        * self._profile_scalar(
                            self.manifest.payload["residual_scales"][resolution],
                            profile=current_profile,
                        )
                    )
                    for resolution in enabled_resolutions
                },
                force=self._direction.enabled,
            )

        sketch_cache_before = sketch_cache_stats()
        sketch_started = time.perf_counter()
        cone_transitions: dict[str, PrototypeTransition] = {}
        cone_assignments: dict[str, tuple[str, ...]] = {}
        cone_similarities: dict[str, tuple[float, ...]] = {}
        round_sketches: dict[str, tuple[np.ndarray, ...]] = {}
        if self.implementation_phase >= 3 and self._cones_enabled:
            dimension = int(self._cone_config.get("dimension", 256))
            seed = int(self._cone_config.get("seed", 0))
            for resolution in enabled_resolutions:
                sketches = [
                    signed_count_sketch(
                        [
                            client_residuals[position]
                            for position in positions[resolution]
                        ],
                        dimension=dimension,
                        seed=seed,
                        algorithm=self.sketch_algorithm_version,
                    )
                    for client_residuals in residual_arrays
                ]
                transition = self._prototype_banks[resolution].prepare(
                    server_round=self._server_round,
                    sketches=sketches,
                    nominal_masses=nominal,
                    client_ids=self._client_ids,
                    principal_ids=self._principal_ids,
                )
                cone_transitions[resolution] = transition
                cone_assignments[resolution] = transition.assignments
                cone_similarities[resolution] = transition.similarities
                round_sketches[resolution] = tuple(
                    np.asarray(value, dtype=np.float32) for value in sketches
                )
        direction_transition = DirectionTransition(
            observations={}, next_history={}
        )
        if self.implementation_phase >= 6 and self._direction.enabled:
            direction_transition = self._direction.prepare(
                principal_ids=self._principal_ids,
                nominal_masses=nominal,
                cone_assignments=cone_assignments,
                z_values={
                    key: observation.z
                    for key, observation in cumulative_transition.observations.items()
                },
            )
        sketch_seconds = time.perf_counter() - sketch_started
        sketch_cache_after = sketch_cache_stats()

        ledger_started = time.perf_counter()
        enabled_windows = (
            (1,)
            if self.implementation_phase == 1
            else tuple(int(value) for value in self.manifest.payload["server_windows"])
        )
        if 1 not in enabled_windows:
            raise ValueError("RTC-v3 server windows must include W=1")
        coefficients = {
            resolution: {
                "anchor": np.full(
                    len(updates), gamma_anchors[resolution], dtype=np.float64
                ),
                "residual": gamma_residuals[resolution].copy(),
                "total": (
                    gamma_residuals[resolution] + gamma_anchors[resolution]
                ),
            }
            for resolution in enabled_resolutions
        }
        histories: dict[tuple[str, str, int], float] = {}
        budget_values: dict[tuple[str, str, int], float] = {}
        budgets: list[LinearBudget] = []
        verification_budgets: list[LinearBudget] = []
        for resolution in enabled_resolutions:
            for window in enabled_windows:
                window_spec = self.manifest.payload["server_budgets"][
                    resolution
                ].get(str(window))
                if not isinstance(window_spec, Mapping):
                    raise ValueError(
                        f"RTC-v3 missing {resolution!r} budget for W={window}"
                    )
                for exposure_type in ("anchor", "residual", "total"):
                    history = self._server_ledger.history(
                        resolution,
                        exposure_type,
                        self._server_round,
                        window,
                    )
                    active_budget = self._active_window_scalar(
                        window_spec[exposure_type], window
                    )
                    histories[(resolution, exposure_type, window)] = history
                    budget_values[
                        (resolution, exposure_type, window)
                    ] = active_budget
                    remaining = max(0.0, active_budget - history)
                    budgets.append(
                        LinearBudget(
                            f"{resolution}:{exposure_type}:W{window}",
                            coefficients[resolution][exposure_type][positive],
                            remaining,
                        )
                    )
                    verification_budgets.append(
                        LinearBudget(
                            f"{resolution}:{exposure_type}:W{window}",
                            coefficients[resolution][exposure_type],
                            remaining,
                        )
                    )

        cone_coefficients: dict[tuple[str, str], np.ndarray] = {}
        if cone_assignments:
            cone_budget_manifest = self.manifest.payload.get("cone_budgets")
            if not isinstance(cone_budget_manifest, Mapping):
                raise ValueError(
                    "enabled stable cones require calibrated cone_budgets"
                )
            for resolution, assignments in cone_assignments.items():
                resolution_spec = cone_budget_manifest.get(resolution)
                if not isinstance(resolution_spec, Mapping):
                    raise ValueError(
                        f"missing cone budgets for resolution {resolution!r}"
                    )
                for cone_id in sorted(set(assignments)):
                    mask = np.asarray(
                        [assigned == cone_id for assigned in assignments],
                        dtype=np.float64,
                    )
                    cone_coefficient = gamma_residuals[resolution] * mask
                    cone_coefficients[(resolution, cone_id)] = cone_coefficient
                    exposure_key = f"cone:{cone_id}"
                    for window in enabled_windows:
                        raw_spec = resolution_spec.get(str(window))
                        if raw_spec is None:
                            raise ValueError(
                                f"missing cone budget for {resolution!r}, W={window}"
                            )
                        if isinstance(raw_spec, Mapping) and "residual" in raw_spec:
                            raw_spec = raw_spec["residual"]
                        cone_budget = min(
                            self._active_window_scalar(raw_spec, window),
                            budget_values[(resolution, "residual", window)],
                        )
                        history = self._server_ledger.history(
                            resolution,
                            exposure_key,
                            self._server_round,
                            window,
                        )
                        histories[(resolution, exposure_key, window)] = history
                        budget_values[
                            (resolution, exposure_key, window)
                        ] = cone_budget
                        remaining = max(0.0, cone_budget - history)
                        budgets.append(
                            LinearBudget(
                                f"{resolution}:{exposure_key}:W{window}",
                                cone_coefficient[positive],
                                remaining,
                            )
                        )
                        verification_budgets.append(
                            LinearBudget(
                                f"{resolution}:{exposure_key}:W{window}",
                                cone_coefficient,
                                remaining,
                            )
                        )

        semantic_group_coefficients: dict[str, np.ndarray] = {}
        semantic_group_seconds = 0.0
        semantic_ledger_seconds = 0.0
        if semantic_batch is not None:
            semantic_group_started = time.perf_counter()
            global_windows = tuple(
                int(value) for value in self._semantic_config["global_windows"]
            )
            for window in global_windows:
                semantic_history_started = time.perf_counter()
                history = self._server_ledger.history(
                    "semantic", "risk", self._server_round, window
                )
                semantic_ledger_seconds += time.perf_counter() - semantic_history_started
                active_budget = self._profile_scalar(
                    self._semantic_config["global_budgets"][str(window)],
                    profile=current_profile,
                )
                histories[("semantic", "risk", window)] = history
                budget_values[("semantic", "risk", window)] = active_budget
                remaining = max(0.0, active_budget - history)
                if self._semantic_ablation == "exposure":
                    budgets.append(
                        LinearBudget(
                            f"semantic:risk:W{window}",
                            semantic_coefficients[positive],
                            remaining,
                        )
                    )
                    verification_budgets.append(
                        LinearBudget(
                            f"semantic:risk:W{window}",
                            semantic_coefficients,
                            remaining,
                        )
                    )
            selected_pairs = set(
                select_top_pair_indices(
                    client_pair_indices=semantic_batch.top_pair_indices,
                    exposure_coefficients=semantic_coefficients,
                    nominal_masses=nominal,
                    pairs=semantic_batch.pairs,
                    top_k=int(self._semantic_config["top_k_pairs"]),
                )
            )
            for pair_index in sorted(selected_pairs):
                source, target = semantic_batch.pairs[pair_index]
                key = f"pair:{source}->{target}"
                semantic_group_coefficients[key] = semantic_coefficients * (
                    semantic_batch.top_pair_indices == pair_index
                )
            semantic_group_coefficients["other"] = semantic_coefficients * np.asarray(
                [int(value) not in selected_pairs for value in semantic_batch.top_pair_indices],
                dtype=np.float64,
            )
            for key, coefficient in semantic_group_coefficients.items():
                exposure_key = f"semantic:{key}"
                for window in tuple(
                    int(value) for value in self._semantic_config["pair_windows"]
                ):
                    semantic_history_started = time.perf_counter()
                    history = self._server_ledger.history(
                        "semantic", exposure_key, self._server_round, window
                    )
                    semantic_ledger_seconds += time.perf_counter() - semantic_history_started
                    active_budget = self._profile_scalar(
                        self._semantic_config["pair_budgets"][str(window)],
                        profile=current_profile,
                    )
                    histories[("semantic", exposure_key, window)] = history
                    budget_values[("semantic", exposure_key, window)] = active_budget
                    remaining = max(0.0, active_budget - history)
                    if self._semantic_ablation == "exposure":
                        budgets.append(
                            LinearBudget(
                                f"semantic:{key}:W{window}",
                                coefficient[positive],
                                remaining,
                            )
                        )
                        verification_budgets.append(
                            LinearBudget(
                                f"semantic:{key}:W{window}", coefficient, remaining
                            )
                        )
            semantic_group_seconds += (
                time.perf_counter() - semantic_group_started - semantic_ledger_seconds
            )

        principal_diagnostics: dict[
            tuple[str, str, int], tuple[float, float, float]
        ] = {}
        direction_q_values: dict[tuple[str, str], float] = {}
        principal_groups = {
            principal: np.asarray(
                [value == principal for value in self._principal_ids], dtype=bool
            )
            for principal in sorted(set(self._principal_ids))
        }
        if semantic_batch is not None:
            semantic_principal_group_started = time.perf_counter()
            semantic_principal_ledger_before = semantic_ledger_seconds
            for principal, group_mask in principal_groups.items():
                coefficient = semantic_coefficients * group_mask.astype(np.float64)
                for window in tuple(
                    int(value) for value in self._semantic_config["principal_windows"]
                ):
                    semantic_history_started = time.perf_counter()
                    history = self._principal_ledger.history(
                        principal, "semantic", window
                    )
                    semantic_ledger_seconds += time.perf_counter() - semantic_history_started
                    active_budget = self._profile_scalar(
                        self._semantic_config["principal_budgets"][str(window)],
                        profile=current_profile,
                    )
                    remaining = max(0.0, active_budget - history)
                    principal_diagnostics[(principal, "semantic", window)] = (
                        history,
                        active_budget,
                        float(nominal[group_mask].sum(dtype=np.float64)),
                    )
                    if self._semantic_ablation == "exposure":
                        budgets.append(
                            LinearBudget(
                                f"principal:{principal}:semantic:M{window}",
                                coefficient[positive],
                                remaining,
                            )
                        )
                        verification_budgets.append(
                            LinearBudget(
                                f"principal:{principal}:semantic:M{window}",
                                coefficient,
                                remaining,
                            )
                        )
            semantic_group_seconds += max(
                0.0,
                time.perf_counter()
                - semantic_principal_group_started
                - (semantic_ledger_seconds - semantic_principal_ledger_before),
            )
        if self.implementation_phase >= 4:
            principal_windows = tuple(
                int(value) for value in self.manifest.payload["principal_windows"]
            )
            for principal, group_mask in principal_groups.items():
                principal_mass = float(nominal[group_mask].sum(dtype=np.float64))
                for resolution in enabled_resolutions:
                    beta_spec = self.manifest.payload["principal_betas"].get(
                        resolution
                    )
                    if not isinstance(beta_spec, Mapping):
                        raise ValueError(
                            f"missing principal beta for {resolution!r}"
                        )
                    principal_coefficient = (
                        gamma_residuals[resolution] * group_mask.astype(np.float64)
                    )
                    cumulative_q = 1.0
                    observation = cumulative_transition.observations.get(
                        (principal, resolution)
                    )
                    if observation is not None and self._cumulative.enabled:
                        cumulative_q = observation.q
                    direction_q = 1.0
                    if self._direction.enabled:
                        server_window = self._direction.server_window(resolution)
                        principal_window = self._direction.principal_window(
                            resolution
                        )
                        if server_window not in enabled_windows:
                            raise ValueError(
                                "direction server_window must be an enabled "
                                "server window"
                            )
                        if principal_window not in principal_windows:
                            raise ValueError(
                                "direction principal_window must be an enabled "
                                "principal window"
                            )
                        server_budget = budget_values[
                            (resolution, "residual", server_window)
                        ]
                        server_nominal_use = float(
                            np.dot(
                                gamma_residuals[resolution],
                                nominal,
                            )
                        )
                        server_used = (
                            histories[
                                (resolution, "residual", server_window)
                            ]
                            + server_nominal_use
                        )
                        server_usage = (
                            server_used / server_budget
                            if server_budget > 0.0
                            else float("inf")
                        )
                        direction_history = self._principal_ledger.history(
                            principal, resolution, principal_window
                        )
                        direction_mass = (
                            self._principal_ledger.nominal_mass_with_current(
                                principal, principal_window, principal_mass
                            )
                        )
                        direction_profiles = (
                            self._principal_ledger.profiles_with_current(
                                principal, principal_window, current_profile
                            )
                        )
                        direction_beta = min(
                            self._profile_scalar(
                                beta_spec[str(principal_window)],
                                profile=profile,
                            )
                            for profile in direction_profiles
                        )
                        direction_budget = direction_beta * direction_mass
                        principal_nominal_use = float(
                            np.dot(principal_coefficient, nominal)
                        )
                        principal_used = (
                            direction_history + principal_nominal_use
                        )
                        principal_usage = (
                            principal_used / direction_budget
                            if direction_budget > 0.0
                            else float("inf")
                        )
                        direction_q = self._direction.q(
                            direction_transition,
                            principal_id=principal,
                            resolution=resolution,
                            server_usage=server_usage,
                            principal_usage=principal_usage,
                        )
                        direction_q_values[(principal, resolution)] = direction_q
                    for window in principal_windows:
                        if str(window) not in beta_spec:
                            raise ValueError(
                                f"missing principal beta for {resolution!r}, M={window}"
                            )
                        history = self._principal_ledger.history(
                            principal, resolution, window
                        )
                        mass_window = (
                            self._principal_ledger.nominal_mass_with_current(
                                principal, window, principal_mass
                            )
                        )
                        profiles = (
                            self._principal_ledger.profiles_with_current(
                                principal, window, current_profile
                            )
                        )
                        beta = min(
                            self._profile_scalar(
                                beta_spec[str(window)], profile=profile
                            )
                            for profile in profiles
                        )
                        q = min(1.0, cumulative_q, direction_q)
                        active_budget = q * beta * mass_window
                        remaining = max(0.0, active_budget - history)
                        principal_diagnostics[
                            (principal, resolution, window)
                        ] = (history, active_budget, principal_mass)
                        budgets.append(
                            LinearBudget(
                                f"principal:{principal}:{resolution}:M{window}",
                                principal_coefficient[positive],
                                remaining,
                            )
                        )
                        verification_budgets.append(
                            LinearBudget(
                                f"principal:{principal}:{resolution}:M{window}",
                                principal_coefficient,
                                remaining,
                            )
                        )
        ledger_seconds = time.perf_counter() - ledger_started

        semantic_client_q = (
            semantic_optimization_q
            if self._semantic_ablation in {"temporal", "exposure"}
            else np.ones(len(updates), dtype=np.float64)
        )
        client_q_cap = np.asarray(semantic_client_q, dtype=np.float64).copy()
        if self.cumulative_q_cap_power:
            if "full" not in enabled_resolutions:
                raise ValueError(
                    "cumulative q client cap requires the full resolution"
                )
            cumulative_full = np.asarray(
                [
                    float(
                        cumulative_transition.observations.get(
                            (principal, "full")
                        ).q
                    )
                    if cumulative_transition.observations.get(
                        (principal, "full")
                    ) is not None
                    else 1.0
                    for principal in self._principal_ids
                ],
                dtype=np.float64,
            )
            direction_full = np.asarray(
                [
                    float(direction_q_values.get((principal, "full"), 1.0))
                    for principal in self._principal_ids
                ],
                dtype=np.float64,
            )
            client_q_cap = np.minimum.reduce(
                (
                    client_q_cap,
                    np.power(cumulative_full, self.cumulative_q_cap_power),
                    direction_full,
                )
            )
        reference_client_q_cap = client_q_cap.copy()
        residual_rank_cap_multiplier = np.ones(len(updates), dtype=np.float64)
        residual_rank_cap_indices: tuple[int, ...] = ()
        if self.residual_rank_cap_top_k > 0:
            ranked_positive = sorted(
                (int(index) for index in np.flatnonzero(positive)),
                key=lambda index: (
                    -float(gamma_residuals["full"][index]),
                    str(self._principal_ids[index]),
                    str(self._client_ids[index]),
                    index,
                ),
            )
            residual_rank_cap_indices = tuple(
                ranked_positive[: self.residual_rank_cap_top_k]
            )
            residual_rank_cap_multiplier[
                list(residual_rank_cap_indices)
            ] = self.residual_rank_cap_factor
            client_q_cap = np.minimum(
                client_q_cap, residual_rank_cap_multiplier
            )
        reference_upper_bounds = nominal * reference_client_q_cap
        upper_bounds = nominal * client_q_cap

        solver_started = time.perf_counter()
        solver_risk = (
            semantic_optimization_risks[positive]
            if self._semantic_ablation in {"soft", "temporal", "exposure"}
            else np.zeros(int(np.count_nonzero(positive)), dtype=np.float64)
        )
        solver_risk_lambda = (
            float(self._semantic_config.get("risk_lambda", 0.0))
            if self._semantic_ablation in {"soft", "temporal", "exposure"}
            else 0.0
        )
        reference_solution = None
        residual_rank_reference_valid = True
        if self.residual_rank_cap_top_k > 0:
            reference_solution = solve_subprobability_qp(
                nominal[positive],
                reference_upper_bounds[positive],
                budgets,
                risk=solver_risk,
                risk_lambda=solver_risk_lambda,
                tolerance=1e-8,
                ftol=1e-10,
                maxiter=100,
            )
            residual_rank_reference_valid = (
                max_violation(
                    reference_solution.weights,
                    reference_upper_bounds[positive],
                    budgets,
                )
                <= 1e-8
            )
        solution = solve_subprobability_qp(
            nominal[positive],
            upper_bounds[positive],
            budgets,
            risk=solver_risk,
            risk_lambda=solver_risk_lambda,
            tolerance=1e-8,
            ftol=1e-10,
            maxiter=100,
        )
        weights = np.zeros(len(updates), dtype=np.float64)
        weights[positive] = solution.weights
        verified_violation = max_violation(
            weights,
            upper_bounds,
            verification_budgets,
        )
        if verified_violation > 1e-8:
            weights.fill(0.0)
            solver_status = "zero_fallback"
            verified_violation = max_violation(
                weights,
                upper_bounds,
                verification_budgets,
            )
        else:
            solver_status = solution.status
        solver_seconds = time.perf_counter() - solver_started

        weight_sum = float(weights.sum(dtype=np.float64))
        reference_weight_sum = weight_sum
        if reference_solution is not None and residual_rank_reference_valid:
            reference_weight_sum = float(
                np.asarray(reference_solution.weights, dtype=np.float64).sum(
                    dtype=np.float64
                )
            )
        residual_rank_removed_mass = (
            max(0.0, reference_weight_sum - weight_sum)
            if residual_rank_reference_valid
            else 0.0
        )
        anchor_recycle_base_target_mass = self.anchor_recycle_fraction * max(
            0.0, 1.0 - reference_weight_sum
        )
        residual_rank_recycle_target_mass = (
            self.residual_rank_recycle_fraction * residual_rank_removed_mass
        )
        anchor_recycle_target_mass = min(
            max(0.0, 1.0 - weight_sum),
            anchor_recycle_base_target_mass
            + residual_rank_recycle_target_mass,
        )
        anchor_recycle_mass = 0.0
        recycle_anchors = anchors
        recycle_gamma_anchors = gamma_anchors
        anchor_recycle_source_available = True
        if (
            anchor_recycle_target_mass > np.finfo(np.float64).eps
            and self.anchor_recycle_weighting == "accepted"
        ):
            anchor_recycle_source_available = (
                weight_sum > np.finfo(np.float64).eps
            )
            if anchor_recycle_source_available:
                accepted_anchor_started = time.perf_counter()
                recycle_anchors = []
                for tensor_index in range(len(floating_indices)):
                    recycle_anchors.append(
                        weighted_coordinate_median(
                            [
                                client_deltas[tensor_index]
                                for client_deltas in clipped
                            ],
                            weights,
                            principal_ids=self._principal_ids,
                            client_ids=self._client_ids,
                            block_size=self.coordinate_median_block_size,
                        )
                    )
                recycle_gamma_anchors = {}
                for resolution in enabled_resolutions:
                    scale = self._profile_scalar(
                        self.manifest.payload["residual_scales"][resolution],
                        profile=current_profile,
                    )
                    recycle_gamma_anchors[resolution] = (
                        np.sqrt(
                            squared_norm(
                                recycle_anchors, positions[resolution]
                            )
                        )
                        / scale
                    )
                anchor_seconds += time.perf_counter() - accepted_anchor_started
        if (
            anchor_recycle_target_mass > np.finfo(np.float64).eps
            and not str(solver_status).startswith("zero_")
            and anchor_recycle_source_available
        ):
            recycle_limits = [anchor_recycle_target_mass]
            for resolution in enabled_resolutions:
                anchor_coefficient = float(recycle_gamma_anchors[resolution])
                if anchor_coefficient <= np.finfo(np.float64).eps:
                    continue
                for exposure_type in ("anchor", "total"):
                    current_exposure = float(
                        np.dot(coefficients[resolution][exposure_type], weights)
                    )
                    for window in enabled_windows:
                        remaining = (
                            float(budget_values[(resolution, exposure_type, window)])
                            - float(histories[(resolution, exposure_type, window)])
                            - current_exposure
                        )
                        recycle_limits.append(
                            max(0.0, remaining) / anchor_coefficient
                        )
            anchor_recycle_mass = max(0.0, min(recycle_limits))
            # The recycled coordinate median is server-owned: it has anchor
            # exposure but no client/principal residual or semantic exposure.
            # Recheck the affected hard budgets after adding that exposure.
            for resolution in enabled_resolutions:
                anchor_coefficient = float(recycle_gamma_anchors[resolution])
                recycled_exposure = anchor_coefficient * anchor_recycle_mass
                for exposure_type in ("anchor", "total"):
                    current_exposure = float(
                        np.dot(coefficients[resolution][exposure_type], weights)
                    )
                    for window in enabled_windows:
                        verified_violation = max(
                            verified_violation,
                            float(histories[(resolution, exposure_type, window)])
                            + current_exposure
                            + recycled_exposure
                            - float(
                                budget_values[(resolution, exposure_type, window)]
                            ),
                        )

        aggregation_started = time.perf_counter()
        result: list[np.ndarray] = []
        floating_position = 0
        for parameter_index, global_parameter in enumerate(self._global_params or []):
            if np.issubdtype(global_parameter.dtype, np.floating):
                aggregate_delta = np.zeros(
                    global_parameter.shape, dtype=np.float64
                )
                for weight, client_deltas in zip(weights, clipped):
                    aggregate_delta += weight * client_deltas[floating_position]
                if anchor_recycle_mass > 0.0:
                    aggregate_delta += (
                        anchor_recycle_mass * recycle_anchors[floating_position]
                    )
                value = (
                    np.asarray(global_parameter, dtype=np.float64) + aggregate_delta
                )
                if not np.all(np.isfinite(value)):
                    weights.fill(0.0)
                    anchor_recycle_mass = 0.0
                    anchor_recycle_target_mass = 0.0
                    anchor_recycle_base_target_mass = 0.0
                    residual_rank_recycle_target_mass = 0.0
                    residual_rank_removed_mass = 0.0
                    result = [
                        np.asarray(parameter).copy()
                        for parameter in self._global_params or []
                    ]
                    solver_status = "zero_nonfinite_aggregate"
                    break
                result.append(value.astype(global_parameter.dtype, copy=False))
                floating_position += 1
            else:
                result.append(np.asarray(global_parameter).copy())
        aggregation_seconds = time.perf_counter() - aggregation_started

        exposures: dict[str, dict[str, float]] = {
            resolution: {
                exposure_type: float(np.dot(coefficient, weights))
                for exposure_type, coefficient in typed_coefficients.items()
            }
            for resolution, typed_coefficients in coefficients.items()
        }
        for resolution in enabled_resolutions:
            recycled_anchor_exposure = (
                float(recycle_gamma_anchors[resolution]) * anchor_recycle_mass
            )
            exposures[resolution]["anchor"] += recycled_anchor_exposure
            exposures[resolution]["total"] += recycled_anchor_exposure
        for (resolution, cone_id), coefficient in cone_coefficients.items():
            exposures[resolution][f"cone:{cone_id}"] = float(
                np.dot(coefficient, weights)
            )
        committed_exposures = {
            resolution: dict(values) for resolution, values in exposures.items()
        }
        if semantic_batch is not None:
            committed_exposures["semantic"] = {
                "risk": float(np.dot(semantic_coefficients, weights)),
                **{
                    f"semantic:{key}": float(np.dot(coefficient, weights))
                    for key, coefficient in semantic_group_coefficients.items()
                },
            }
        self._server_ledger.commit(self._server_round, committed_exposures)
        principal_current_exposures: dict[tuple[str, str], float] = {}
        if self.implementation_phase >= 4 or semantic_batch is not None:
            principal_events: dict[str, PrincipalParticipation] = {}
            for principal, group_mask in principal_groups.items():
                principal_exposures = {
                    resolution: float(
                        np.dot(
                            gamma_residuals[resolution]
                            * group_mask.astype(np.float64),
                            weights,
                        )
                    )
                    for resolution in enabled_resolutions
                }
                if semantic_batch is not None:
                    principal_exposures["semantic"] = float(
                        np.dot(
                            semantic_coefficients * group_mask.astype(np.float64),
                            weights,
                        )
                    )
                principal_current_exposures.update(
                    {
                        (principal, resolution): value
                        for resolution, value in principal_exposures.items()
                    }
                )
                principal_events[principal] = PrincipalParticipation(
                    server_round=self._server_round,
                    nominal_mass=float(
                        nominal[group_mask].sum(dtype=np.float64)
                    ),
                    profile=current_profile,
                    exposures=principal_exposures,
                    solver_failed=solver_status in {
                        "zero_fallback",
                        "zero_nonfinite_aggregate",
                    },
                )
            self._principal_ledger.commit(principal_events)
            if self._cumulative.enabled:
                self._cumulative.commit(cumulative_transition)
            if self._direction.enabled:
                self._direction.commit(direction_transition)
        if semantic_transition is not None and self._semantic_temporal is not None:
            self._semantic_temporal.commit(semantic_transition)
        cone_update_eligible: dict[str, tuple[bool, ...]] = {}
        cumulative_q_by_resolution: dict[str, tuple[float, ...]] = {}
        direction_q_by_resolution: dict[str, tuple[float, ...]] = {}
        for resolution, transition in cone_transitions.items():
            cumulative_values = tuple(
                float(
                    cumulative_transition.observations.get(
                        (principal, resolution)
                    ).q
                )
                if cumulative_transition.observations.get(
                    (principal, resolution)
                ) is not None
                else 1.0
                for principal in self._principal_ids
            )
            direction_values = tuple(
                float(direction_q_values.get((principal, resolution), 1.0))
                for principal in self._principal_ids
            )
            eligible = np.asarray(
                [
                    (not bool(was_clipped))
                    and float(weight) >= float(mass) - 1e-10
                    and (
                        cumulative_transition.observations.get(
                            (principal, resolution)
                        ) is None
                        or cumulative_transition.observations[
                            (principal, resolution)
                        ].q >= 1.0 - 1e-12
                    )
                    and direction_q_values.get(
                        (principal, resolution), 1.0
                    ) >= 1.0 - 1e-12
                    for principal, was_clipped, weight, mass in zip(
                        self._principal_ids, clipped_mask, weights, nominal
                    )
                ],
                dtype=bool,
            )
            cone_update_eligible[resolution] = tuple(bool(value) for value in eligible)
            cumulative_q_by_resolution[resolution] = cumulative_values
            direction_q_by_resolution[resolution] = direction_values
            self._prototype_banks[resolution].commit(
                transition,
                eligible if self._cone_online_updates_enabled else np.zeros_like(eligible),
            )
        self._last_cone_assignments = dict(cone_assignments)
        self._last_cone_similarities = dict(cone_similarities)
        self._last_sketches = dict(round_sketches) if self._export_sketches else {}
        self._last_cone_update_eligible = cone_update_eligible
        self._last_cumulative_q = cumulative_q_by_resolution
        self._last_direction_q = direction_q_by_resolution
        self._last_semantic_batch = semantic_batch
        self._last_semantic_risks = tuple(float(value) for value in semantic_risks)
        self._last_semantic_q = tuple(float(value) for value in semantic_q)
        self._last_client_q_cap = tuple(float(value) for value in client_q_cap)
        self._record_round(
            nominal=nominal,
            validated=validated,
            weights=weights,
            norms=norms,
            clipped_mask=clipped_mask,
            residual_norms=gamma_residuals["full"]
            * self._profile_scalar(
                self.manifest.payload["residual_scales"]["full"],
                profile=current_profile,
            ),
            clip_norm=clip_norm,
            solver_status=solver_status,
            max_constraint_violation=verified_violation,
            anchor_recycle_mass=anchor_recycle_mass,
            anchor_recycle_target_mass=anchor_recycle_target_mass,
            anchor_recycle_base_target_mass=anchor_recycle_base_target_mass,
            residual_rank_cap_multiplier=residual_rank_cap_multiplier,
            residual_rank_reference_weight_sum=reference_weight_sum,
            residual_rank_reference_valid=residual_rank_reference_valid,
            residual_rank_removed_mass=residual_rank_removed_mass,
            residual_rank_recycle_target_mass=residual_rank_recycle_target_mass,
            exposures=exposures,
            histories=histories,
            budgets=budget_values,
            server_windows=enabled_windows,
            cone_assignments=cone_assignments,
            cone_transitions=cone_transitions,
            principal_diagnostics=principal_diagnostics,
            principal_current_exposures=principal_current_exposures,
            cumulative_transition=cumulative_transition,
            direction_transition=direction_transition,
            direction_q_values=direction_q_values,
            semantic_batch=semantic_batch,
            semantic_transition=semantic_transition,
            semantic_risks=semantic_risks,
            semantic_q=semantic_q,
            client_q_cap=client_q_cap,
            semantic_intervention_active=semantic_intervention_active,
            semantic_coefficients=semantic_coefficients,
            timings={
                "scoring_seconds": scoring_seconds,
                "anchor_seconds": anchor_seconds,
                "residual_seconds": residual_seconds,
                "semantic_seconds": semantic_seconds,
                "semantic_extraction_seconds": semantic_extraction_seconds,
                "semantic_temporal_seconds": semantic_temporal_seconds,
                "semantic_group_seconds": semantic_group_seconds,
                "semantic_ledger_seconds": semantic_ledger_seconds,
                "semantic_total_seconds": (
                    semantic_extraction_seconds
                    + semantic_temporal_seconds
                    + semantic_group_seconds
                    + semantic_ledger_seconds
                ),
                "sketch_seconds": sketch_seconds,
                "sketch_cache_hits": (
                    sketch_cache_after["hits"] - sketch_cache_before["hits"]
                ),
                "sketch_cache_misses": (
                    sketch_cache_after["misses"] - sketch_cache_before["misses"]
                ),
                "sketch_cache_precompute_seconds": (
                    sketch_cache_after["precompute_seconds"]
                    - sketch_cache_before["precompute_seconds"]
                ),
                "sketch_cache_entries": sketch_cache_after["entries"],
                "sketch_cache_bytes": sketch_cache_after["bytes"],
                "coordinate_median_block_size": float(
                    self.coordinate_median_block_size
                ),
                # Conservative bound for the largest vectorized median block:
                # values, order, sorted weights/values and cumulative weights.
                "coordinate_median_workspace_peak_bytes_bound": float(
                    min(
                        self.coordinate_median_block_size,
                        max((array.size for array in anchors), default=0),
                    )
                    * max(int(np.count_nonzero(positive)), 1)
                    * 40
                ),
                "ledger_seconds": ledger_seconds,
                "solver_seconds": solver_seconds,
                "aggregation_seconds": aggregation_seconds,
                "total_defense_seconds": time.perf_counter() - started,
            },
        )
        self._runtime_validated = False
        return result

    def _validate_updates(self, updates: UpdateList) -> None:
        if len(updates) != len(self._client_ids):
            raise ValueError("update count must match the validated client context")
        if not updates:
            raise ValueError("RTC-v3 requires at least one update")
        reference = self._global_params or []
        for client_index, (parameters, num_examples) in enumerate(updates):
            if len(parameters) != len(reference):
                raise ValueError(
                    f"client {self._client_ids[client_index]!r} parameter count mismatch"
                )
            if not np.isfinite(float(num_examples)):
                raise ValueError("reported num_examples must be finite")
            for parameter_index, (parameter, expected) in enumerate(
                zip(parameters, reference)
            ):
                candidate = np.asarray(parameter)
                if candidate.shape != expected.shape or candidate.dtype != expected.dtype:
                    raise ValueError(
                        f"client {self._client_ids[client_index]!r} parameter "
                        f"{parameter_index} metadata mismatch"
                    )
                if np.issubdtype(candidate.dtype, np.floating) and not np.all(
                    np.isfinite(candidate)
                ):
                    raise ValueError("RTC-v3 rejects NaN/Inf client parameters")

    def _validated_mass(self, updates: UpdateList) -> np.ndarray:
        if self._validated_masses is not None:
            masses = np.asarray(self._validated_masses, dtype=np.float64)
            client_caps = {
                str(key): float(value)
                for key, value in dict(
                    self.params.get("client_mass_caps", {})
                ).items()
            }
            if client_caps:
                masses = np.asarray(
                    [
                        min(value, max(0.0, client_caps.get(client_id, 0.0)))
                        for client_id, value in zip(self._client_ids, masses)
                    ],
                    dtype=np.float64,
                )
            return self._enforce_principal_caps(
                masses,
                default_cap=(
                    1.0 if self.validated_mass_policy == "unit_principal" else None
                ),
            )
        policy = self.validated_mass_policy
        if policy == "unit_principal":
            counts: dict[str, int] = {}
            for principal in self._principal_ids:
                counts[principal] = counts.get(principal, 0) + 1
            return np.asarray(
                [1.0 / counts[principal] for principal in self._principal_ids],
                dtype=np.float64,
            )
        if policy == "capped_num_examples":
            client_caps = {
                str(key): float(value)
                for key, value in dict(self.params.get("client_mass_caps", {})).items()
            }
            principal_caps = {
                str(key): float(value)
                for key, value in dict(
                    self.params.get("principal_mass_caps", {})
                ).items()
            }
            if not client_caps or not principal_caps:
                raise ValueError(
                    "capped_num_examples requires server client_mass_caps and "
                    "principal_mass_caps"
                )
            masses = np.asarray(
                [
                    min(
                        max(0.0, float(num_examples)),
                        client_caps.get(client_id, 0.0),
                    )
                    for client_id, (_, num_examples) in zip(
                        self._client_ids, updates
                    )
                ],
                dtype=np.float64,
            )
            return self._enforce_principal_caps(masses, default_cap=None)
        raise ValueError(f"unsupported RTC-v3 validated_mass_policy: {policy!r}")

    def _enforce_principal_caps(
        self,
        masses: np.ndarray,
        *,
        default_cap: float | None,
    ) -> np.ndarray:
        result = np.asarray(masses, dtype=np.float64).copy()
        principal_caps = {
            str(key): float(value)
            for key, value in dict(
                self.params.get("principal_mass_caps", {})
            ).items()
        }
        for principal in sorted(set(self._principal_ids)):
            indices = np.asarray(
                [value == principal for value in self._principal_ids],
                dtype=bool,
            )
            group_total = float(result[indices].sum(dtype=np.float64))
            raw_cap = principal_caps.get(principal, default_cap)
            if raw_cap is None:
                raise ValueError(
                    f"missing registered mass cap for principal {principal!r}"
                )
            cap = max(0.0, float(raw_cap))
            if group_total > cap and group_total > 0.0:
                result[indices] *= cap / group_total
        return result

    def _profile_scalar(self, raw: Any, *, profile: str | None = None) -> float:
        if isinstance(raw, Mapping):
            selected = profile or str(self.manifest.payload["profile"])
            if selected in raw:
                return self._profile_scalar(raw[selected], profile=selected)
            if "default" in raw:
                return self._profile_scalar(raw["default"], profile=selected)
            raise ValueError(
                f"RTC-v3 calibration value has no entry for profile {selected!r}"
            )
        value = float(raw)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("RTC-v3 calibration scalar must be finite and non-negative")
        return value

    def _profile_for_round(self, server_round: int) -> str:
        selected = str(self.manifest.payload["profile"])
        schedule = self.manifest.payload.get("training_phases", ())
        if schedule is None:
            return selected
        if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
            raise ValueError("training_phases must be a sequence")
        previous_start = 0
        for entry in schedule:
            if not isinstance(entry, Mapping):
                raise ValueError("each training phase must be a mapping")
            start = int(entry.get("start_round", 0))
            profile = str(entry.get("profile", ""))
            if start <= previous_start or not profile:
                raise ValueError(
                    "training phases require increasing positive start_round values "
                    "and non-empty profiles"
                )
            previous_start = start
            if int(server_round) >= start:
                selected = profile
        return selected

    def _active_window_scalar(self, raw: Any, window: int) -> float:
        start = max(1, self._server_round - int(window) + 1)
        profiles = {
            self._profile_for_round(round_value)
            for round_value in range(start, self._server_round + 1)
        }
        return min(
            self._profile_scalar(raw, profile=profile) for profile in profiles
        )

    def _commit_zero_round(
        self,
        updates: UpdateList,
        validated: np.ndarray,
        status: str,
        started: float,
    ) -> list[np.ndarray]:
        count = len(updates)
        resolutions = (
            tuple(self._layout.indices) if self._layout is not None else ("full",)
        )
        windows = (
            (1,)
            if self.implementation_phase == 1
            else tuple(int(value) for value in self.manifest.payload["server_windows"])
        )
        zero_exposures = {
            resolution: {"anchor": 0.0, "residual": 0.0, "total": 0.0}
            for resolution in resolutions
        }
        self._server_ledger.commit(
            self._server_round,
            zero_exposures,
        )
        if self.implementation_phase >= 4:
            profile = self._profile_for_round(self._server_round)
            self._principal_ledger.commit(
                {
                    principal: PrincipalParticipation(
                        server_round=self._server_round,
                        nominal_mass=0.0,
                        profile=profile,
                        exposures={
                            resolution: 0.0 for resolution in resolutions
                        },
                        solver_failed=False,
                    )
                    for principal in sorted(set(self._principal_ids))
                }
            )
        zero_histories = {
            (resolution, exposure_type, window): 0.0
            for resolution in resolutions
            for exposure_type in ("anchor", "residual", "total")
            for window in windows
        }
        self._record_round(
            nominal=np.zeros(count, dtype=np.float64),
            validated=validated,
            weights=np.zeros(count, dtype=np.float64),
            norms=np.zeros(count, dtype=np.float64),
            clipped_mask=np.zeros(count, dtype=bool),
            residual_norms=np.zeros(count, dtype=np.float64),
            clip_norm=self._profile_scalar(
                self.manifest.payload["clip_lower"],
                profile=self._profile_for_round(self._server_round),
            ),
            solver_status=status,
            max_constraint_violation=0.0,
            anchor_recycle_mass=0.0,
            anchor_recycle_target_mass=0.0,
            anchor_recycle_base_target_mass=0.0,
            residual_rank_cap_multiplier=np.ones(count, dtype=np.float64),
            residual_rank_reference_weight_sum=0.0,
            residual_rank_reference_valid=True,
            residual_rank_removed_mass=0.0,
            residual_rank_recycle_target_mass=0.0,
            exposures=zero_exposures,
            histories=zero_histories,
            budgets=zero_histories,
            server_windows=windows,
            cone_assignments={},
            cone_transitions={},
            principal_diagnostics={},
            principal_current_exposures={},
            cumulative_transition=CumulativeTransition(observations={}),
            direction_transition=DirectionTransition(
                observations={}, next_history={}
            ),
            direction_q_values={},
            semantic_batch=None,
            semantic_transition=None,
            semantic_risks=np.zeros(count, dtype=np.float64),
            semantic_q=np.ones(count, dtype=np.float64),
            client_q_cap=np.ones(count, dtype=np.float64),
            semantic_intervention_active=np.zeros(count, dtype=bool),
            semantic_coefficients=np.zeros(count, dtype=np.float64),
            timings={"total_defense_seconds": time.perf_counter() - started},
        )
        self._runtime_validated = False
        return [np.asarray(parameter).copy() for parameter in self._global_params or []]

    def _record_round(
        self,
        *,
        nominal: np.ndarray,
        validated: np.ndarray,
        weights: np.ndarray,
        norms: np.ndarray,
        clipped_mask: np.ndarray,
        residual_norms: np.ndarray,
        clip_norm: float,
        solver_status: str,
        max_constraint_violation: float,
        anchor_recycle_mass: float,
        anchor_recycle_target_mass: float,
        anchor_recycle_base_target_mass: float,
        residual_rank_cap_multiplier: np.ndarray,
        residual_rank_reference_weight_sum: float,
        residual_rank_reference_valid: bool,
        residual_rank_removed_mass: float,
        residual_rank_recycle_target_mass: float,
        exposures: Mapping[str, Mapping[str, float]],
        histories: Mapping[tuple[str, str, int], float],
        budgets: Mapping[tuple[str, str, int], float],
        server_windows: Sequence[int],
        cone_assignments: Mapping[str, Sequence[str]],
        cone_transitions: Mapping[str, PrototypeTransition],
        principal_diagnostics: Mapping[
            tuple[str, str, int], tuple[float, float, float]
        ],
        principal_current_exposures: Mapping[tuple[str, str], float],
        cumulative_transition: CumulativeTransition,
        direction_transition: DirectionTransition,
        direction_q_values: Mapping[tuple[str, str], float],
        semantic_batch: SemanticBatch | None,
        semantic_transition: SemanticTemporalTransition | None,
        semantic_risks: np.ndarray,
        semantic_q: np.ndarray,
        client_q_cap: np.ndarray,
        semantic_intervention_active: np.ndarray,
        semantic_coefficients: np.ndarray,
        timings: Mapping[str, float],
    ) -> None:
        self.last_client_weights = {
            client_id: float(mass)
            for client_id, mass in zip(self._client_ids, nominal)
        }
        self.last_client_aggregation_weights = {
            client_id: float(weight)
            for client_id, weight in zip(self._client_ids, weights)
        }
        self.last_client_trusts = {
            client_id: float(q_value)
            for client_id, q_value in zip(self._client_ids, semantic_q)
        }
        self._last_clipped_mask = [bool(value) for value in clipped_mask]
        self._last_clip_norm = float(clip_norm)
        self._last_constraint_tags = [
            "capped" if weight < mass - 1e-10 else ""
            for weight, mass in zip(weights, nominal)
        ]
        semantic_magnitude_risks = (
            1.0 - np.exp(-np.max(semantic_batch.z_values, axis=1))
            if semantic_batch is not None
            else np.zeros(len(nominal), dtype=np.float64)
        )
        synchronized_pairs = set(
            semantic_transition.synchronized_pairs
            if semantic_transition is not None
            else ()
        )
        semantic_direction_risks = (
            np.asarray(
                [
                    float(semantic_magnitude_risks[index])
                    if int(pair_index) in synchronized_pairs
                    else 0.0
                    for index, pair_index in enumerate(
                        semantic_batch.top_pair_indices
                        if semantic_batch is not None
                        else ()
                    )
                ],
                dtype=np.float64,
            )
            if semantic_batch is not None
            else np.zeros(len(nominal), dtype=np.float64)
        )
        semantic_influence_risks = np.minimum(
            1.0, np.asarray(semantic_coefficients, dtype=np.float64)
        )
        self._last_records = [
            RTCv3ClientRecord(
                client_id=client_id,
                principal_id=principal_id,
                validated_mass=float(validated_mass),
                nominal_mass=float(mass),
                aggregation_weight=float(weight),
                norm=float(norm),
                clipped=bool(was_clipped),
                residual_norm=float(residual_norm),
                state=(
                    "normal"
                    if float(risk) < float(self._semantic_state_thresholds["watch"])
                    else (
                        "watch"
                        if float(risk) < float(self._semantic_state_thresholds["restricted"])
                        else (
                            "restricted"
                            if float(risk) < float(self._semantic_state_thresholds["quarantined"])
                            else "quarantined"
                        )
                    )
                ),
                magnitude_risk=float(magnitude_risk),
                direction_risk=float(direction_risk),
                temporal_risk=float(risk),
                influence_risk=float(influence_risk),
                total_risk=float(risk),
                event_risk=float(risk),
            )
            for (
                client_id,
                principal_id,
                validated_mass,
                mass,
                weight,
                norm,
                was_clipped,
                residual_norm,
                risk,
                magnitude_risk,
                direction_risk,
                influence_risk,
            ) in zip(
                self._client_ids,
                self._principal_ids,
                validated,
                nominal,
                weights,
                norms,
                clipped_mask,
                residual_norms,
                semantic_risks,
                semantic_magnitude_risks,
                semantic_direction_risks,
                semantic_influence_risks,
            )
        ]
        weight_sum = float(weights.sum(dtype=np.float64))
        effective_update_mass = min(1.0, weight_sum + float(anchor_recycle_mass))
        metrics: dict[str, Any] = {
            "time_consistency_version": self.version,
            "rtc_v3_phase": float(self.implementation_phase),
            "rtc_v3_calibration_hash": self.manifest.hash,
            "rtc_v3_sketch_algorithm_version": self.sketch_algorithm_version,
            "rtc_v3_weight_sum": weight_sum,
            "rtc_v3_anchor_recycle_fraction": self.anchor_recycle_fraction,
            "rtc_v3_anchor_recycle_weighting": self.anchor_recycle_weighting,
            "rtc_v3_anchor_recycle_source_available": float(
                self.anchor_recycle_weighting == "nominal"
                or weight_sum > np.finfo(np.float64).eps
            ),
            "rtc_v3_anchor_recycle_target_mass": float(
                anchor_recycle_target_mass
            ),
            "rtc_v3_anchor_recycle_base_target_mass": float(
                anchor_recycle_base_target_mass
            ),
            "rtc_v3_anchor_recycle_mass": float(anchor_recycle_mass),
            "rtc_v3_anchor_recycle_budget_limited": float(
                anchor_recycle_mass < anchor_recycle_target_mass - 1e-10
            ),
            "rtc_v3_effective_update_mass": effective_update_mass,
            "rtc_v3_zero_update_mass": max(0.0, 1.0 - effective_update_mass),
            "rtc_v3_clip_norm": float(clip_norm),
            "rtc_v3_norm_clip_mad_k": self.norm_clip_mad_k,
            "rtc_v3_clipping_rate": float(np.mean(clipped_mask)),
            "rtc_v3_solver_status": solver_status,
            "rtc_v3_solver_fallback": float(solver_status != "optimized"),
            "rtc_v3_max_constraint_violation": float(max_constraint_violation),
            "rtc_v3_principal_count": float(len(set(self._principal_ids))),
            "rtc_v3_cone_profile": str(self._active_cone_profile or "legacy"),
            "rtc_v3_cone_mode": self._cone_mode,
            "rtc_v3_semantic_ablation": self._semantic_ablation,
            "rtc_v3_semantic_intervention_risk_floor": (
                self._semantic_intervention_risk_floor
            ),
            "rtc_v3_semantic_hard_exposure_risk_floor": (
                self._semantic_hard_exposure_risk_floor
            ),
            "rtc_v3_cumulative_q_cap_power": float(
                self.cumulative_q_cap_power
            ),
            "rtc_v3_residual_rank_cap_top_k": float(
                self.residual_rank_cap_top_k
            ),
            "rtc_v3_residual_rank_cap_factor": float(
                self.residual_rank_cap_factor
            ),
            "rtc_v3_residual_rank_recycle_fraction": float(
                self.residual_rank_recycle_fraction
            ),
            "rtc_v3_residual_rank_cap_active_count": float(
                np.count_nonzero(
                    np.asarray(residual_rank_cap_multiplier, dtype=np.float64)
                    < 1.0 - 1e-12
                )
            ),
            "rtc_v3_residual_rank_cap_client_ids": "|".join(
                str(client_id)
                for client_id, multiplier in zip(
                    self._client_ids,
                    np.asarray(residual_rank_cap_multiplier, dtype=np.float64),
                )
                if float(multiplier) < 1.0 - 1e-12
            ),
            "rtc_v3_residual_rank_reference_weight_sum": float(
                residual_rank_reference_weight_sum
            ),
            "rtc_v3_residual_rank_reference_valid": float(
                residual_rank_reference_valid
            ),
            "rtc_v3_residual_rank_removed_mass": float(
                residual_rank_removed_mass
            ),
            "rtc_v3_residual_rank_recycle_target_mass": float(
                residual_rank_recycle_target_mass
            ),
            "rtc_v3_client_q_cap_min": float(
                np.min(client_q_cap) if len(client_q_cap) else 1.0
            ),
            "rtc_v3_client_q_cap_active_count": float(
                np.count_nonzero(client_q_cap < 1.0 - 1e-12)
            ),
            "rtc_v3_semantic_intervention_active_count": float(
                np.count_nonzero(semantic_intervention_active)
            ),
            "rtc_v3_cone_online_updates_enabled": float(
                self._cone_online_updates_enabled
            ),
        }
        if semantic_batch is not None:
            metrics.update(
                {
                    "rtc_v3_semantic_enabled": 1.0,
                    "rtc_v3_semantic_risk_mean": float(np.mean(semantic_risks)),
                    "rtc_v3_semantic_risk_max": float(np.max(semantic_risks)),
                    "rtc_v3_semantic_q_min_round": float(np.min(semantic_q)),
                    "rtc_v3_semantic_exposure": float(
                        np.dot(semantic_coefficients, weights)
                    ),
                    "rtc_v3_semantic_synchronized_pair_count": float(
                        len(
                            semantic_transition.synchronized_pairs
                            if semantic_transition is not None
                            else ()
                        )
                    ),
                    "rtc_v3_semantic_top_pairs_json": json.dumps(
                        [
                            semantic_batch.pairs[int(index)]
                            for index in semantic_batch.top_pair_indices
                        ],
                        separators=(",", ":"),
                    ),
                }
            )
            for window in tuple(
                int(value) for value in self._semantic_config["global_windows"]
            ):
                history = float(histories[("semantic", "risk", window)])
                budget = float(budgets[("semantic", "risk", window)])
                metrics[f"rtc_v3_semantic_history_w{window}"] = history
                metrics[f"rtc_v3_semantic_budget_w{window}"] = budget
                metrics[f"rtc_v3_semantic_used_w{window}"] = history + float(
                    metrics["rtc_v3_semantic_exposure"]
                )
                metrics[f"rtc_v3_semantic_active_w{window}"] = float(
                    float(metrics[f"rtc_v3_semantic_used_w{window}"])
                    >= budget - 1e-8
                )
        for resolution, typed_exposures in exposures.items():
            safe_resolution = resolution.replace(":", "_")
            for exposure_type in ("anchor", "residual", "total"):
                exposure = float(typed_exposures[exposure_type])
                metrics[
                    f"rtc_v3_{safe_resolution}_{exposure_type}_exposure"
                ] = exposure
                for window in server_windows:
                    history = float(
                        histories[(resolution, exposure_type, int(window))]
                    )
                    budget = float(
                        budgets[(resolution, exposure_type, int(window))]
                    )
                    prefix = (
                        f"rtc_v3_{safe_resolution}_{exposure_type}"
                    )
                    metrics[f"{prefix}_history_w{window}"] = history
                    metrics[f"{prefix}_budget_w{window}"] = budget
                    metrics[f"{prefix}_used_w{window}"] = history + exposure
                    metrics[f"{prefix}_active_w{window}"] = float(
                        history + exposure >= budget - 1e-8
                    )
            for exposure_type, exposure in typed_exposures.items():
                if not exposure_type.startswith("cone:"):
                    continue
                cone_name = exposure_type.replace(":", "_")
                for window in server_windows:
                    history = float(
                        histories[(resolution, exposure_type, int(window))]
                    )
                    budget = float(
                        budgets[(resolution, exposure_type, int(window))]
                    )
                    prefix = f"rtc_v3_{safe_resolution}_{cone_name}"
                    metrics[f"{prefix}_exposure"] = float(exposure)
                    metrics[f"{prefix}_used_w{window}"] = history + float(exposure)
                    metrics[f"{prefix}_budget_w{window}"] = budget
        if cone_assignments:
            overflow_count = sum(
                assignment == "overflow"
                for assignments in cone_assignments.values()
                for assignment in assignments
            )
            metrics["rtc_v3_cone_overflow_assignments"] = float(overflow_count)
            metrics["rtc_v3_cone_prototype_count"] = float(
                sum(
                    len(bank.prototypes)
                    for bank in self._prototype_banks.values()
                )
            )
            for resolution, bank in self._prototype_banks.items():
                safe_resolution = resolution.replace(":", "_")
                metrics[f"rtc_v3_cone_prototype_ids_{safe_resolution}"] = ",".join(
                    prototype.cone_id for prototype in bank.prototypes
                )
            similarities = [
                similarity
                for transition in cone_transitions.values()
                for similarity in transition.similarities
            ]
            if similarities:
                metrics["rtc_v3_cone_similarity_min"] = float(min(similarities))
                metrics["rtc_v3_cone_similarity_mean"] = float(
                    np.mean(similarities)
                )
            update_metrics = [
                bank.last_update_metrics for bank in self._prototype_banks.values()
            ]
            metrics["rtc_v3_cone_update_candidate_count"] = float(
                sum(value["candidate_count"] for value in update_metrics)
            )
            metrics["rtc_v3_cone_update_applied_count"] = float(
                sum(value["applied_count"] for value in update_metrics)
            )
            metrics["rtc_v3_cone_update_max_angular_drift"] = float(
                max(
                    (value["max_angular_drift"] for value in update_metrics),
                    default=0.0,
                )
            )
            metrics["rtc_v3_cone_update_drift_clipped_count"] = float(
                sum(value["drift_clipped_count"] for value in update_metrics)
            )
            support_by_resolution = {
                resolution: bank.last_update_metrics[
                    "principal_support_by_cone"
                ]
                for resolution, bank in sorted(self._prototype_banks.items())
            }
            applied_by_resolution = {
                resolution: list(bank.last_update_metrics["applied_cone_ids"])
                for resolution, bank in sorted(self._prototype_banks.items())
            }
            metrics["rtc_v3_cone_update_principal_support_json"] = json.dumps(
                support_by_resolution, sort_keys=True, separators=(",", ":")
            )
            metrics["rtc_v3_cone_update_applied_ids_json"] = json.dumps(
                applied_by_resolution, sort_keys=True, separators=(",", ":")
            )
        if principal_diagnostics:
            active = 0
            maximum_usage = 0.0
            for (
                principal,
                resolution,
                window,
            ), (history, budget, _) in principal_diagnostics.items():
                current = float(
                    principal_current_exposures[(principal, resolution)]
                )
                used = history + current
                if budget > 0.0:
                    maximum_usage = max(maximum_usage, used / budget)
                active += int(used >= budget - 1e-8)
            metrics["rtc_v3_principal_budget_active_count"] = float(active)
            metrics["rtc_v3_principal_budget_max_usage"] = float(maximum_usage)
        if cumulative_transition.observations and self._cumulative.enabled:
            observations = list(cumulative_transition.observations.values())
            metrics["rtc_v3_cumulative_q_min"] = float(
                min(observation.q for observation in observations)
            )
            metrics["rtc_v3_cumulative_statistic_max"] = float(
                max(observation.statistic for observation in observations)
            )
            metrics["rtc_v3_cumulative_active_count"] = float(
                sum(observation.q < 1.0 for observation in observations)
            )
        if direction_transition.observations:
            observations = list(direction_transition.observations.values())
            metrics["rtc_v3_direction_stable_count"] = float(
                sum(observation.cone_id != "overflow" for observation in observations)
            )
            metrics["rtc_v3_direction_repeat_max"] = float(
                max(observation.repeat_count for observation in observations)
            )
            metrics["rtc_v3_direction_q_min"] = float(
                min(direction_q_values.values(), default=1.0)
            )
            metrics["rtc_v3_direction_active_count"] = float(
                sum(value < 1.0 for value in direction_q_values.values())
            )
        metrics.update({key: float(value) for key, value in timings.items()})
        self.last_round_metrics = metrics
