"""Strictly paired generalized-Byzantine benchmark for promoted RTC-V3."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from attacks.spec import attack_contract_payload, attack_source_hash, get_attack_spec
from attacks.spec import COORDINATED_ATTACKS
from config.config_loader import AttackConfig
from experiments import periodic_attack
from experiments.rtc_fedavg_comparison import build_rtc_v3_custom_params
from experiments.trial_plan import attach_trial_plans, paired_fedavg_clean_specs


CANONICAL_ATTACKS = (
    "gaussian_noise",
    "random_noise",
    "sign_flip",
    "lie",
    "min_max",
    "min_sum",
    "scaling_backdoor",
    "label_flip_targeted",
    "label_flip_all_reverse",
    "dba",
)
ROBUST_AGR_REPRODUCTION_ATTACKS = ("lie", "min_max", "min_sum")
ROBUST_AGR_REPRODUCTION_DEFENSES = (
    "krum", "multi_krum", "median", "trimmed_mean"
)
BACKDOOR_SCREEN_ATTACK_START = 11

SUPPORTED_ATTACKS = ("none", *CANONICAL_ATTACKS, "alie")
DEFAULT_DEFENSES = (
    "fedavg",
    "rtc_full",
    "krum",
    "trimmed_mean",
    "median",
    "foolsgold",
    "rfa",
    "freqfed",
    "fltrust",
)

DEFENSES: Mapping[str, tuple[str, Mapping[str, Any]]] = {
    "fedavg": ("none", {}),
    "rtc_full": ("rtc_full", {}),
    "rtc_anchor_recycle": ("rtc_full", {}),
    "rtc_semantic_observe": ("rtc_full", {}),
    "rtc_semantic_risk_gated": ("rtc_full", {}),
    "rtc_semantic_restricted_only": ("rtc_full", {}),
    "rtc_cumulative_q_cap": ("rtc_full", {}),
    "rtc_cumulative_q_cap_anchor": ("rtc_full", {}),
    "rtc_cumulative_q_cap_accepted_anchor": ("rtc_full", {}),
    "rtc_r0_direction_observe": ("rtc_full", {}),
    "rtc_r0b_geometric_observe": ("rtc_full", {}),
    "rtc_r1_spectral_baseline": ("rtc_full", {}),
    "rtc_r1_spectral_cap": ("rtc_full", {}),
    "rtc_r1_spectral_multikrum": ("krum", {}),
    "rtc_r1c_pairwise_baseline": ("rtc_full", {}),
    "rtc_r1c_pairwise_cap": ("rtc_full", {}),
    "rtc_r1c_pairwise_multikrum": ("krum", {}),
    "rtc_r2_raw_baseline": ("rtc_full", {}),
    "rtc_r2_raw_cap": ("rtc_full", {}),
    "rtc_r2_raw_multikrum": ("krum", {}),
    "rtc_r2_raw_rfa": ("rfa", {"num_iterations": 3, "smoothing": 1e-6, "use_num_examples": True}),
    "rtc_i12_b0": ("rtc_full", {}),
    "rtc_i12_direction": ("rtc_full", {}),
    "rtc_i12_norm": ("rtc_full", {}),
    "rtc_i12_combined": ("rtc_full", {}),
    "rtc_i12_temporal_observe": ("rtc_full", {"temporal_residual_observe_only": True}),
    "rtc_i12_lower_observe": ("rtc_full", {"lower_tail_mode": "observe", "lower_tail_calibration": "config/rtc_r3_lower_tail_calibration.json"}),
    "rtc_i12_lower_cap": ("rtc_full", {"lower_tail_mode": "cap", "lower_tail_calibration": "config/rtc_r3_lower_tail_calibration.json"}),
    "rtc_i12_reference_observe": ("rtc_full", {"reference_history_observe_only": True}),
    "rtc_i12_guard_observe": ("rtc_full", {"reference_guard_mode": "observe"}),
    "rtc_i12_guard_cap": ("rtc_full", {"reference_guard_mode": "cap"}),
    "rtc_i12_eligibility_observe": ("rtc_full", {"reference_eligibility_mode": "observe"}),
    "rtc_i12_eligibility_cap": ("rtc_full", {"reference_eligibility_mode": "cap"}),
    "rtc_i12_multikrum": ("krum", {}),
    "rtc_i12_rfa": ("rfa", {"num_iterations": 3, "smoothing": 1e-6, "use_num_examples": True}),
    "rtc_b4_residual_rank_cap": ("rtc_full", {}),
    "rtc_b5_clip_mad_225": ("rtc_full", {}),
    "krum": ("krum", {}),
    "multi_krum": ("krum", {}),
    "trimmed_mean": ("trimmed_mean", {}),
    "median": ("median", {}),
    "foolsgold": ("foolsgold", {}),
    "rfa": (
        "rfa",
        {
            "num_iterations": 3,
            "smoothing": 1e-6,
            "use_num_examples": True,
        },
    ),
    "freqfed": (
        "freqfed",
        {
            "min_cluster_size": 2,
            "min_samples": 1,
            "min_selected_clients": 2,
            "allow_single_cluster": True,
        },
    ),
    "fltrust": ("fltrust", {}),
}

ATTACK_PARAMETERS: Mapping[str, Mapping[str, Any]] = {
    "none": {},
    "gaussian_noise": {"gaussian_noise_mean": 0.0, "gaussian_noise_std": 0.1},
    "random_noise": {
        "random_noise_scale": 10.0,
        "random_noise_distribution": "rademacher",
    },
    "sign_flip": {"sign_flip_scale": 10.0},
    "lie": {"lie_z": 0.5, "coordinated_attack_knowledge": "all_updates"},
    "alie": {"lie_z": 0.5, "coordinated_attack_knowledge": "all_updates"},
    "min_max": {
        "coordinated_attack_knowledge": "all_updates",
        "optimization_perturbation": "inverse_sign",
        "optimization_gamma_init": 0.001,
        "optimization_tolerance": 1e-6,
        "optimization_max_iterations": 64,
    },
    "min_sum": {
        "coordinated_attack_knowledge": "all_updates",
        "optimization_perturbation": "inverse_sign",
        "optimization_gamma_init": 0.001,
        "optimization_tolerance": 1e-6,
        "optimization_max_iterations": 64,
    },
    "scaling_backdoor": {
        "aggregation_aware_scaling": True,
        "replacement_gain": 0.65,
        "poison_fraction": 0.2,
    },
    "label_flip_targeted": {
        "label_flip_source_label": 5,
        "label_flip_target_label": 3,
        "label_flip_poison_fraction": 1.0,
    },
    "label_flip_all_reverse": {"label_flip_poison_fraction": 1.0},
    "dba": {
        "aggregation_aware_scaling": True,
        "replacement_gain": 0.7,
        "poison_fraction": 0.2,
        "dba_scale_update": True,
        "dba_pattern_mode": "paper_cifar_1x6_2x2",
        "dba_trigger_value_mode": "cifar10_normalized_white",
    },
}

STRENGTH_GRID: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "gaussian_noise": {
        "weak": {"gaussian_noise_mean": 0.0, "gaussian_noise_std": 0.01},
        "medium": {"gaussian_noise_mean": 0.0, "gaussian_noise_std": 0.05},
        "strong": {"gaussian_noise_mean": 0.0, "gaussian_noise_std": 0.1},
    },
    "random_noise": {
        level: {"random_noise_scale": scale, "random_noise_distribution": "rademacher"}
        for level, scale in (("weak", 1.0), ("medium", 5.0), ("strong", 10.0))
    },
    "sign_flip": {
        level: {"sign_flip_scale": scale}
        for level, scale in (("weak", 1.0), ("medium", 5.0), ("strong", 10.0))
    },
    "lie": {
        level: {"lie_z": value, "coordinated_attack_knowledge": "all_updates"}
        for level, value in (("weak", 0.25), ("medium", 0.5), ("strong", 1.0))
    },
    "alie": {
        level: {"lie_z": value, "coordinated_attack_knowledge": "all_updates"}
        for level, value in (("weak", 0.25), ("medium", 0.5), ("strong", 1.0))
    },
    "min_max": {
        level: {
            **ATTACK_PARAMETERS["min_max"],
            "optimization_gamma_fraction": fraction,
        }
        for level, fraction in (("weak", 0.25), ("medium", 0.5), ("strong", 1.0))
    },
    "min_sum": {
        level: {
            **ATTACK_PARAMETERS["min_sum"],
            "optimization_gamma_fraction": fraction,
        }
        for level, fraction in (("weak", 0.25), ("medium", 0.5), ("strong", 1.0))
    },
    "scaling_backdoor": {
        "weak": {
            "aggregation_aware_scaling": True,
            "replacement_gain": 0.35,
            "poison_fraction": 0.1,
        },
        "medium": {
            "aggregation_aware_scaling": True,
            "replacement_gain": 0.65,
            "poison_fraction": 0.2,
        },
        "strong": {
            "aggregation_aware_scaling": True,
            "replacement_gain": 1.0,
            "poison_fraction": 0.3,
        },
    },
    "label_flip_targeted": {
        level: {
            "label_flip_source_label": 5,
            "label_flip_target_label": 3,
            "label_flip_poison_fraction": fraction,
        }
        for level, fraction in (("weak", 0.25), ("medium", 0.5), ("strong", 1.0))
    },
    "label_flip_all_reverse": {
        level: {"label_flip_poison_fraction": fraction}
        for level, fraction in (("weak", 0.25), ("medium", 0.5), ("strong", 1.0))
    },
    "dba": {
        level: {
            "aggregation_aware_scaling": True,
            "replacement_gain": gain,
            "poison_fraction": poison,
            "dba_scale_update": True,
            "dba_pattern_mode": "paper_cifar_1x6_2x2",
            "dba_trigger_value_mode": "cifar10_normalized_white",
        }
        for level, gain, poison in (
            ("weak", 0.4, 0.1),
            ("medium", 0.7, 0.2),
            ("strong", 1.0, 0.3),
        )
    },
}

_ATTACK_PARAMETER_KEYS = frozenset(
    key
    for profiles in STRENGTH_GRID.values()
    for parameters in profiles.values()
    for key in parameters
) | {"boost_factor"}


def _runtime_attack_config(
    attack: str,
    parameters: Mapping[str, Any],
    *,
    malicious_fraction: float,
    attack_start_round: int,
    attack_end_round: int,
) -> AttackConfig:
    config = AttackConfig(
        enabled=attack != "none",
        type=attack,
        malicious_fraction=float(malicious_fraction),
        attack_start_round=int(attack_start_round),
        attack_end_round=int(attack_end_round),
    )
    runtime_aliases = {
        "label_flip_source_label": "source_label",
        "label_flip_target_label": "target_label",
    }
    for key, value in parameters.items():
        runtime_key = runtime_aliases.get(key, key)
        if hasattr(config, runtime_key):
            setattr(config, runtime_key, value)
    if "boost_factor" in parameters:
        boost = float(parameters["boost_factor"])
        config.dba_boost_factor = boost
        config.model_replacement_boost_factor = boost
        config.mpaf_lambda = boost
    return config


def _csv(raw: str | Iterable[Any], defaults: Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = tuple(value.strip() for value in raw.split(",") if value.strip())
    else:
        values = tuple(str(value) for value in raw)
    return values or tuple(defaults)


def _seeds(raw: str | Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in _csv(raw, ("42", "43", "44")))


def _load_attack_freeze(path_value: str | Path, required_attacks=None) -> Mapping[str, Mapping[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "RTCByzantineAttackFreezeV1":
        raise ValueError(f"Unsupported attack-freeze schema in {path}")
    observed = str(payload.get("implementation_source_sha256", ""))
    expected = attack_source_hash()
    if observed != expected:
        raise ValueError(
            "Attack freeze implementation hash is stale: "
            f"observed={observed or '<missing>'} expected={expected}"
        )
    attacks = payload.get("attacks")
    if not isinstance(attacks, Mapping):
        raise ValueError("Attack freeze must contain an attacks mapping")
    if 'random_noise' in attacks and (required_attacks is None or 'random_noise' in required_attacks):
        from experiments.rtc_v3.random_noise_validation import validate_freeze_entry
        validate_freeze_entry(attacks['random_noise'], path.parent)
    return attacks


def build_matrix(args) -> list[dict[str, Any]]:
    attacks = _csv(getattr(args, "attacks", ""), CANONICAL_ATTACKS)
    defenses = _csv(getattr(args, "defenses", ""), DEFAULT_DEFENSES)
    unknown_attacks = set(attacks) - set(SUPPORTED_ATTACKS)
    unknown_defenses = set(defenses) - set(DEFENSES)
    if unknown_attacks:
        raise ValueError(f"Unsupported Byzantine attacks: {sorted(unknown_attacks)}")
    if unknown_defenses:
        raise ValueError(f"Unsupported Byzantine defenses: {sorted(unknown_defenses)}")
    freeze_path = str(getattr(args, "byzantine_attack_freeze", "") or "").strip()
    frozen = _load_attack_freeze(freeze_path, attacks) if freeze_path else None
    if frozen is not None:
        missing_frozen = set(attacks).difference({"none"}).difference(frozen)
        if missing_frozen:
            raise ValueError(
                f"Attack freeze is missing attacks: {sorted(missing_frozen)}"
            )

    num_clients = int(args.num_clients)
    clients_per_round = (
        num_clients
        if bool(getattr(args, "smoke", False))
        else max(
            1,
            min(
                num_clients,
                int(math.ceil(num_clients * float(args.participation_rate))),
            ),
        )
    )
    malicious_per_round = max(
        1, int(math.floor(clients_per_round * float(args.malicious_fraction)))
    )
    rtc_custom = build_rtc_v3_custom_params(
        manifest_path=getattr(args, "rtc_v3_manifest", "") or "",
        implementation_phase=int(getattr(args, "rtc_v3_phase", 6)),
        num_clients=num_clients,
    )
    rows: list[dict[str, Any]] = []
    attack_start = 1 if bool(getattr(args, "smoke", False)) else int(args.attack_start_round)
    attack_end = 3 if bool(getattr(args, "smoke", False)) else int(args.attack_end_round)
    for seed in _seeds(getattr(args, "seeds", "")):
        for attack in attacks:
            attack_spec = get_attack_spec(attack)
            frozen_entry = frozen.get(attack, {}) if frozen is not None else {}
            parameters = dict(
                frozen_entry.get("parameters", ATTACK_PARAMETERS[attack])
            )
            # The shared runner materializes these legacy scale fields for every
            # attack config. Bind them into the hash even when the attack does
            # not consume them so manifest and runtime dataclasses are identical.
            parameters.setdefault("boost_factor", float(args.boost_factor))
            runtime_config = _runtime_attack_config(
                attack,
                parameters,
                malicious_fraction=float(args.malicious_fraction),
                attack_start_round=attack_start,
                attack_end_round=attack_end,
            )
            contract = attack_contract_payload(attack, vars(runtime_config))
            group = (
                "clean"
                if attack == "none"
                else "targeted"
                if attack_spec.objective == "targeted_integrity"
                else "untargeted"
            )
            for defense in defenses:
                defense_type, custom = DEFENSES[defense]
                if defense.startswith('rtc_i12_'):
                    custom = {**custom,
                              'spectral_direction_mode': 'cap' if defense in ('rtc_i12_direction', 'rtc_i12_combined', 'rtc_i12_temporal_observe', 'rtc_i12_lower_observe', 'rtc_i12_lower_cap', 'rtc_i12_reference_observe', 'rtc_i12_guard_observe', 'rtc_i12_guard_cap', 'rtc_i12_eligibility_observe', 'rtc_i12_eligibility_cap') else 'observe',
                              'spectral_direction_calibration': 'config/rtc_r1c_pairwise_calibration.json',
                              'raw_norm_mode': 'cap' if defense in ('rtc_i12_norm', 'rtc_i12_combined', 'rtc_i12_temporal_observe', 'rtc_i12_lower_observe', 'rtc_i12_lower_cap', 'rtc_i12_reference_observe', 'rtc_i12_guard_observe', 'rtc_i12_guard_cap', 'rtc_i12_eligibility_observe', 'rtc_i12_eligibility_cap') else 'observe',
                              'raw_norm_calibration': 'config/rtc_r2_raw_norm_calibration.json'}
                    if defense_type == 'rtc_full':
                        custom = {**rtc_custom, **custom, 'semantic_intervention_risk_floor': .5,
                                  'cumulative_q_cap_power': 1, 'anchor_recycle_fraction': .51,
                                  'anchor_recycle_weighting': 'accepted'}
                if defense.startswith('rtc_r2_raw_'):
                    custom = {**custom, 'spectral_direction_mode': 'observe',
                              'spectral_direction_calibration': 'config/rtc_r1c_pairwise_calibration.json',
                              'raw_norm_mode': 'cap' if defense == 'rtc_r2_raw_cap' else 'observe',
                              'raw_norm_calibration': 'config/rtc_r2_raw_norm_calibration.json'}
                    if defense in ('rtc_r2_raw_baseline', 'rtc_r2_raw_cap'):
                        custom = {**rtc_custom, **custom, 'semantic_intervention_risk_floor': .5,
                                  'cumulative_q_cap_power': 1, 'anchor_recycle_fraction': .51,
                                  'anchor_recycle_weighting': 'accepted'}
                if defense.startswith('rtc_r1c_pairwise_'):
                    custom = {'spectral_direction_mode': 'cap' if defense == 'rtc_r1c_pairwise_cap' else 'observe',
                              'spectral_direction_calibration': 'config/rtc_r1c_pairwise_calibration.json'}
                    if defense != 'rtc_r1c_pairwise_multikrum':
                        custom = {**rtc_custom, **custom, 'semantic_intervention_risk_floor': .5,
                                  'cumulative_q_cap_power': 1, 'anchor_recycle_fraction': .51,
                                  'anchor_recycle_weighting': 'accepted'}
                if defense.startswith('rtc_r1_spectral_'):
                    custom = {'spectral_direction_mode': 'cap' if defense == 'rtc_r1_spectral_cap' else 'observe',
                              'spectral_direction_calibration': 'config/rtc_r1_spectral_calibration.json'}
                    if defense != 'rtc_r1_spectral_multikrum':
                        custom = {**rtc_custom, **custom, 'semantic_intervention_risk_floor': .5,
                                  'cumulative_q_cap_power': 1, 'anchor_recycle_fraction': .51,
                                  'anchor_recycle_weighting': 'accepted'}
                if defense in ("rtc_r0_direction_observe", "rtc_r0b_geometric_observe"):
                    custom = {**rtc_custom, "semantic_intervention_risk_floor": 0.5,
                              "cumulative_q_cap_power": 1, "anchor_recycle_fraction": 0.51,
                              "anchor_recycle_weighting": "accepted", "direction_observe_only": True}
                    if defense == "rtc_r0b_geometric_observe":
                        custom["direction_observe_reference"] = "geometric_median"
                if defense in {
                    "rtc_full",
                    "rtc_anchor_recycle",
                    "rtc_semantic_observe",
                    "rtc_semantic_risk_gated",
                    "rtc_semantic_restricted_only",
                    "rtc_cumulative_q_cap",
                    "rtc_cumulative_q_cap_anchor",
                    "rtc_cumulative_q_cap_accepted_anchor",
                    "rtc_b4_residual_rank_cap",
                    "rtc_b5_clip_mad_225",
                }:
                    custom = dict(rtc_custom)
                if defense == "rtc_anchor_recycle":
                    custom["anchor_recycle_fraction"] = float(
                        getattr(args, "rtc_v3_anchor_recycle_fraction", 1.0)
                    )
                if defense == "rtc_semantic_observe":
                    custom["semantic_ablation"] = "observe"
                if defense == "rtc_semantic_risk_gated":
                    custom["semantic_intervention_risk_floor"] = 0.10
                if defense == "rtc_semantic_restricted_only":
                    custom["semantic_intervention_risk_floor"] = 0.50
                if defense == "rtc_cumulative_q_cap":
                    custom["semantic_intervention_risk_floor"] = 0.50
                    custom["cumulative_q_cap_power"] = 1
                if defense in {
                    "rtc_cumulative_q_cap_anchor",
                    "rtc_cumulative_q_cap_accepted_anchor",
                    "rtc_b4_residual_rank_cap",
                    "rtc_b5_clip_mad_225",
                }:
                    custom["semantic_intervention_risk_floor"] = 0.50
                    custom["cumulative_q_cap_power"] = 1
                    custom["anchor_recycle_fraction"] = float(
                        getattr(args, "rtc_v3_anchor_recycle_fraction", 1.0)
                    )
                if defense in {
                    "rtc_cumulative_q_cap_accepted_anchor",
                    "rtc_b4_residual_rank_cap",
                    "rtc_b5_clip_mad_225",
                }:
                    custom["anchor_recycle_weighting"] = "accepted"
                if defense == "rtc_b4_residual_rank_cap":
                    custom["residual_rank_cap_top_k"] = 2
                    custom["residual_rank_cap_factor"] = 0.5
                    custom["residual_rank_recycle_fraction"] = 1.0
                if defense == "rtc_b5_clip_mad_225":
                    custom["norm_clip_mad_k"] = 2.25
                row: dict[str, Any] = {
                    "benchmark_version": 4,
                    "attack_group": group,
                    "attack": attack,
                    "attack_version": attack_spec.version,
                    "attack_contract_hash": contract["contract_hash"],
                    "attack_implementation_hash": contract["implementation_hash"],
                    "strength_level": str(
                        "not_applicable"
                        if attack == "none"
                        else frozen_entry.get("strength_level", "provisional")
                    ),
                    "attack_parameter_status": (
                        "not_applicable"
                        if attack == "none"
                        else "frozen"
                        if frozen is not None
                        else "provisional"
                    ),
                    "period": "continuous_1_0",
                    "on_rounds": 1,
                    "off_rounds": 0,
                    "malicious_fraction": float(args.malicious_fraction),
                    "seed": seed,
                    "defense": defense,
                    "defense_type": defense_type,
                    "custom_params": dict(custom),
                    "attack_start_round": attack_start,
                    "attack_end_round": attack_end,
                    "partition": str(args.partition),
                    "dirichlet_alpha": float(args.dirichlet_alpha),
                    "participation_rate": float(args.participation_rate),
                    "krum_num_malicious": malicious_per_round,
                    "krum_num_to_select": (
                        max(1, clients_per_round - malicious_per_round - 2)
                        if defense in ("multi_krum", "rtc_r1_spectral_multikrum", "rtc_r1c_pairwise_multikrum", "rtc_r2_raw_multikrum", "rtc_i12_multikrum")
                        else 1
                    ),
                    "trim_fraction": float(args.malicious_fraction),
                    **parameters,
                }
                rows.append(row)
    return rows


def build_screening_matrix(args) -> list[dict[str, Any]]:
    """Build FedAvg-only weak/medium/strong cells before parameter freeze."""
    strength_levels = _csv(
        getattr(args, "strength_levels", ""), ("weak", "medium", "strong")
    )
    unknown_levels = set(strength_levels) - {"weak", "medium", "strong"}
    if unknown_levels:
        raise ValueError(f"Unsupported Byzantine strength levels: {sorted(unknown_levels)}")
    original_defenses = getattr(args, "defenses", "")
    args.defenses = "fedavg"
    try:
        base_rows = build_matrix(args)
    finally:
        args.defenses = original_defenses
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        attack = str(base["attack"])
        for level, strength in STRENGTH_GRID[attack].items():
            if level not in strength_levels:
                continue
            row = dict(base)
            for key in _ATTACK_PARAMETER_KEYS:
                row.pop(key, None)
            parameters = dict(strength)
            parameters.setdefault("boost_factor", float(args.boost_factor))
            row.update(parameters)
            row["strength_level"] = level
            if attack in {"dba", "scaling_backdoor"} and not bool(args.smoke):
                screening_rounds = 25 if args.rounds is None else int(args.rounds)
                row["attack_start_round"] = min(
                    BACKDOOR_SCREEN_ATTACK_START,
                    max(1, screening_rounds),
                )
            else:
                row["attack_start_round"] = 11 if attack == "random_noise" and not args.smoke else 1
            runtime_config = _runtime_attack_config(
                attack,
                parameters,
                malicious_fraction=float(row["malicious_fraction"]),
                attack_start_round=int(row["attack_start_round"]),
                attack_end_round=int(row["attack_end_round"]),
            )
            contract = attack_contract_payload(attack, vars(runtime_config))
            row["attack_contract_hash"] = contract["contract_hash"]
            row["attack_implementation_hash"] = contract["implementation_hash"]
            rows.append(row)
    return rows


def _write_matrix(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            **{key: value for key, value in row.items() if key != "custom_params"},
            "custom_params": json.dumps(row["custom_params"], sort_keys=True),
        }
        for row in rows
    )
    frame.to_csv(path, index=False)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _interleave_plan_clean(
    matrix: Sequence[dict[str, Any]], clean: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    clean_by_group = {
        str(spec.get("counterfactual_for", spec.get("pairing_group_id", ""))): spec
        for spec in clean
    }
    emitted: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for index, spec in enumerate(matrix):
        ordered.append(spec)
        group = str(spec.get("pairing_group_id", ""))
        # Add the shared clean only after the final defense cell for this plan.
        later_same_group = any(
            str(other.get("pairing_group_id", "")) == group
            for other in matrix[index + 1:]
        )
        if not later_same_group and group in clean_by_group and group not in emitted:
            ordered.append(clean_by_group[group])
            emitted.add(group)
    ordered.extend(
        spec
        for group, spec in clean_by_group.items()
        if group not in emitted
    )
    return ordered


def validate_attack_execution(
    specs: Sequence[Mapping[str, Any]], rounds_dir: Path
) -> list[dict[str, Any]]:
    """Validate attack-specific smoke/formal execution evidence."""
    gates: list[dict[str, Any]] = []
    for spec in specs:
        attack = str(spec["attack"])
        if attack == "none":
            continue
        path = rounds_dir / f"{periodic_attack.run_id(dict(spec))}.csv"
        passed = True
        detail = "complete"
        status_path = rounds_dir.parent / "status" / f"{periodic_attack.run_id(dict(spec))}.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if periodic_attack.valid_numerical_terminal(status, dict(spec)):
                gates.append({"gate": f"attack_execution:{periodic_attack.run_id(dict(spec))}",
                              "passed": True, "observed": "terminated_numerical",
                              "required": "completed or structured numerical termination"})
                continue
        if not path.is_file() and status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                status = {}
                detail = str(exc)
            if status.get("state") == "diverged" and status.get("error_type") == "nonfinite_aggregate":
                gates.append({
                    "gate": f"attack_execution:{periodic_attack.run_id(dict(spec))}",
                    "passed": True,
                    "observed": f"collapsed_invalid at round {status.get('divergence_round', status.get('last_round', 'unknown'))}",
                    "required": "complete training or explicitly recorded collapsed/invalid evidence",
                })
                continue
        try:
            frame = pd.read_csv(path)
            active = frame[
                (pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1)
                & (
                    pd.to_numeric(
                        frame.get("fit_selected_active_attackers"), errors="coerce"
                    ).fillna(0)
                    > 0
                )
            ]
            if active.empty:
                raise ValueError("no round contains an active selected attacker")
            if attack == "random_noise":
                from experiments.rtc_v3.random_noise_validation import validate_client_evidence
                validate_client_evidence(spec, frame, rounds_dir.parent / "raw")
            if attack in COORDINATED_ATTACKS:
                applied = pd.to_numeric(
                    active["fit_attack_coordinator_applied"], errors="coerce"
                )
                if not (applied == 1).all():
                    raise ValueError("coordinator was not applied in every active round")
                if attack in {"min_max", "min_sum"}:
                    value = pd.to_numeric(
                        active["fit_attack_coordinator_constraint_value"],
                        errors="coerce",
                    )
                    limit = pd.to_numeric(
                        active["fit_attack_coordinator_constraint_limit"],
                        errors="coerce",
                    )
                    if value.isna().any() or limit.isna().any() or not (
                        value <= limit + 1e-5 * (1.0 + limit.abs())
                    ).all():
                        raise ValueError("published optimization constraint violated")
                    for column in (
                        "fit_attack_coordinator_r_max",
                        "fit_attack_coordinator_r_sum",
                    ):
                        ratio = pd.to_numeric(active[column], errors="coerce")
                        if ratio.isna().any() or not (
                            np.isfinite(ratio) & (ratio >= 0.0)
                        ).all():
                            raise ValueError(f"invalid optimization diagnostic: {column}")
                if attack in {"lie", "alie"}:
                    gamma = pd.to_numeric(
                        active["fit_attack_coordinator_gamma"], errors="coerce"
                    )
                    if gamma.isna().any() or not (gamma > 0).all():
                        raise ValueError("LIE degenerated to a zero perturbation")
                    sign_ratio = pd.to_numeric(
                        active["fit_attack_coordinator_sign_flippable_ratio"],
                        errors="coerce",
                    )
                    if sign_ratio.isna().any() or not (
                        np.isfinite(sign_ratio)
                        & (sign_ratio >= 0.0)
                        & (sign_ratio <= 1.0)
                    ).all():
                        raise ValueError("invalid LIE sign-flippable ratio")
            if attack == "dba":
                active_clients = pd.to_numeric(
                    active["fit_dba_active_clients"], errors="coerce"
                )
                scaled = pd.to_numeric(
                    active["fit_dba_scaled_clients"], errors="coerce"
                )
                selected_attackers = pd.to_numeric(
                    active["fit_selected_active_attackers"], errors="coerce"
                )
                plan_path = Path(str(spec.get("trial_plan_path", "")))
                if not plan_path.is_file():
                    raise ValueError("DBA validation requires a trial plan")
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                malicious_ids = sorted(
                    int(value) for value in plan.get("malicious_partition_ids", ())
                )
                if not malicious_ids:
                    raise ValueError("DBA trial plan has no malicious identities")
                fragment_count = max(1, int(spec.get("dba_trigger_num", 4)))
                fragment_by_client = {
                    client_id: rank % fragment_count
                    for rank, client_id in enumerate(malicious_ids)
                }
                if active_clients.isna().any() or selected_attackers.isna().any():
                    raise ValueError("DBA active-client evidence is incomplete")
                for row_index, row in active.iterrows():
                    try:
                        selected_ids = [
                            int(value)
                            for value in json.loads(
                                str(row["fit_planned_partition_ids_json"])
                            )
                        ]
                        observed_fragments = sorted(
                            int(value)
                            for value in json.loads(
                                str(row["fit_dba_fragment_indices_json"])
                            )
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            "DBA fragment assignment evidence is malformed"
                        ) from exc
                    expected_fragments = sorted(
                        fragment_by_client[client_id]
                        for client_id in selected_ids
                        if client_id in fragment_by_client
                    )
                    expected_count = int(round(float(selected_attackers.loc[row_index])))
                    if (
                        int(round(float(active_clients.loc[row_index])))
                        != expected_count
                        or len(observed_fragments) != expected_count
                        or observed_fragments != expected_fragments
                    ):
                        raise ValueError("DBA fragment assignment is incomplete")
                if bool(spec.get("dba_scale_update", True)) and not (
                    scaled == active_clients
                ).all():
                    raise ValueError("DBA scaling evidence is incomplete")
                required = {
                    "server_dba_full_trigger_asr",
                    "server_dba_local_asr_mean",
                    "server_dba_local_asr_min",
                    "server_dba_local_asr_max",
                }
                if required.difference(frame.columns):
                    raise ValueError("DBA full/local ASR metrics are missing")
        except (KeyError, OSError, ValueError) as exc:
            passed = False
            detail = str(exc)
        gates.append({
            "gate": f"attack_execution:{periodic_attack.run_id(dict(spec))}",
            "passed": passed,
            "observed": detail,
            "required": "complete",
        })
    return gates


_SERVER_EVAL_RE = re.compile(
    r"Server eval \| round=(?P<round>\d+) loss=(?P<loss>[^ ]+) acc=(?P<acc>[^\s]+)"
)


def _diverged_screening_rows(
    specs: Sequence[Mapping[str, Any]], summary: pd.DataFrame, output: Path
) -> pd.DataFrame:
    """Materialize explicit summary rows for expected non-finite termination."""
    rows: list[dict[str, Any]] = []
    existing = set(summary.get("run_id", pd.Series(dtype=str)).astype(str))
    for raw_spec in specs:
        spec = dict(raw_spec)
        identifier = periodic_attack.run_id(spec)
        if identifier in existing or str(spec.get("attack")) == "none":
            continue
        status_path = output / "status" / f"{identifier}.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if status.get("state") != "diverged":
            continue

        last_finite_accuracy = math.nan
        last_finite_round = math.nan
        log_path = output / "raw" / f"{identifier}.log"
        if log_path.is_file():
            for match in _SERVER_EVAL_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
                try:
                    loss = float(match.group("loss"))
                    accuracy = float(match.group("acc"))
                except ValueError:
                    continue
                if math.isfinite(loss) and math.isfinite(accuracy):
                    last_finite_round = int(match.group("round"))
                    last_finite_accuracy = accuracy

        row = {key: value for key, value in spec.items() if key != "custom_params"}
        row.update({
            "run_id": identifier,
            "numerical_divergence": True,
            "diverged_success": False,
            "collapsed_invalid": True,
            "divergence_round": int(status.get("divergence_round", status.get("last_round", 0))),
            "last_finite_round": last_finite_round,
            "last_finite_accuracy": last_finite_accuracy,
            "final_accuracy": math.nan,
            "min_accuracy": math.nan,
            "active_accuracy": math.nan,
            "divergence_error": str(status.get("error_summary", "")),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def select_screened_strengths(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Apply attack-specific FedAvg screening rules without consulting RTC."""
    order = {"weak": 0, "medium": 1, "strong": 2}
    metric_rules: dict[str, tuple[str, float]] = {
        "gaussian_noise": ("accuracy_drop", 0.03),
        "random_noise": ("accuracy_drop", 0.03),
        "lie": ("accuracy_drop", 0.03),
        "alie": ("accuracy_drop", 0.03),
        "min_max": ("accuracy_drop", 0.05),
        "min_sum": ("accuracy_drop", 0.05),
        "scaling_backdoor": ("participating_asr", 0.80),
        "dba": ("participating_asr", 0.80),
        "label_flip_targeted": ("participating_asr", 0.20),
        "label_flip_all_reverse": ("participating_asr", math.nan),
        "sign_flip": ("accuracy_drop", math.nan),
    }
    recommendations: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    frozen_attacks: dict[str, Any] = {}
    attacked = summary[
        (summary["attack"] != "none") & (summary["defense"] == "fedavg")
    ].copy()
    for attack in sorted(attacked["attack"].astype(str).unique()):
        rows = attacked[attacked["attack"].astype(str) == attack].copy()
        rows["_order"] = rows["strength_level"].map(order)
        rows = rows.sort_values("_order")
        metric, threshold = metric_rules.get(attack, ("accuracy_drop", 0.03))
        values = pd.to_numeric(
            rows.get(metric, pd.Series(math.nan, index=rows.index)),
            errors="coerce",
        )
        finite = values.notna() & values.map(math.isfinite)
        numerical_valid = ~rows.get(
            "numerical_divergence", pd.Series(False, index=rows.index)
        ).fillna(False).astype(bool)
        numerical_valid &= ~rows.get(
            "collapsed_invalid", pd.Series(False, index=rows.index)
        ).fillna(False).astype(bool)
        clean_accuracy = pd.to_numeric(
            rows.get("active_accuracy", pd.Series(1.0, index=rows.index)),
            errors="coerce",
        )
        minimum_accuracy = 0.60 if attack in {"dba", "scaling_backdoor"} else 0.0
        acceptable_accuracy = (
            clean_accuracy.notna()
            & clean_accuracy.map(math.isfinite)
            & (clean_accuracy >= minimum_accuracy)
        )
        targeted_objective = attack in {
            "dba",
            "scaling_backdoor",
            "label_flip_targeted",
            "label_flip_all_reverse",
        }
        asr_valid = (
            rows.get("asr_valid", pd.Series(False, index=rows.index))
            .fillna(False)
            .astype(bool)
            if targeted_objective
            else pd.Series(True, index=rows.index)
        )
        valid_configuration = numerical_valid & acceptable_accuracy & asr_valid

        accuracy_drop = pd.to_numeric(
            rows.get("accuracy_drop", pd.Series(math.nan, index=rows.index)),
            errors="coerce",
        )
        if attack in {"dba", "scaling_backdoor"}:
            valid_configuration &= accuracy_drop.notna()
            valid_configuration &= accuracy_drop.map(math.isfinite)
            valid_configuration &= accuracy_drop <= 0.10
        source_metric_missing = False
        if attack == "label_flip_targeted":
            source_drop = pd.to_numeric(
                rows.get(
                    "source_recall_drop",
                    pd.Series(math.nan, index=rows.index),
                ),
                errors="coerce",
            )
            valid_configuration &= source_drop.notna()
            valid_configuration &= source_drop.map(math.isfinite)
            valid_configuration &= source_drop > 0.0
            source_metric_missing = not bool(
                source_drop.notna().any() and source_drop.map(math.isfinite).any()
            )

        monotonic = bool(
            len(values) == 3
            and finite.all()
            and all(
                float(values.iloc[index + 1]) + 0.03
                >= float(values.iloc[index])
                for index in range(len(values) - 1)
            )
        )
        selected = None
        selection_reason = ""
        if attack == "sign_flip":
            candidate = rows[rows["strength_level"].astype(str) == "weak"]
            if not candidate.empty:
                candidate_index = candidate.index[0]
                if bool(valid_configuration.loc[candidate_index]):
                    selected = rows.loc[candidate_index]
                    selection_reason = (
                        "formal sign-flip baseline uses scale=1; amplified levels are stress tests"
                    )
        elif attack == "label_flip_all_reverse":
            candidate = rows[rows["strength_level"].astype(str) == "strong"]
            if not candidate.empty:
                candidate_index = candidate.index[0]
                if (
                    bool(valid_configuration.loc[candidate_index])
                    and bool(finite.loc[candidate_index])
                    and monotonic
                ):
                    selected = rows.loc[candidate_index]
                    selection_reason = (
                        "full deterministic reverse mapping with nondecreasing effect; "
                        "no universal effect threshold"
                    )
        else:
            eligible = rows[
                finite
                & valid_configuration
                & (values >= float(threshold))
            ]
            if attack == "random_noise":
                passing_levels = [level for level, group in rows.groupby("strength_level")
                                  if set(group.index).issubset(set(eligible.index))]
                eligible = eligible[eligible["strength_level"].isin(passing_levels)]
            if not eligible.empty:
                selected = eligible.iloc[0]
                selection_reason = (
                    f"weakest valid strength meeting attack-specific {metric} threshold"
                )

        passed = selected is not None
        selected_level = str(selected["strength_level"]) if passed else ""
        status_by_level = {
            str(level): ("VALID" if bool(valid) else "COLLAPSED/INVALID")
            for level, valid in zip(rows["strength_level"], valid_configuration)
        }
        value_by_level = {
            str(level): (
                float(value) if math.isfinite(float(value)) else math.nan
            )
            for level, value in zip(rows["strength_level"], values)
        }
        missing_metric = not bool(finite.any())
        reproduction_status = (
            "SELECTED"
            if passed
            else "METRIC_VALIDATION_REQUIRED"
            if missing_metric or source_metric_missing
            else "NO_VALID_EFFECTIVE_STRENGTH"
        )
        attack_metric_value = float(selected[metric]) if passed else math.nan
        clean_metric = (
            float(selected.get("active_accuracy", math.nan)) if passed else math.nan
        )
        secondary_metric = (
            "source_recall_drop" if attack == "label_flip_targeted"
            else "accuracy_drop" if attack in {"dba", "scaling_backdoor"}
            else ""
        )
        secondary_metric_value = (
            float(selected.get(secondary_metric, math.nan))
            if passed and secondary_metric
            else math.nan
        )
        recommendations.append({
            "attack": attack,
            "metric": metric,
            "threshold": threshold,
            "weak": value_by_level.get("weak", math.nan),
            "medium": value_by_level.get("medium", math.nan),
            "strong": value_by_level.get("strong", math.nan),
            "monotonic_with_tolerance": monotonic,
            "weak_status": status_by_level.get("weak", "MISSING"),
            "medium_status": status_by_level.get("medium", "MISSING"),
            "strong_status": status_by_level.get("strong", "MISSING"),
            "invalid_strengths": ",".join(
                level for level in ("weak", "medium", "strong")
                if status_by_level.get(level) == "COLLAPSED/INVALID"
            ),
            "selected_strength": selected_level,
            "selected_formal_strength": selected_level,
            "reproduction_status": reproduction_status,
            "attack_metric_value": attack_metric_value,
            "secondary_metric": secondary_metric,
            "secondary_metric_value": secondary_metric_value,
            "clean_metric_name": "active_accuracy",
            "clean_metric": clean_metric,
            "numerical_validity": (
                "VALID"
                if passed
                else "VALID_CONFIGURATIONS_PRESENT"
                if bool(valid_configuration.any())
                else "NO_VALID_SELECTED_CONFIGURATION"
            ),
            "selection_reason": selection_reason or (
                "rerun required or no valid configuration met the attack-specific gate"
            ),
            "passed": passed,
        })
        gates.append({
            "gate": f"attack_strength:{attack}",
            "passed": passed,
            "observed": (
                f"{metric}={values.tolist()} selected={selected_level or '<none>'} "
                f"valid={valid_configuration.tolist()}"
            ),
            "required": (
                "attack-specific effectiveness plus finite model/update/logit/loss, "
                f"clean accuracy>={minimum_accuracy:.2f}; monotonicity is diagnostic only"
            ),
        })
        if passed:
            frozen_attacks[attack] = {
                "strength_level": selected_level,
                "parameters": dict(STRENGTH_GRID[attack][selected_level]),
                "selection_metric": metric,
                "selection_value": attack_metric_value,
                "selection_threshold": threshold,
            }
    payload = {
        "schema_version": "RTCByzantineAttackFreezeV1",
        "implementation_source_sha256": attack_source_hash(),
        "selection_policy": (
            "attack-specific FedAvg-only screening; finite utility collapse remains measurable; "
            "numerical termination is not a completed attack outcome; monotonicity is diagnostic only"
        ),
        "attacks": frozen_attacks,
    }
    return pd.DataFrame(recommendations), payload, gates


def annotate_screening_outcomes(
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach configuration-level validity/effect labels to screen results."""
    result = summary.copy()
    result["attack_metric"] = ""
    result["attack_metric_value"] = math.nan
    result["numerical_validity"] = "NOT_SCREENED"
    result["attack_effect_too_weak"] = False
    result["successful_attack"] = False
    result["reproduction_status"] = "NOT_SCREENED"
    lookup = recommendations.set_index("attack")
    for index, row in result.iterrows():
        attack = str(row.get("attack"))
        if attack == "none" or attack not in lookup.index:
            continue
        recommendation = lookup.loc[attack]
        metric = str(recommendation["metric"])
        value = pd.to_numeric(row.get(metric, math.nan), errors="coerce")
        level = str(row.get("strength_level", ""))
        status = str(recommendation.get(f"{level}_status", "MISSING"))
        valid = (status == "VALID" if 'execution_state' not in row else
                 row['execution_state'] == 'completed' and not bool(row.get('collapsed_invalid', False)))
        threshold = pd.to_numeric(recommendation.get("threshold", math.nan), errors="coerce")
        if attack == "sign_flip":
            effective = level == "weak"
        elif attack == "label_flip_all_reverse":
            effective = level == "strong"
        else:
            effective = bool(
                np.isfinite(value)
                and np.isfinite(threshold)
                and float(value) >= float(threshold)
            )
        successful = bool(valid and effective)
        result.at[index, "attack_metric"] = metric
        result.at[index, "attack_metric_value"] = value
        result.at[index, "numerical_validity"] = (
            "VALID" if valid else "COLLAPSED/INVALID"
        )
        result.at[index, "attack_effect_too_weak"] = bool(valid and not effective)
        result.at[index, "successful_attack"] = successful
        result.at[index, "reproduction_status"] = (
            "VALID_EFFECTIVE"
            if successful
            else "VALID_BUT_TOO_WEAK"
            if valid
            else "COLLAPSED/INVALID"
        )
    return result


def run(args) -> pd.DataFrame:
    args.pairing_mode = "strict"
    args.sampling_protocol = "principal_uniform"
    screening = bool(getattr(args, "byzantine_screening", False))
    random_screen = screening and 'random_noise' in _csv(getattr(args, 'attacks', ''), CANONICAL_ATTACKS)
    args.rounds = (60 if random_screen or not screening else 25) if args.rounds is None else int(args.rounds)
    if screening:
        args.attack_start_round = 1
        args.attack_end_round = -1
    args.output = args.output or "logs/rtc_v3_byzantine"
    args.smoke = bool(getattr(args, "smoke", False))
    if (
        not screening
        and not args.smoke
        and not bool(getattr(args, "dry_run", False))
        and not str(getattr(args, "byzantine_attack_freeze", "") or "").strip()
    ):
        raise ValueError(
            "Formal rtc-byzantine runs require --byzantine-attack-freeze "
            "from the completed FedAvg-only strength screen"
        )
    if args.smoke:
        args.smoke_rounds = 3
        args.smoke_num_clients = int(args.num_clients)
        args.smoke_all_clients = False
    output = Path(args.output)
    matrix = (
        build_screening_matrix(args)
        if bool(getattr(args, "byzantine_screening", False))
        else build_matrix(args)
    )
    matrix = attach_trial_plans(matrix, args, output)
    clean = [] if args.skip_clean else paired_fedavg_clean_specs(matrix)
    for clean_spec in clean:
        clean_spec["counterfactual_attack_version"] = clean_spec.get(
            "attack_version", ""
        )
        clean_spec["counterfactual_attack_contract_hash"] = clean_spec.get(
            "attack_contract_hash", ""
        )
        clean_spec["counterfactual_attack_implementation_hash"] = clean_spec.get(
            "attack_implementation_hash", ""
        )
        clean_parameters = {
            key: clean_spec[key]
            for key in _ATTACK_PARAMETER_KEYS
            if key in clean_spec
        }
        clean_runtime = _runtime_attack_config(
            "none",
            clean_parameters,
            malicious_fraction=0.0,
            attack_start_round=int(clean_spec["attack_start_round"]),
            attack_end_round=int(clean_spec["attack_end_round"]),
        )
        clean_contract = attack_contract_payload("none", vars(clean_runtime))
        clean_spec["attack_version"] = get_attack_spec("none").version
        clean_spec["attack_contract_hash"] = clean_contract["contract_hash"]
        clean_spec["attack_implementation_hash"] = clean_contract[
            "implementation_hash"
        ]
    specs = periodic_attack._deduplicate(
        clean if args.clean_only else _interleave_plan_clean(matrix, clean)
    )
    periodic_attack.write_experiment_manifest(specs, output)
    _write_matrix(specs, output / "experiment_matrix.csv")
    if args.dry_run:
        return pd.DataFrame()

    rounds_dir = output / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    summary = periodic_attack._run_specs(specs, args, output, rounds_dir)
    diverged = _diverged_screening_rows(specs, summary, output)
    if not diverged.empty:
        summary = pd.concat([summary, diverged], ignore_index=True, sort=False)
    summary = periodic_attack.add_accuracy_drop_auc(
        periodic_attack.add_accuracy_drop(summary), rounds_dir
    )
    # Preserve the complete screening evidence even when a strength or
    # execution gate rejects parameter freezing below.
    _atomic_write_csv(summary, output / "byzantine_attack_runs.csv")
    complete = summary[summary.get("execution_state", pd.Series("completed", index=summary.index)).eq("completed")]
    expected_ids = set(complete["run_id"].astype(str))
    gates = periodic_attack.validate_sampling_manifests(
        rounds_dir, output / "raw", expected_ids=expected_ids
    )
    gates.extend(validate_attack_execution(specs, rounds_dir))
    screening_recommendations = pd.DataFrame()
    freeze_payload: dict[str, Any] | None = None
    if screening:
        screening_recommendations, freeze_payload, screening_gates = (
            select_screened_strengths(summary)
        )
        summary = annotate_screening_outcomes(summary, screening_recommendations)
        _atomic_write_csv(summary, output / "byzantine_attack_runs.csv")
        gates.extend(screening_gates)
        if "random_noise" in freeze_payload.get("attacks", {}):
            from experiments.rtc_v3.random_noise_validation import attach_freeze_evidence
            try:
                attach_freeze_evidence(freeze_payload, summary, specs, output)
            except ValueError as exc:
                freeze_payload["attacks"].pop("random_noise", None)
                gates.append({"gate": "random_noise_freeze_evidence", "passed": False,
                              "observed": str(exc), "required": "two seeds with paired effective finite runs"})
    coverage = summary[[column for column in ("run_id", "attack", "defense", "seed", "execution_state", "numerical_state", "failure_round") if column in summary]].copy()
    _atomic_write_csv(coverage, output / "execution_coverage.csv")
    periodic_attack.write_quality_gates(gates, output / "quality_gates.csv")
    if screening:
        _atomic_write_csv(
            screening_recommendations, output / "attack_strength_screening.csv"
        )
    failed = [gate for gate in gates if not gate["passed"]]
    if failed:
        raise RuntimeError(
            "Byzantine quality gates failed: "
            + "; ".join(str(gate["gate"]) for gate in failed)
        )
    if screening:
        assert freeze_payload is not None
        periodic_attack._atomic_write_json(
            freeze_payload, output / "attack_freeze.json"
        )
    _atomic_write_csv(
        periodic_attack.bootstrap_summary(summary),
        output / "byzantine_attack_summary.csv",
    )
    _atomic_write_csv(
        periodic_attack.statistical_tests(summary),
        output / "statistical_tests.csv",
    )
    periodic_attack.generate_plots(summary, rounds_dir, output / "plots")
    return summary
