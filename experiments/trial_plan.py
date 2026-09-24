"""Defense-independent, hash-bound trial plans for paired FL experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "trial-plan-v1"

# A TrialPlan owns only common-random-number factors.  Attack strength,
# implementation/version hashes, optimizer parameters, and defense settings
# must never perturb participant selection, malicious identities, data, model
# initialization, or client-local random streams.
PLAN_CONDITION_KEYS = (
    "attack",
    "period",
    "on_rounds",
    "off_rounds",
    "malicious_fraction",
    "seed",
    "attack_start_round",
    "attack_end_round",
    "partition",
    "dirichlet_alpha",
    "participation_rate",
)

COUNTERFACTUAL_ATTACK_PARAMETER_KEYS = frozenset({
    "aggregation_aware_scaling",
    "boost_factor",
    "coordinated_attack_knowledge",
    "dba_pattern_mode",
    "dba_scale_update",
    "dba_trigger_value_mode",
    "gaussian_noise_mean",
    "gaussian_noise_std",
    "label_flip_poison_fraction",
    "label_flip_source_label",
    "label_flip_target_label",
    "lie_z",
    "optimization_gamma_fraction",
    "optimization_gamma_init",
    "optimization_max_iterations",
    "optimization_perturbation",
    "optimization_tolerance",
    "poison_fraction",
    "random_noise_distribution",
    "random_noise_scale",
    "replacement_gain",
    "sign_flip_scale",
})


class TrialPlanError(RuntimeError):
    """Raised when a strict trial plan is invalid or cannot be consumed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable uint32 seed without relying on Python's randomized hash."""
    payload = {"base_seed": int(base_seed), "parts": list(parts)}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def logical_id_key(value: Any) -> tuple[int, Any]:
    raw = str(value)
    try:
        return (0, int(raw))
    except (TypeError, ValueError):
        return (1, raw)


def _plan_payload_without_hash(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result.pop("trial_plan_hash", None)
    return result


@dataclass(frozen=True)
class TrialPlanV1:
    payload: Mapping[str, Any]

    @property
    def trial_plan_hash(self) -> str:
        return str(self.payload["trial_plan_hash"])

    @property
    def pairing_group_id(self) -> str:
        return str(self.payload["pairing_group_id"])

    @property
    def sampling_protocol(self) -> str:
        return str(self.payload["sampling_protocol"])

    @property
    def malicious_partition_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["malicious_partition_ids"])

    def round(self, server_round: int) -> Mapping[str, Any]:
        rounds = self.payload.get("rounds", [])
        index = int(server_round) - 1
        if index < 0 or index >= len(rounds):
            raise TrialPlanError(
                f"Trial plan {self.trial_plan_hash} has no round {server_round}"
            )
        round_plan = rounds[index]
        if int(round_plan.get("round", -1)) != int(server_round):
            raise TrialPlanError(f"Trial plan round index mismatch at {server_round}")
        return round_plan

    @classmethod
    def load(cls, path: str | Path) -> "TrialPlanV1":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise TrialPlanError(f"Strict trial plan not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise TrialPlanError(
                f"Unsupported trial-plan schema: {payload.get('schema_version')!r}"
            )
        observed = str(payload.get("trial_plan_hash", ""))
        expected = sha256_json(_plan_payload_without_hash(payload))
        if not observed or observed != expected:
            raise TrialPlanError(
                f"Trial-plan hash mismatch: observed={observed or '<missing>'} expected={expected}"
            )
        return cls(payload=payload)


def _condition_payload(spec: Mapping[str, Any], args: Any) -> Dict[str, Any]:
    """Return the defense-free condition that owns a plan."""
    condition = {
        key: spec[key]
        for key in PLAN_CONDITION_KEYS
        if key in spec
    }
    # Do not change historical plan hashes for specs without an explicit model.
    for key in ("dataset", "architecture"):
        if key in spec:
            condition[key] = str(spec[key])
    smoke = bool(getattr(args, "smoke", False))
    smoke_all_clients = smoke and bool(
        getattr(args, "smoke_all_clients", True)
    )
    num_clients = (
        int(getattr(args, "smoke_num_clients", 10))
        if bool(getattr(args, "smoke", False))
        else int(getattr(args, "num_clients", 20))
    )
    condition.update({
        "num_clients": num_clients,
        "rounds": (
            int(getattr(args, "smoke_rounds", 6))
            if bool(getattr(args, "smoke", False))
            else int(args.rounds)
        ),
        "clients_per_round": (
            num_clients if smoke_all_clients
            else max(1, min(num_clients, int(np.ceil(
                num_clients * float(spec.get("participation_rate", 0.5))
            ))))
        ),
        "sampling_protocol": str(getattr(args, "sampling_protocol", "principal_uniform")),
        "max_client_samples": int(getattr(args, "max_client_samples", 0)),
    })
    return condition


def _principal_map_for_group(
    specs: Sequence[Mapping[str, Any]], num_clients: int,
) -> Dict[str, str]:
    identity = {str(client_id): str(client_id) for client_id in range(num_clients)}
    observed: list[Dict[str, str]] = []
    for spec in specs:
        custom = spec.get("custom_params", {})
        mapping = custom.get("principal_map") if isinstance(custom, Mapping) else None
        if mapping:
            normalized = {str(key): str(value) for key, value in mapping.items()}
            missing = sorted(set(identity).difference(normalized), key=logical_id_key)
            if missing:
                raise TrialPlanError(f"Principal map is missing partitions: {missing}")
            observed.append(normalized)
    if observed and any(mapping != observed[0] for mapping in observed[1:]):
        raise TrialPlanError("Defenses in one paired condition have inconsistent principal maps")
    return observed[0] if observed else identity


def generate_trial_plan(
    condition: Mapping[str, Any],
    principal_map: Mapping[str, str],
) -> Dict[str, Any]:
    num_clients = int(condition["num_clients"])
    num_rounds = int(condition["rounds"])
    participation_rate = float(condition.get("participation_rate", 0.5))
    clients_per_round = int(condition.get(
        "clients_per_round",
        max(1, min(num_clients, int(np.ceil(num_clients * participation_rate)))),
    ))
    protocol = str(condition.get("sampling_protocol", "principal_uniform"))
    if protocol not in {"principal_uniform", "endpoint_uniform"}:
        raise TrialPlanError(f"Unsupported sampling protocol: {protocol}")

    partition_ids = [str(value) for value in range(num_clients)]
    normalized_map = {str(key): str(value) for key, value in principal_map.items()}
    missing = sorted(set(partition_ids).difference(normalized_map), key=logical_id_key)
    if missing:
        raise TrialPlanError(f"Principal map is missing partitions: {missing}")

    group_payload = dict(condition)
    group_payload["principal_map_sha256"] = sha256_json(normalized_map)
    pairing_group_id = sha256_json(group_payload)[:20]
    seed = int(condition["seed"])
    fraction = float(condition.get("malicious_fraction", 0.0))
    malicious_count = 0 if fraction <= 0 else max(1, int(num_clients * fraction))
    malicious_rng = np.random.default_rng(
        derive_seed(seed, pairing_group_id, "malicious-identities")
    )
    malicious_ids = sorted(
        (
            str(int(value))
            for value in malicious_rng.choice(
                num_clients, size=malicious_count, replace=False
            )
        ),
        key=logical_id_key,
    )

    groups: Dict[str, list[str]] = {}
    for partition_id in partition_ids:
        groups.setdefault(normalized_map[partition_id], []).append(partition_id)
    principals = sorted(groups, key=logical_id_key)
    if protocol == "principal_uniform" and clients_per_round > len(principals):
        raise TrialPlanError(
            f"principal_uniform needs {clients_per_round} distinct principals, only {len(principals)} exist"
        )

    rounds = []
    for server_round in range(1, num_rounds + 1):
        rng = np.random.default_rng(
            derive_seed(seed, pairing_group_id, "participants", server_round)
        )
        if protocol == "principal_uniform":
            indices = sorted(int(value) for value in rng.choice(
                len(principals), size=clients_per_round, replace=False
            ))
            selected_principals = [principals[index] for index in indices]
            selected_ids = []
            for principal in selected_principals:
                endpoints = sorted(groups[principal], key=logical_id_key)
                selected_ids.append(endpoints[int(rng.integers(0, len(endpoints)))])
        else:
            endpoints = sorted(partition_ids, key=logical_id_key)
            indices = sorted(int(value) for value in rng.choice(
                len(endpoints), size=clients_per_round, replace=False
            ))
            selected_ids = [endpoints[index] for index in indices]
            selected_principals = [normalized_map[value] for value in selected_ids]

        fit_seeds = {
            partition_id: derive_seed(
                seed, pairing_group_id, "fit", server_round, partition_id
            )
            for partition_id in selected_ids
        }
        attack_seeds = {
            partition_id: derive_seed(
                seed, pairing_group_id, "attack", server_round, partition_id
            )
            for partition_id in selected_ids
        }
        evaluate_seeds = {
            partition_id: derive_seed(
                seed, pairing_group_id, "evaluate", server_round, partition_id
            )
            for partition_id in selected_ids
        }
        rounds.append({
            "round": server_round,
            "partition_ids": selected_ids,
            "principal_ids": selected_principals,
            "fit_seeds": fit_seeds,
            "attack_seeds": attack_seeds,
            "evaluate_seeds": evaluate_seeds,
            "fit_seed_digest": sha256_json(fit_seeds),
            "attack_seed_digest": sha256_json(attack_seeds),
            "evaluate_seed_digest": sha256_json(evaluate_seeds),
        })

    data_config = {
        "dataset": str(condition.get("dataset", "cifar10")),
        "seed": seed,
        "partition": condition.get("partition"),
        "dirichlet_alpha": condition.get("dirichlet_alpha"),
        "num_clients": num_clients,
        "max_client_samples": condition.get("max_client_samples", 0),
    }
    initial_model_config = {
        "seed": seed, "dataset": str(condition.get("dataset", "cifar10")),
        "architecture": str(condition.get("architecture", "resnet18"))
    }
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pairing_group_id": pairing_group_id,
        "base_seed": seed,
        "condition": dict(condition),
        "sampling_protocol": protocol,
        "num_clients": num_clients,
        "clients_per_round": clients_per_round,
        "principal_map": normalized_map,
        "malicious_partition_ids": malicious_ids,
        "data_config": data_config,
        "data_config_sha256": sha256_json(data_config),
        "initial_model_config": initial_model_config,
        "initial_model_config_sha256": sha256_json(initial_model_config),
        "rounds": rounds,
    }
    payload["trial_plan_hash"] = sha256_json(payload)
    return payload


def attach_trial_plans(
    specs: Sequence[Mapping[str, Any]], args: Any, output: str | Path,
) -> list[Dict[str, Any]]:
    """Generate one plan per attack condition and attach it across defenses."""
    rows = [dict(spec) for spec in specs]
    if str(getattr(args, "pairing_mode", "strict")) != "strict":
        for row in rows:
            row.update({
                "pairing_mode": "legacy",
                "sampling_protocol": str(getattr(args, "sampling_protocol", "endpoint_uniform")),
            })
        return rows

    groups: Dict[str, list[Dict[str, Any]]] = {}
    conditions: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        condition = _condition_payload(row, args)
        key = canonical_json(condition)
        groups.setdefault(key, []).append(row)
        conditions[key] = condition

    plan_dir = Path(output) / "trial_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    planned: list[Dict[str, Any]] = []
    for key, group in groups.items():
        condition = conditions[key]
        principal_map = _principal_map_for_group(group, int(condition["num_clients"]))
        payload = generate_trial_plan(condition, principal_map)
        path = (plan_dir / f"{payload['pairing_group_id']}.json").resolve()
        if path.exists():
            existing = TrialPlanV1.load(path)
            if existing.trial_plan_hash != payload["trial_plan_hash"]:
                raise TrialPlanError(f"Existing trial plan conflicts with generated plan: {path}")
        else:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        for source in group:
            row = dict(source)
            row.update({
                "pairing_mode": "strict",
                "sampling_protocol": payload["sampling_protocol"],
                "pairing_group_id": payload["pairing_group_id"],
                "trial_plan_hash": payload["trial_plan_hash"],
                "trial_plan_path": str(path),
            })
            custom = dict(row.get("custom_params", {}))
            if "principal_first_sampling_verified" in custom:
                custom["principal_first_sampling_verified"] = (
                    payload["sampling_protocol"] == "principal_uniform"
                    and all(
                        len(round_plan["principal_ids"])
                        == len(set(round_plan["principal_ids"]))
                        for round_plan in payload["rounds"]
                    )
                )
            row["custom_params"] = custom
            planned.append(row)
    return planned


def paired_fedavg_clean_specs(planned_attacked: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Create one FedAvg clean counterfactual for each attack-owned plan."""
    rows: Dict[str, Dict[str, Any]] = {}
    for source in planned_attacked:
        if source.get("attack") == "none":
            continue
        group_id = str(source["pairing_group_id"])
        if group_id in rows:
            continue
        row = dict(source)
        for key in COUNTERFACTUAL_ATTACK_PARAMETER_KEYS:
            row.pop(key, None)
        row.update({
            "attack_group": "clean",
            "attack": "none",
            "period": "clean",
            "on_rounds": 1,
            "off_rounds": 0,
            "malicious_fraction": 0.0,
            "defense": "fedavg",
            "defense_type": "none",
            "custom_params": {},
            "counterfactual_for": group_id,
        })
        rows[group_id] = row
    return list(rows.values())
