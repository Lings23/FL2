"""Defense-independent server-side coordination for colluding attacks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from config.config_loader import AttackConfig
from defenses.defense_base import UpdateList
from attacks.spec import COORDINATED_ATTACKS, attack_contract_payload


@dataclass(frozen=True)
class CoordinatedAttackResult:
    updates: UpdateList
    metrics: Mapping[str, float | int | str | bool]


def _floating_layout(
    global_params: Sequence[np.ndarray],
    indices: set[int] | frozenset[int] | None = None,
) -> list[tuple[int, tuple[int, ...], int]]:
    layout: list[tuple[int, tuple[int, ...], int]] = []
    for index, parameter in enumerate(global_params):
        if (
            np.issubdtype(parameter.dtype, np.floating)
            and (indices is None or index in indices)
        ):
            layout.append((index, tuple(parameter.shape), int(parameter.size)))
    return layout


def _flatten_delta(
    parameters: Sequence[np.ndarray],
    global_params: Sequence[np.ndarray],
    indices: set[int] | frozenset[int] | None = None,
) -> np.ndarray:
    pieces = [
        (np.asarray(parameters[index], dtype=np.float64) - np.asarray(global_params[index], dtype=np.float64)).reshape(-1)
        for index, _, _ in _floating_layout(global_params, indices)
    ]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


def _replace_delta(
    parameters: Sequence[np.ndarray],
    global_params: Sequence[np.ndarray],
    delta: np.ndarray,
    indices: set[int] | frozenset[int] | None = None,
) -> list[np.ndarray]:
    result = [np.asarray(value).copy() for value in parameters]
    offset = 0
    for index, shape, size in _floating_layout(global_params, indices):
        piece = delta[offset : offset + size].reshape(shape)
        value = np.asarray(global_params[index], dtype=np.float64) + piece
        result[index] = value.astype(global_params[index].dtype, copy=False)
        offset += size
    if offset != int(delta.size):
        raise ValueError("coordinated attack delta does not match model layout")
    return result


def _pairwise_squared(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise FloatingPointError("Min-Max reference updates must be finite 2-D vectors")
    # Direct differences avoid cancellation in ||x||^2 + ||y||^2 - 2<x,y>.
    # Compute one pair at a time: a full n x n x D tensor is several GB for
    # ResNet-sized updates even when n is only ten.
    distances = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for left in range(values.shape[0]):
                for right in range(left + 1, values.shape[0]):
                    difference = values[left] - values[right]
                    distance = float(np.dot(difference, difference))
                    if not np.isfinite(distance):
                        raise FloatingPointError
                    distances[left, right] = distance
                    distances[right, left] = distance
        except FloatingPointError as exc:
            raise FloatingPointError("Min-Max pairwise distance overflowed") from exc
    if not np.all(np.isfinite(distances)):
        raise FloatingPointError("Min-Max pairwise distances must be finite")
    return np.maximum(distances, 0.0)


def _perturbation(reference: np.ndarray, vectors: np.ndarray, kind: str) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(vectors)):
        raise FloatingPointError("Min-Max direction inputs must be finite")
    key = str(kind).lower()
    if key == "inverse_std":
        return -np.std(vectors, axis=0, ddof=0)
    if key == "inverse_unit":
        norm = float(np.linalg.norm(reference))
        if not np.isfinite(norm):
            raise FloatingPointError("Min-Max inverse-unit norm is non-finite")
        return -reference / max(norm, 1e-12)
    if key == "inverse_sign":
        return -np.sign(reference)
    raise ValueError(f"Unsupported coordinated perturbation: {kind!r}")


def _maximize_gamma(
    feasible,
    *,
    initial: float,
    tolerance: float,
    max_iterations: int,
    gamma_max: float,
) -> float:
    initial = max(float(initial), float(tolerance))
    tolerance = float(tolerance)
    gamma_max = float(gamma_max)
    if (
        not np.isfinite(initial)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or not np.isfinite(gamma_max)
        or gamma_max <= 0.0
    ):
        raise ValueError("Min-Max gamma search parameters must be finite and positive")
    initial = min(initial, gamma_max)
    low = 0.0
    high = initial
    iterations = 0
    while feasible(high) and iterations < max_iterations and high < gamma_max:
        low = high
        high = min(high * 2.0, gamma_max)
        iterations += 1
    while high - low > tolerance and iterations < max_iterations:
        middle = (low + high) / 2.0
        if feasible(middle):
            low = middle
        else:
            high = middle
        iterations += 1
    if not np.isfinite(low):
        raise FloatingPointError("Min-Max gamma search produced a non-finite value")
    return float(low)


class AttackCoordinator:
    """Transform colluding malicious updates before any defense sees them."""

    def __init__(
        self,
        config: AttackConfig,
        *,
        num_clients: int,
        scalable_parameter_indices: set[int] | None = None,
    ):
        self.config = config
        self.num_clients = int(num_clients)
        self.scalable_parameter_indices = (
            None
            if scalable_parameter_indices is None
            else frozenset(int(index) for index in scalable_parameter_indices)
        )
        self.contract = attack_contract_payload(
            config.type, vars(config)
        ) if str(config.type).lower() in COORDINATED_ATTACKS else None

    def transform(
        self,
        *,
        server_round: int,
        updates: UpdateList,
        global_params: Sequence[np.ndarray],
        client_ids: Sequence[str],
        malicious: Sequence[bool],
        attack_active: Sequence[bool],
    ) -> CoordinatedAttackResult:
        attack = str(self.config.type).lower()
        active_indices = [
            index for index, (is_malicious, active) in enumerate(zip(malicious, attack_active))
            if bool(is_malicious) and bool(active)
        ]
        base_metrics: dict[str, float | int | str | bool] = {
            "attack_coordinator_applied": False,
            "attack_coordinator_active_clients": len(active_indices),
            "attack_scaling_applied": False,
        }
        if not active_indices:
            return CoordinatedAttackResult(list(updates), base_metrics)

        if not all(
            np.all(np.isfinite(np.asarray(parameter, dtype=np.float64)))
            for parameter in global_params
            if np.issubdtype(np.asarray(parameter).dtype, np.floating)
        ):
            raise FloatingPointError(f"{attack} global parameters are non-finite")

        scaling_requested = bool(
            getattr(self.config, "aggregation_aware_scaling", False)
        ) and attack in {"dba", "model_replacement", "scaling_backdoor"}
        if attack == "dba" and not bool(getattr(self.config, "dba_scale_update", True)):
            scaling_requested = False
        if scaling_requested:
            sample_counts = np.asarray(
                [float(num_examples) for _, num_examples in updates],
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(sample_counts))
                or np.any(sample_counts < 0.0)
                or float(sample_counts.sum()) <= 0.0
            ):
                raise ValueError(
                    "aggregation-aware scaling requires finite non-negative sample counts"
                )
            malicious_weight_share = float(
                sample_counts[active_indices].sum() / sample_counts.sum()
            )
            requested_gain = float(getattr(self.config, "replacement_gain", 1.0))
            if (
                malicious_weight_share <= 0.0
                or not np.isfinite(requested_gain)
                or requested_gain <= 0.0
            ):
                raise FloatingPointError(
                    "aggregation-aware scaling produced an invalid coefficient"
                )
            client_multiplier = requested_gain / malicious_weight_share
            if not np.isfinite(client_multiplier):
                raise FloatingPointError("aggregation-aware multiplier is non-finite")
            transformed = list(updates)
            raw_delta_squared_norm = 0.0
            scaled_delta_squared_norm = 0.0
            for index in active_indices:
                parameters, num_examples = updates[index]
                raw_delta = _flatten_delta(
                    parameters,
                    global_params,
                    self.scalable_parameter_indices,
                )
                scaled_delta = client_multiplier * raw_delta
                if not np.all(np.isfinite(scaled_delta)):
                    raise FloatingPointError(
                        f"{attack} produced a non-finite aggregation-compensated delta"
                    )
                raw_delta_squared_norm += float(np.dot(raw_delta, raw_delta))
                scaled_delta_squared_norm += float(
                    np.dot(scaled_delta, scaled_delta)
                )
                transformed[index] = (
                    _replace_delta(
                        parameters,
                        global_params,
                        scaled_delta,
                        self.scalable_parameter_indices,
                    ),
                    num_examples,
                )
                if not all(
                    not np.issubdtype(value.dtype, np.floating)
                    or np.all(np.isfinite(value))
                    for value in transformed[index][0]
                ):
                    raise FloatingPointError(f"{attack} overflowed after dtype conversion")
            updates = transformed
            base_metrics.update({
                "attack_scaling_applied": True,
                "attack_scaling_mode": "fedavg_weight_compensated_model_delta",
                "attack_scaling_malicious_weight_share": malicious_weight_share,
                "attack_scaling_requested_gain": requested_gain,
                "attack_scaling_client_multiplier": client_multiplier,
                "attack_scaling_effective_gain": malicious_weight_share * client_multiplier,
                "attack_scaling_raw_trainable_delta_norm": math.sqrt(
                    raw_delta_squared_norm
                ),
                "attack_scaling_scaled_trainable_delta_norm": math.sqrt(
                    scaled_delta_squared_norm
                ),
            })

        if attack not in COORDINATED_ATTACKS:
            return CoordinatedAttackResult(list(updates), base_metrics)

        raw = np.stack([
            _flatten_delta(
                parameters,
                global_params,
                self.scalable_parameter_indices,
            )
            for parameters, _ in updates
        ])
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("Min-Max client deltas are non-finite")
        benign_indices = [index for index in range(len(updates)) if index not in active_indices]
        knowledge = str(getattr(self.config, "coordinated_attack_knowledge", "all_updates"))
        if knowledge == "all_updates":
            reference_indices = benign_indices
        elif knowledge == "malicious_only":
            reference_indices = active_indices
        else:
            raise ValueError(
                "coordinated_attack_knowledge must be all_updates or malicious_only"
            )
        if len(reference_indices) < 2:
            raise RuntimeError(
                f"{attack} needs at least two reference updates, got {len(reference_indices)}"
            )
        reference_vectors = raw[reference_indices]
        mean = np.mean(reference_vectors, axis=0)
        crafted: np.ndarray
        gamma = 0.0
        constraint_value = 0.0
        constraint_limit = 0.0
        sign_flippable_ratio: float | None = None
        r_max: float | None = None
        r_sum: float | None = None

        if attack in {"lie", "alie"}:
            std = np.std(reference_vectors, axis=0, ddof=0)
            configured_z = float(getattr(self.config, "lie_z", 0.0))
            if configured_z > 0.0:
                z_value = configured_z
            else:
                n = len(updates)
                m = len(active_indices)
                supporters = max(0, math.floor(n / 2 + 1) - m)
                probability = (n - m - supporters) / max(1, n - m)
                probability = min(max(probability, 1e-6), 1.0 - 1e-6)
                z_value = max(0.0, NormalDist().inv_cdf(probability))
            # The original prevention attack places all corrupt workers on the
            # lower side of the coordinate distribution.
            crafted = mean - z_value * std
            gamma = z_value
            # Fraction of trainable coordinates whose sign is changed by the
            # published LIE update relative to the reference mean.  Zero-valued
            # reference coordinates count when LIE moves them away from zero.
            sign_flippable_ratio = (
                float(np.mean(np.sign(crafted) != np.sign(mean)))
                if mean.size
                else 0.0
            )
        else:
            distances = _pairwise_squared(reference_vectors)
            direction = _perturbation(
                mean,
                reference_vectors,
                getattr(self.config, "optimization_perturbation", "inverse_sign"),
            )
            # For candidate = mean + gamma * direction, every squared
            # distance is a quadratic in gamma.  Precomputing its coefficients
            # preserves the published Min-Max/Min-Sum constraints exactly while
            # avoiding a full ResNet-sized vector scan at every bisection step.
            offsets = mean[None, :] - reference_vectors
            distance_base = np.einsum("ij,ij->i", offsets, offsets)
            distance_linear = 2.0 * (offsets @ direction)
            distance_quadratic = float(np.dot(direction, direction))
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm):
                raise FloatingPointError("Min-Max direction norm is non-finite")

            def candidate_distances(value: float) -> np.ndarray:
                return (
                    distance_base
                    + value * distance_linear
                    + (value * value) * distance_quadratic
                )

            initial = float(getattr(self.config, "optimization_gamma_init", 1e-3))
            tolerance = float(getattr(self.config, "optimization_tolerance", 1e-6))
            max_iterations = int(getattr(self.config, "optimization_max_iterations", 64))
            gamma_max = float(getattr(self.config, "optimization_gamma_max", 1.0e6))
            if attack == "min_max":
                constraint_limit = float(np.max(distances))

                def feasible(value: float) -> bool:
                    observed = candidate_distances(value)
                    return float(np.max(observed)) <= constraint_limit + 1e-10

            else:
                constraint_limit = float(np.max(np.sum(distances, axis=1)))

                def feasible(value: float) -> bool:
                    observed = candidate_distances(value)
                    return float(np.sum(observed)) <= constraint_limit + 1e-10

            gamma = _maximize_gamma(
                feasible,
                initial=initial,
                tolerance=tolerance,
                max_iterations=max_iterations,
                gamma_max=gamma_max,
            )
            gamma *= float(
                getattr(self.config, "optimization_gamma_fraction", 1.0)
            )
            crafted = mean + gamma * direction
            observed = candidate_distances(gamma)
            constraint_value = (
                float(np.max(observed)) if attack == "min_max" else float(np.sum(observed))
            )
            max_observed = float(np.max(observed))
            max_pairwise = float(np.max(distances))
            sum_observed = float(np.sum(observed))
            max_pairwise_sum = float(np.max(np.sum(distances, axis=1)))
            r_max = (
                math.sqrt(max_observed / max_pairwise)
                if max_pairwise > 0.0
                else 0.0
            )
            r_sum = (
                sum_observed / max_pairwise_sum
                if max_pairwise_sum > 0.0
                else 0.0
            )

        if not np.all(np.isfinite(crafted)):
            raise FloatingPointError(f"{attack} produced a non-finite malicious update")
        transformed = list(updates)
        for index in active_indices:
            parameters, num_examples = updates[index]
            transformed[index] = (
                _replace_delta(
                    parameters,
                    global_params,
                    crafted,
                    self.scalable_parameter_indices,
                ),
                num_examples,
            )
            if not all(
                np.all(np.isfinite(np.asarray(value)))
                for value in transformed[index][0]
                if np.issubdtype(np.asarray(value).dtype, np.floating)
            ):
                raise FloatingPointError(
                    f"{attack} produced non-finite model parameters after dtype conversion"
                )
        crafted_norm = float(np.linalg.norm(crafted))
        reference_norm = float(np.linalg.norm(mean))
        metrics = {
            **base_metrics,
            "attack_coordinator_applied": True,
            "attack_coordinator_name": attack,
            "attack_coordinator_round": int(server_round),
            "attack_coordinator_reference_clients": len(reference_indices),
            "attack_coordinator_knowledge": knowledge,
            "attack_coordinator_gamma": float(gamma),
            "attack_coordinator_gamma_fraction": float(
                getattr(self.config, "optimization_gamma_fraction", 1.0)
            ),
            "attack_coordinator_crafted_norm": crafted_norm,
            "attack_coordinator_reference_norm": reference_norm,
            "attack_coordinator_direction_norm": float(
                np.linalg.norm(direction) if attack not in {"lie", "alie"} else 0.0
            ),
            "attack_coordinator_reference_median_norm": float(
                np.median(np.linalg.norm(reference_vectors, axis=1))
            ),
            "attack_coordinator_reference_max_norm": float(
                np.max(np.linalg.norm(reference_vectors, axis=1))
            ),
            "attack_coordinator_pairwise_max_distance": float(
                constraint_limit
            ),
            "attack_coordinator_gamma_max": float(
                getattr(self.config, "optimization_gamma_max", 1.0e6)
            ),
            "attack_coordinator_constraint_value": float(constraint_value),
            "attack_coordinator_constraint_limit": float(constraint_limit),
            "attack_contract_hash": str(self.contract["contract_hash"]),
            "attack_implementation_hash": str(self.contract["implementation_hash"]),
        }
        if sign_flippable_ratio is not None:
            metrics["attack_coordinator_sign_flippable_ratio"] = float(
                sign_flippable_ratio
            )
        if r_max is not None and r_sum is not None:
            metrics["attack_coordinator_r_max"] = float(r_max)
            metrics["attack_coordinator_r_sum"] = float(r_sum)
        return CoordinatedAttackResult(transformed, metrics)
