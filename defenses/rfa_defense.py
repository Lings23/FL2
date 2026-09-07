"""Robust Federated Aggregation via a smoothed geometric median."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList


class RFADefense(BaseDefense):
    """Approximate the weighted geometric median with smoothed Weiszfeld steps.

    The implementation follows Pillutla, Kakade, and Harchaoui's robust
    aggregation oracle.  It operates on client model deltas so non-floating
    model buffers can be preserved from the current global model.
    """

    def __init__(self, cfg: DefenseConfig, **_: object):
        super().__init__(cfg)
        params = cfg.custom_params or {}
        self.num_iterations = int(params.get("num_iterations", 3))
        self.smoothing = float(params.get("smoothing", 1e-6))
        self.use_num_examples = bool(params.get("use_num_examples", True))
        if self.num_iterations < 1:
            raise ValueError("RFA num_iterations must be >= 1")
        if not np.isfinite(self.smoothing) or self.smoothing <= 0.0:
            raise ValueError("RFA smoothing must be positive and finite")

    def aggregate(self, updates: UpdateList) -> list[np.ndarray]:
        if not updates:
            raise ValueError("RFA requires at least one client update")
        if self._global_params is None:
            raise RuntimeError("RFA requires global model context")

        self._validate_updates(updates)
        vectors = np.asarray(
            [self._flatten_delta(params) for params, _ in updates],
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("RFA requires at least one floating-point parameter")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("RFA received NaN or Inf in a client update")

        if self.use_num_examples:
            input_weights = np.asarray(
                [num_examples for _, num_examples in updates], dtype=np.float64
            )
            if (
                not np.all(np.isfinite(input_weights))
                or np.any(input_weights <= 0.0)
            ):
                raise ValueError(
                    "RFA requires positive finite client sample weights"
                )
        else:
            input_weights = np.ones(len(updates), dtype=np.float64)
        input_weights /= float(input_weights.sum())

        estimate = self._weighted_vector_average(vectors, input_weights)
        final_raw_weights = input_weights.copy()
        final_weights = input_weights.copy()
        final_step_norm = 0.0
        for _ in range(self.num_iterations):
            distances = self._distances(vectors, estimate)
            final_raw_weights = input_weights / np.maximum(
                distances, self.smoothing
            )
            raw_total = float(final_raw_weights.sum())
            if not np.isfinite(raw_total) or raw_total <= 0.0:
                raise FloatingPointError("RFA produced invalid Weiszfeld weights")
            final_weights = final_raw_weights / raw_total
            next_estimate = self._weighted_vector_average(vectors, final_weights)
            final_step_norm = float(
                np.linalg.norm(
                    next_estimate.astype(np.float64) - estimate.astype(np.float64)
                )
            )
            if not np.isfinite(final_step_norm):
                raise FloatingPointError("RFA produced a non-finite iterate")
            estimate = next_estimate

        final_distances = self._distances(vectors, estimate)
        objective = float(np.dot(input_weights, final_distances))
        update_norm = float(np.linalg.norm(estimate.astype(np.float64)))
        if not np.isfinite(objective) or not np.isfinite(update_norm):
            raise FloatingPointError("RFA produced a non-finite aggregate")

        client_ids = self._client_ids_for(len(updates))
        self._record_scalar_weights(client_ids, final_raw_weights)
        self.last_round_metrics = {
            "rfa_iterations": float(self.num_iterations),
            "rfa_objective": objective,
            "rfa_final_step_norm": final_step_norm,
            "rfa_update_norm": update_norm,
            "rfa_min_weight": float(np.min(final_weights)),
            "rfa_max_weight": float(np.max(final_weights)),
            "rfa_effective_clients": float(
                1.0 / max(float(np.dot(final_weights, final_weights)), 1e-12)
            ),
        }
        return self._apply_delta(estimate)

    def _validate_updates(self, updates: UpdateList) -> None:
        assert self._global_params is not None
        expected_count = len(self._global_params)
        for params, _ in updates:
            if len(params) != expected_count:
                raise ValueError("RFA client/global parameter counts do not match")
            for param, reference in zip(params, self._global_params):
                if param.shape != reference.shape:
                    raise ValueError("RFA client/global parameter shapes do not match")

    @staticmethod
    def _weighted_vector_average(
        vectors: np.ndarray, weights: Sequence[float]
    ) -> np.ndarray:
        # Keeping the model-sized iterate in float32 avoids doubling the peak
        # memory cost for large models while the scalar weights stay float64.
        result = np.asarray(weights, dtype=np.float32) @ vectors
        result = np.asarray(result, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("RFA produced a non-finite weighted average")
        return result

    @staticmethod
    def _distances(vectors: np.ndarray, estimate: np.ndarray) -> np.ndarray:
        # Compute one temporary model vector at a time instead of materializing
        # an additional clients-by-parameters matrix.  Float64 distance
        # arithmetic also prevents a finite high-magnitude Byzantine vector
        # from overflowing the float32 squared norm into infinity.
        estimate64 = estimate.astype(np.float64)
        distances = []
        for vector in vectors:
            difference = vector.astype(np.float64)
            difference -= estimate64
            distances.append(float(np.linalg.norm(difference)))
        return np.asarray(distances, dtype=np.float64)

    def _apply_delta(self, flat_delta: np.ndarray) -> list[np.ndarray]:
        assert self._global_params is not None
        result: list[np.ndarray] = []
        offset = 0
        for global_param in self._global_params:
            if np.issubdtype(global_param.dtype, np.floating):
                size = global_param.size
                delta = flat_delta[offset:offset + size].reshape(global_param.shape)
                value = global_param.astype(np.float32) + delta
                result.append(value.astype(global_param.dtype, copy=False))
                offset += size
            else:
                result.append(global_param.copy())
        if offset != flat_delta.size:
            raise ValueError("RFA aggregate size does not match global parameters")
        return result
