"""Hard-constrained sub-probability aggregation for RTC-v2."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Mapping, Sequence

import numpy as np

from defenses.defense_base import UpdateList
from .history import RoundRecord


class InfluenceAggregator:
    """Clip deltas and enforce state, client, and direction exposure bounds."""

    def __init__(self, params: Mapping[str, object], eps: float = 1e-9):
        self.eps = float(eps)
        self.enable_soft_trust_weighting = bool(params.get("enable_soft_trust_weighting", True))
        self.enable_delta_clipping = bool(params.get("enable_delta_clipping", params.get("clipping_enabled", True)))
        self.enable_trust_caps = bool(params.get("enable_trust_caps", True))
        self.enable_exposure_budgets = bool(params.get("enable_exposure_budgets", True))
        self.clip_multiplier = float(params.get("clip_multiplier", params.get("norm_clip_factor", 1.0)))
        self.clip_mad_k = float(params.get("norm_clip_mad_k", 0.0))
        normal_cap = float(params.get("normal_weight_cap_multiplier", params.get("per_client_weight_multiplier", 2.0)))
        self.state_weight_cap_multipliers = self._state_weight_cap_multipliers(
            params.get("state_weight_cap_multipliers"), normal_cap
        )
        self.min_effective_clients = int(params.get("min_effective_clients", 3))
        self.exposure_window = max(1, int(params.get("exposure_window", 4)))
        self.client_exposure_budget_multiplier = max(
            0.0, float(params.get("client_exposure_budget_multiplier", 4.0))
        )
        self.direction_exposure_budget_multiplier = max(
            0.0, float(params.get("direction_exposure_budget_multiplier", 4.0))
        )
        self.direction_group_bits = max(1, int(params.get("direction_group_bits", 8)))
        self.direction_group_seed = int(params.get("direction_group_seed", 42))

        self._client_exposures: DefaultDict[str, List[tuple[int, float]]] = defaultdict(list)
        self._direction_exposures: DefaultDict[str, List[tuple[int, float]]] = defaultdict(list)
        self._direction_projection: np.ndarray | None = None
        self._last_clip_norm = 0.0
        self._last_fallback_used = False
        self._last_fallback_reason = ""
        self._last_weight_sum = 0.0
        self._last_zero_mass = 1.0
        self._last_max_cap_violation = 0.0
        self._last_max_budget_violation = 0.0

    def assign_weights(self, records: Sequence[RoundRecord]) -> None:
        """Project nominal sample/trust weights onto per-state upper bounds.

        Nominal weights sum to one.  Capping may reduce their sum; the missing
        mass deliberately represents the server's zero update and is never
        redistributed to other clients.
        """
        if not records:
            return
        n = len(records)
        nominal = np.asarray(
            [max(0.0, float(record.raw_weight)) for record in records], dtype=np.float64
        )
        if self.enable_soft_trust_weighting:
            nominal *= np.asarray(
                [float(np.clip(record.trust, 0.0, 1.0)) for record in records], dtype=np.float64
            )
        total = float(nominal.sum())
        if total <= self.eps or not np.isfinite(total):
            nominal[:] = 0.0
        else:
            nominal /= total

        caps = np.full(n, 1.0, dtype=np.float64)
        if self.enable_trust_caps:
            caps = np.asarray(
                [self.state_weight_cap_multipliers.get(record.state, 0.0) / n for record in records],
                dtype=np.float64,
            )
        weights = np.minimum(nominal, caps)
        weights = np.maximum(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        zero_mass = max(0.0, 1.0 - float(weights.sum()))

        for record, weight, cap, target in zip(records, weights, caps, nominal):
            record.weight_cap = float(cap)
            record.effective_weight = float(weight)
            record.aggregation_weight = float(weight)
            record.zero_mass = float(zero_mass)
            record.capped = bool(weight + self.eps < target)
            record.quarantined = record.state == "quarantined"

    def aggregate(
        self,
        updates: UpdateList,
        records: Sequence[RoundRecord],
        global_params: Sequence[np.ndarray],
        *,
        server_round: int = 0,
    ) -> List[np.ndarray]:
        self._reset_round_diagnostics()
        if not updates:
            return []

        delta_norms = np.asarray([record.norm for record in records], dtype=np.float64)
        clip_norm = self.compute_clip_norm(delta_norms)
        self._last_clip_norm = 0.0 if not np.isfinite(clip_norm) else float(clip_norm)
        clipped_flat: List[np.ndarray] = []
        for record in records:
            record.clipped = bool(np.isfinite(clip_norm) and record.norm > clip_norm + self.eps)
            clipped_flat.append(self.clip_delta(record.delta_flat, record.norm, clip_norm))

        effective = np.asarray([record.effective_weight for record in records], dtype=np.float64)
        effective = np.maximum(np.nan_to_num(effective, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        effective = self._apply_exposure_budgets(
            records, clipped_flat, effective, clip_norm, int(server_round)
        )

        total = float(effective.sum())
        if total > 1.0 + self.eps:
            return self._safe_zero_update(records, global_params, "weight_mass_exceeds_one")
        cap_violation = max(
            (max(0.0, float(weight) - float(record.weight_cap)) for weight, record in zip(effective, records)),
            default=0.0,
        )
        self._last_max_cap_violation = float(cap_violation)
        if cap_violation > self.eps:
            return self._safe_zero_update(records, global_params, "weight_cap_violation")

        active = int(np.count_nonzero(effective > self.eps))
        if active < min(self.min_effective_clients, len(records)):
            return self._safe_zero_update(records, global_params, "insufficient_effective_clients")
        if total <= self.eps or not np.isfinite(total):
            return self._safe_zero_update(records, global_params, "zero_effective_weight")

        self._last_weight_sum = total
        self._last_zero_mass = max(0.0, 1.0 - total)
        for record, weight in zip(records, effective):
            record.effective_weight = float(weight)
            record.aggregation_weight = float(weight)
            record.zero_mass = self._last_zero_mass

        result: List[np.ndarray] = []
        for param_idx, global_param in enumerate(global_params):
            if not np.issubdtype(global_param.dtype, np.floating):
                result.append(global_param.copy())
                continue
            global_float = global_param.astype(np.float32)
            aggregate_delta = np.zeros_like(global_float, dtype=np.float32)
            for weight, record, (params, _) in zip(effective, records, updates):
                client_delta = params[param_idx].astype(np.float32) - global_float
                aggregate_delta += float(weight) * self.clip_delta(
                    client_delta, record.norm, clip_norm
                )
            result.append((global_float + aggregate_delta).astype(global_param.dtype, copy=False))

        self._record_effective_influence(records, clipped_flat, effective, clip_norm, int(server_round))
        return result

    def compute_clip_norm(self, norms: np.ndarray) -> float:
        if not self.enable_delta_clipping:
            return float("inf")
        finite = norms[np.isfinite(norms)]
        if finite.size == 0:
            return float("inf")
        median = float(np.median(finite))
        if self.clip_mad_k > 0.0:
            mad = float(np.median(np.abs(finite - median)))
            if mad > self.eps:
                return float(max(0.0, median + self.clip_mad_k * 1.4826 * mad))
        return float(max(0.0, median * self.clip_multiplier))

    def clip_delta(self, delta: np.ndarray, norm: float, clip_norm: float) -> np.ndarray:
        if not self.enable_delta_clipping or not np.isfinite(clip_norm):
            return delta
        if norm <= clip_norm + self.eps or norm <= self.eps:
            return delta
        return delta * float(clip_norm / (norm + self.eps))

    def _apply_exposure_budgets(
        self,
        records: Sequence[RoundRecord],
        clipped_flat: Sequence[np.ndarray],
        weights: np.ndarray,
        clip_norm: float,
        server_round: int,
    ) -> np.ndarray:
        if not self.enable_exposure_budgets or not records:
            return weights
        n = len(records)
        reference_norm = self._exposure_reference_norm(clipped_flat, clip_norm)
        norm_ratios = np.asarray(
            [float(np.linalg.norm(delta)) / reference_norm for delta in clipped_flat], dtype=np.float64
        )
        groups = self._direction_groups(records, clipped_flat)
        client_budget = self.client_exposure_budget_multiplier / n
        direction_budget = self.direction_exposure_budget_multiplier / n

        constrained = weights.copy()
        for idx, record in enumerate(records):
            spent = self._window_spent(self._client_exposures[record.client_id], server_round)
            remaining = max(0.0, client_budget - spent)
            record.client_budget_remaining = remaining
            if norm_ratios[idx] > self.eps:
                constrained[idx] = min(constrained[idx], remaining / norm_ratios[idx])

        grouped_indices: DefaultDict[str, List[int]] = defaultdict(list)
        for idx, group in enumerate(groups):
            records[idx].direction_group = group
            grouped_indices[group].append(idx)
        for group, indices in grouped_indices.items():
            if group == "consensus":
                for idx in indices:
                    records[idx].direction_budget_remaining = float("inf")
                continue
            spent = self._window_spent(self._direction_exposures[group], server_round)
            group_budget = direction_budget * len(indices)
            remaining = max(0.0, group_budget - spent)
            attempted = float(sum(constrained[idx] * norm_ratios[idx] for idx in indices))
            scale = 1.0 if attempted <= remaining + self.eps else remaining / max(attempted, self.eps)
            for idx in indices:
                constrained[idx] *= scale
                records[idx].direction_budget_remaining = remaining

        constrained = np.maximum(constrained, 0.0)
        for record, before, after in zip(records, weights, constrained):
            if after + self.eps < before:
                record.capped = True
        return constrained

    def _record_effective_influence(
        self,
        records: Sequence[RoundRecord],
        clipped_flat: Sequence[np.ndarray],
        weights: np.ndarray,
        clip_norm: float,
        server_round: int,
    ) -> None:
        reference_norm = self._exposure_reference_norm(clipped_flat, clip_norm)
        group_round_exposure: DefaultDict[str, float] = defaultdict(float)
        for record, delta, weight in zip(records, clipped_flat, weights):
            delta_norm = float(np.linalg.norm(delta))
            record.influence_effective = float(weight * delta_norm)
            record.normalized_exposure = float(record.influence_effective / reference_norm)
            self._client_exposures[record.client_id].append((server_round, record.normalized_exposure))
            group_round_exposure[record.direction_group] += record.normalized_exposure
            self._last_max_budget_violation = max(
                self._last_max_budget_violation,
                max(0.0, record.normalized_exposure - record.client_budget_remaining),
            )
            record.client_budget_remaining = max(
                0.0, record.client_budget_remaining - record.normalized_exposure
            )
        for group, exposure in group_round_exposure.items():
            if group == "consensus":
                continue
            self._direction_exposures[group].append((server_round, float(exposure)))
            group_records = [record for record in records if record.direction_group == group]
            if group_records:
                self._last_max_budget_violation = max(
                    self._last_max_budget_violation,
                    max(0.0, exposure - group_records[0].direction_budget_remaining),
                )
        for record in records:
            group_exposure = group_round_exposure[record.direction_group]
            if np.isfinite(record.direction_budget_remaining):
                record.direction_budget_remaining = max(
                    0.0, record.direction_budget_remaining - group_exposure
                )

    def _direction_groups(
        self, records: Sequence[RoundRecord], clipped_flat: Sequence[np.ndarray]
    ) -> List[str]:
        if not records:
            return []
        clipped_signatures = []
        for record, delta in zip(records, clipped_flat):
            factor = 0.0 if record.norm <= self.eps else float(np.linalg.norm(delta) / record.norm)
            clipped_signatures.append(record.signature.astype(np.float32, copy=False) * factor)
        matrix = np.vstack(clipped_signatures).astype(np.float32, copy=False)
        residuals = matrix - np.median(matrix, axis=0)
        dim = residuals.shape[1]
        if self._direction_projection is None or self._direction_projection.shape[0] != dim:
            rng = np.random.default_rng(self.direction_group_seed)
            self._direction_projection = rng.standard_normal(
                (dim, self.direction_group_bits), dtype=np.float32
            )
        projected = residuals @ self._direction_projection
        groups = []
        for residual, values in zip(residuals, projected):
            if float(np.linalg.norm(residual)) <= self.eps:
                groups.append("consensus")
            else:
                groups.append("".join("1" if value >= 0.0 else "0" for value in values))
        return groups

    def _window_spent(self, entries: List[tuple[int, float]], server_round: int) -> float:
        first_round = server_round - self.exposure_window + 1
        entries[:] = [(rnd, value) for rnd, value in entries if rnd >= first_round]
        return float(sum(value for _, value in entries))

    def _exposure_reference_norm(
        self, clipped_flat: Sequence[np.ndarray], clip_norm: float
    ) -> float:
        if np.isfinite(clip_norm) and clip_norm > self.eps:
            return float(clip_norm)
        norms = [float(np.linalg.norm(delta)) for delta in clipped_flat]
        positive = [value for value in norms if value > self.eps and np.isfinite(value)]
        return max(float(np.median(positive)) if positive else 1.0, self.eps)

    def _safe_zero_update(
        self,
        records: Sequence[RoundRecord],
        global_params: Sequence[np.ndarray],
        reason: str,
    ) -> List[np.ndarray]:
        self._last_fallback_used = True
        self._last_fallback_reason = reason
        self._last_weight_sum = 0.0
        self._last_zero_mass = 1.0
        for record in records:
            record.fallback_used = True
            record.effective_weight = 0.0
            record.aggregation_weight = 0.0
            record.influence_effective = 0.0
            record.normalized_exposure = 0.0
            record.zero_mass = 1.0
        return [param.copy() for param in global_params]

    def _reset_round_diagnostics(self) -> None:
        self._last_fallback_used = False
        self._last_fallback_reason = ""
        self._last_weight_sum = 0.0
        self._last_zero_mass = 1.0
        self._last_max_cap_violation = 0.0
        self._last_max_budget_violation = 0.0

    @staticmethod
    def _state_weight_cap_multipliers(raw: object, normal_cap: float) -> dict[str, float]:
        defaults = {
            "normal": max(0.0, normal_cap),
            "watch": 1.0,
            "restricted": 0.1,
            "quarantined": 0.0,
        }
        if isinstance(raw, Mapping):
            for key in defaults:
                if key in raw:
                    defaults[key] = max(0.0, float(raw[key]))
        return defaults
