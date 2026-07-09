"""Influence-capped aggregation for RTC-v2."""

from __future__ import annotations

import math
from typing import List, Mapping, Sequence

import numpy as np

from defenses.defense_base import UpdateList
from .history import RoundRecord


class InfluenceAggregator:
    """State-machine weights, norm clipping, caps, and safe fallback."""

    def __init__(self, params: Mapping[str, object], eps: float = 1e-9):
        self.eps = float(eps)
        self.enable_soft_trust_weighting = bool(params.get("enable_soft_trust_weighting", True))
        self.enable_delta_clipping = bool(params.get("enable_delta_clipping", params.get("clipping_enabled", True)))
        self.enable_trust_caps = bool(params.get("enable_trust_caps", True))
        self.clip_multiplier = float(params.get("clip_multiplier", params.get("norm_clip_factor", 1.0)))
        self.clip_mad_k = float(params.get("norm_clip_mad_k", 0.0))
        self.per_client_weight_multiplier = float(params.get("per_client_weight_multiplier", 0.6))
        self.min_effective_clients = int(params.get("min_effective_clients", 3))
        self.fallback_strategy = str(params.get("fallback_strategy", params.get("fallback", "skip_update"))).lower()
        self.state_multipliers = self._state_multipliers(params.get("state_multipliers"))
        self._last_clip_norm = 0.0
        self._last_fallback_used = False
        self._last_fallback_reason = ""

    def assign_weights(self, records: Sequence[RoundRecord]) -> None:
        if not records:
            return
        n = len(records)
        max_share = self.per_client_weight_multiplier / max(1, n)
        raw_weights = []
        for record in records:
            if not self.enable_soft_trust_weighting:
                raw_weights.append(max(0.0, float(record.raw_weight)))
                continue
            multiplier = self.state_multipliers.get(record.state, 1.0)
            weight = record.raw_weight * record.trust * multiplier
            if record.state == "quarantined":
                weight = 0.0
            raw_weights.append(max(0.0, float(weight)))

        weights = np.asarray(raw_weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= self.eps or not np.isfinite(total):
            weights = np.zeros(len(records), dtype=np.float64)
        else:
            normalized = weights / total
            capped = normalized.copy()
            if self.enable_trust_caps:
                for idx, share in enumerate(normalized):
                    if share > max_share:
                        capped[idx] = max_share
                        records[idx].capped = True
                if float(capped.sum()) > self.eps:
                    capped = capped / float(capped.sum())
                weights = capped

        for record, weight in zip(records, weights):
            record.effective_weight = float(weight)
            record.aggregation_weight = float(weight)
            record.quarantined = record.state == "quarantined" or weight <= self.eps and record.state == "quarantined"

    def aggregate(
        self,
        updates: UpdateList,
        records: Sequence[RoundRecord],
        global_params: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        self._last_fallback_used = False
        self._last_fallback_reason = ""
        if not updates:
            return []
        effective = np.asarray([r.effective_weight for r in records], dtype=np.float64)
        effective = np.maximum(np.nan_to_num(effective, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        active = int(np.count_nonzero(effective > self.eps))
        if active < min(self.min_effective_clients, len(records)):
            self._last_fallback_used = True
            self._last_fallback_reason = "insufficient_effective_clients"
            for record in records:
                record.fallback_used = True
                record.aggregation_weight = 0.0
            if self.fallback_strategy in {"fedavg", "unsafe_fedavg"}:
                effective = np.asarray([r.raw_weight for r in records], dtype=np.float64)
            else:
                return [p.copy() for p in global_params]

        delta_norms = np.asarray([r.norm for r in records], dtype=np.float64)
        clip_norm = self.compute_clip_norm(delta_norms)
        self._last_clip_norm = 0.0 if not np.isfinite(clip_norm) else float(clip_norm)
        for record in records:
            record.clipped = bool(np.isfinite(clip_norm) and record.norm > clip_norm + self.eps)

        total = float(effective.sum())
        if total <= self.eps or not np.isfinite(total):
            self._last_fallback_used = True
            self._last_fallback_reason = "zero_effective_weight"
            return [p.copy() for p in global_params]

        normalized = effective / total
        for record, weight in zip(records, normalized):
            record.aggregation_weight = float(weight)

        result: List[np.ndarray] = []
        for param_idx, global_param in enumerate(global_params):
            dtype = global_param.dtype
            if np.issubdtype(dtype, np.floating):
                delta = np.zeros_like(global_param, dtype=np.float32)
                global_float = global_param.astype(np.float32)
                for weight, record, (params, _) in zip(normalized, records, updates):
                    client_delta = params[param_idx].astype(np.float32) - global_float
                    client_delta = self.clip_delta(client_delta, record.norm, clip_norm)
                    delta += float(weight) * client_delta
                result.append((global_float + delta).astype(dtype, copy=False))
            else:
                acc = np.zeros_like(global_param, dtype=np.float32)
                for weight, (params, _) in zip(normalized, updates):
                    acc += float(weight) * params[param_idx].astype(np.float32)
                result.append(np.round(acc).astype(dtype))
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

    @staticmethod
    def _state_multipliers(raw: object) -> dict[str, float]:
        defaults = {"normal": 1.0, "watch": 0.5, "restricted": 0.05, "quarantined": 0.0}
        if isinstance(raw, Mapping):
            for key in defaults:
                if key in raw:
                    defaults[key] = max(0.0, float(raw[key]))
        return defaults
