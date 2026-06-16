"""
Time-consistency defense for stateful federated aggregation.

This defense implements the S1-S7 design from the invention note in a way that
matches FedSec's existing Flower/PyTorch aggregation surface:

* local "gradients" are represented as client model deltas
  (client_params - previous_global_params);
* per-client histories are keyed by the real Flower client id;
* high-dimensional updates are reduced to deterministic signatures for
  temporal feature extraction;
* aggregation remains a soft weighted update, never a hard client drop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList

logger = logging.getLogger(__name__)


SCALE_ORDER = ("instant", "short", "mid", "long")
FEATURE_DIM = 7


@dataclass
class ClientTrustState:
    """Per-client temporal state used by TimeConsistencyDefense."""

    participation_count: int = 0
    last_seen_round: int = 0
    signature_history: List[np.ndarray] = field(default_factory=list)
    norm_history: List[float] = field(default_factory=list)
    feature_history: List[np.ndarray] = field(default_factory=list)
    rank_history: List[int] = field(default_factory=list)
    scale_trust: Dict[str, float] = field(default_factory=dict)
    scale_history: Dict[str, List[float]] = field(default_factory=dict)
    final_trust: float = 0.5
    final_trust_history: List[float] = field(default_factory=list)
    dominant_freq_history: List[float] = field(default_factory=list)


class TimeConsistencyDefense(BaseDefense):
    """
    Multi-scale temporal trust defense.

    Config parameters are read from ``DefenseConfig.custom_params``. All
    parameters have safe defaults so the defense can be enabled with only:

        security.defense.type=time_consistency
    """

    def __init__(self, cfg: DefenseConfig, num_clients: int = 100):
        super().__init__(cfg)
        params = cfg.custom_params or {}

        self.num_clients = int(num_clients)
        self.eps = float(params.get("eps", 1e-9))
        self.projection_dim = int(params.get("projection_dim", 2048))
        self.windows = self._read_windows(params.get("windows"))
        self.max_window = max(self.windows.values())
        self.max_history = int(params.get("max_history", max(2 * self.max_window, 64)))

        self.fixed_scale_weights = self._normalize_weights(
            params.get("fixed_scale_weights", [0.45, 0.30, 0.20, 0.05])
        )
        self.cold_start_rounds = int(params.get("cold_start_rounds", 10))
        self.offline_reset_rounds = int(params.get("offline_reset_rounds", 10))
        self.trust_update_rate_cold = float(params.get("trust_update_rate_cold", 0.35))
        self.trust_update_rate_normal = float(params.get("trust_update_rate_normal", 0.15))
        self.sigmoid_scale = float(params.get("sigmoid_scale", 8.0))
        self.sigmoid_center = float(params.get("sigmoid_center", 0.5))
        self.min_effective_weight = float(params.get("min_effective_weight", 1e-4))

        self.trust_score_scale = float(params.get("trust_score_scale", 4.0))
        self.direction_threshold = float(params.get("direction_threshold", 0.65))
        self.direction_window = int(params.get("direction_window", 3))
        self.direction_penalty = float(params.get("direction_penalty", 0.75))
        self.offset_window = int(params.get("offset_window", 5))
        self.offset_growth_threshold = float(params.get("offset_growth_threshold", 0.35))
        self.offset_penalty = float(params.get("offset_penalty", 0.75))
        self.periodic_entropy_threshold = float(params.get("periodic_entropy_threshold", 0.65))
        self.periodic_freq_var_threshold = float(params.get("periodic_freq_var_threshold", 1e-3))
        self.periodic_penalty = float(params.get("periodic_penalty", 0.60))

        self.derivative_beta = float(params.get("derivative_beta", 4.0))
        self.derivative_max = float(params.get("derivative_max", 3.0))
        self.uncertainty_gamma = float(params.get("uncertainty_gamma", 2.0))
        self.collaboration_coeff = float(params.get("collaboration_coeff", 0.25))
        self.correlation_threshold = float(params.get("correlation_threshold", 0.6))
        self.high_trust_threshold = float(params.get("high_trust_threshold", 0.65))
        self.derivative_exp = float(params.get("derivative_exp", 1.0))
        self.uncertainty_exp = float(params.get("uncertainty_exp", 0.5))
        self.collaboration_exp = float(params.get("collaboration_exp", 1.0))

        self.transition_start_rounds = int(params.get("transition_start_rounds", 0))
        self.transition_end_rounds = int(params.get("transition_end_rounds", self.cold_start_rounds))

        self.feature_weights: Dict[str, np.ndarray] = {
            "instant": np.array([0.35, 0.25, 0.25, 0.15, 0.00, 0.00, 0.00], dtype=np.float32),
            "short": np.array([0.25, 0.20, 0.20, 0.15, 0.08, 0.06, 0.06], dtype=np.float32),
            "mid": np.array([0.16, 0.16, 0.16, 0.14, 0.14, 0.12, 0.12], dtype=np.float32),
            "long": np.array([0.08, 0.08, 0.10, 0.08, 0.25, 0.20, 0.21], dtype=np.float32),
        }

        self._states: Dict[str, ClientTrustState] = {}
        self._projection_indices: np.ndarray | None = None
        self._server_round: int = 0
        self._client_ids: List[str] = []
        self._global_params: List[np.ndarray] | None = None

        self.last_round_metrics: Dict[str, float] = {}
        self.last_client_trusts: Dict[str, float] = {}
        self.last_client_weights: Dict[str, float] = {}

        logger.info(
            "TimeConsistencyDefense init | projection_dim=%d windows=%s",
            self.projection_dim,
            self.windows,
        )

    def set_context(
        self,
        server_round: int,
        client_ids: Sequence[str],
        global_params: Sequence[np.ndarray],
    ) -> None:
        self._server_round = int(server_round)
        self._client_ids = [str(cid) for cid in client_ids]
        self._global_params = [p.copy() for p in global_params]

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        if not updates:
            return []

        if self._global_params is None:
            logger.warning("TimeConsistencyDefense missing global context; falling back to FedAvg.")
            return self._weighted_average(updates)

        client_ids = self._resolve_client_ids(len(updates))
        global_flat = self._flatten(self._global_params)
        vectors = []
        signatures = []
        norms = []

        for params, _ in updates:
            delta_flat = self._flatten(params) - global_flat
            vectors.append(delta_flat)
            signature = self._signature(delta_flat)
            signatures.append(signature)
            norms.append(float(np.linalg.norm(signature)))

        signatures_arr = np.vstack(signatures).astype(np.float32, copy=False)
        distances, ranks = self._group_distances_and_ranks(signatures_arr)

        trust_scores: List[float] = []
        effective_weights: List[float] = []
        round_initial_trust = self._initial_trust()

        for idx, cid in enumerate(client_ids):
            state = self._get_state(cid, round_initial_trust)
            feature, dominant_freq = self._build_feature(
                state=state,
                signature=signatures[idx],
                norm=norms[idx],
                offset_score=distances[idx],
                current_rank=ranks[idx],
                num_round_clients=len(updates),
            )

            self._append_current_observation(
                state=state,
                signature=signatures[idx],
                norm=norms[idx],
                feature=feature,
                current_rank=ranks[idx],
                dominant_freq=dominant_freq,
            )

            final_trust = self._update_trust(state, feature)
            trust_scores.append(final_trust)

            trust_weight = self._sigmoid(self.sigmoid_scale * (final_trust - self.sigmoid_center))
            trust_weight = max(self.min_effective_weight, trust_weight)
            effective_weight = trust_weight * max(1, int(updates[idx][1]))
            effective_weights.append(effective_weight)

        aggregated = self._aggregate_with_effective_weights(updates, effective_weights)
        self._record_round_metrics(client_ids, trust_scores, effective_weights)
        return aggregated

    def _read_windows(self, raw) -> Dict[str, int]:
        defaults = {"instant": 1, "short": 5, "mid": 15, "long": 30}
        if isinstance(raw, dict):
            for key, val in raw.items():
                if key in defaults:
                    defaults[key] = max(1, int(val))
        return defaults

    def _normalize_weights(self, raw) -> np.ndarray:
        arr = np.array(raw, dtype=np.float32)
        if arr.shape[0] != len(SCALE_ORDER) or not np.isfinite(arr).all() or arr.sum() <= self.eps:
            arr = np.array([0.45, 0.30, 0.20, 0.05], dtype=np.float32)
        arr = np.maximum(arr, 0.0)
        return arr / (arr.sum() + self.eps)

    def _resolve_client_ids(self, n_updates: int) -> List[str]:
        if len(self._client_ids) == n_updates:
            return list(self._client_ids)
        return [str(i) for i in range(n_updates)]

    def _get_state(self, cid: str, default_initial_trust: float | None = None) -> ClientTrustState:
        current_round = self._server_round
        state = self._states.get(cid)
        needs_reset = state is None
        if state is not None and state.last_seen_round > 0:
            needs_reset = current_round - state.last_seen_round > self.offline_reset_rounds

        if needs_reset:
            init = self._initial_trust(exclude_cid=cid) if default_initial_trust is None else default_initial_trust
            state = ClientTrustState(
                last_seen_round=current_round,
                scale_trust={scale: init for scale in SCALE_ORDER},
                scale_history={scale: [] for scale in SCALE_ORDER},
                final_trust=init,
            )
            self._states[cid] = state
        return state

    def _initial_trust(self, exclude_cid: str | None = None) -> float:
        vals = [
            s.final_trust
            for cid, s in self._states.items()
            if cid != exclude_cid and np.isfinite(s.final_trust)
        ]
        if not vals:
            return 0.5
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _signature(self, flat_delta: np.ndarray) -> np.ndarray:
        if self._projection_indices is None or self._projection_indices[-1] >= flat_delta.size:
            if flat_delta.size <= self.projection_dim:
                self._projection_indices = np.arange(flat_delta.size, dtype=np.int64)
            else:
                self._projection_indices = np.linspace(
                    0,
                    flat_delta.size - 1,
                    num=self.projection_dim,
                    dtype=np.int64,
                )
        signature = flat_delta[self._projection_indices].astype(np.float32, copy=False)
        return np.nan_to_num(signature, nan=0.0, posinf=0.0, neginf=0.0)

    def _group_distances_and_ranks(self, signatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_clients = signatures.shape[0]
        if n_clients <= 1:
            return np.zeros(n_clients, dtype=np.float32), np.zeros(n_clients, dtype=np.int64)

        center = np.median(signatures, axis=0)
        raw_dist = np.linalg.norm(signatures - center, axis=1)
        median = float(np.median(raw_dist))
        mad = float(np.median(np.abs(raw_dist - median)))
        robust_z = (raw_dist - median) / (1.4826 * mad + self.eps)
        offset_scores = np.clip(robust_z, 0.0, 5.0) / 5.0

        order = np.argsort(raw_dist)
        ranks = np.empty(n_clients, dtype=np.int64)
        ranks[order] = np.arange(n_clients)
        return offset_scores.astype(np.float32), ranks

    def _build_feature(
        self,
        state: ClientTrustState,
        signature: np.ndarray,
        norm: float,
        offset_score: float,
        current_rank: int,
        num_round_clients: int,
    ) -> tuple[np.ndarray, float]:
        if state.signature_history:
            prev_sig = state.signature_history[-1]
            denom = np.linalg.norm(prev_sig) * np.linalg.norm(signature) + self.eps
            cosine = float(np.dot(prev_sig, signature) / denom) if denom > self.eps else 1.0
            direction_anomaly = (1.0 - np.clip(cosine, -1.0, 1.0)) / 2.0
        else:
            direction_anomaly = 0.0

        if state.norm_history:
            prev_norm = state.norm_history[-1]
            relative_change = abs(norm - prev_norm) / (abs(prev_norm) + self.eps)
            magnitude_change = 1.0 - math.exp(-min(relative_change, 10.0))
        else:
            magnitude_change = 0.0

        if state.rank_history and num_round_clients > 1:
            rank_change = abs(int(current_rank) - int(state.rank_history[-1])) / max(1, num_round_clients - 1)
        else:
            rank_change = 0.0

        spectral_entropy_low, dominant_energy, low_freq_power, dominant_freq = self._frequency_features(
            state.norm_history + [norm]
        )

        feature = np.array(
            [
                direction_anomaly,
                magnitude_change,
                offset_score,
                rank_change,
                spectral_entropy_low,
                dominant_energy,
                low_freq_power,
            ],
            dtype=np.float32,
        )
        feature = np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(feature, 0.0, 1.0), dominant_freq

    def _frequency_features(self, values: List[float]) -> tuple[float, float, float, float]:
        if len(values) < 3:
            return 0.0, 0.0, 0.0, 0.0

        window = np.asarray(values[-self.max_window:], dtype=np.float32)
        if not np.isfinite(window).all() or np.max(np.abs(window)) <= self.eps:
            return 0.0, 0.0, 0.0, 0.0

        centered = window - float(np.mean(window))
        spectrum = np.fft.rfft(centered)
        psd = np.abs(spectrum) ** 2
        if psd.shape[0] <= 1:
            return 0.0, 0.0, 0.0, 0.0

        psd = psd[1:]  # discard DC component
        total = float(psd.sum())
        if total <= self.eps:
            return 0.0, 0.0, 0.0, 0.0

        prob = psd / (total + self.eps)
        entropy = -float(np.sum(prob * np.log(prob + self.eps))) / math.log(len(prob) + self.eps)
        dominant_idx = int(np.argmax(prob))
        dominant_energy = float(prob[dominant_idx])
        low_cut = max(1, int(math.ceil(len(prob) * 0.25)))
        low_freq_power = float(prob[:low_cut].sum())
        dominant_freq = float((dominant_idx + 1) / max(1, len(window)))

        return (
            float(np.clip(1.0 - entropy, 0.0, 1.0)),
            float(np.clip(dominant_energy, 0.0, 1.0)),
            float(np.clip(low_freq_power, 0.0, 1.0)),
            dominant_freq,
        )

    def _append_current_observation(
        self,
        state: ClientTrustState,
        signature: np.ndarray,
        norm: float,
        feature: np.ndarray,
        current_rank: int,
        dominant_freq: float,
    ) -> None:
        state.participation_count += 1
        state.last_seen_round = self._server_round
        state.signature_history.append(signature.copy())
        state.norm_history.append(float(norm))
        state.feature_history.append(feature.copy())
        state.rank_history.append(int(current_rank))
        state.dominant_freq_history.append(float(dominant_freq))
        self._trim(state.signature_history)
        self._trim(state.norm_history)
        self._trim(state.feature_history)
        self._trim(state.rank_history)
        self._trim(state.dominant_freq_history)

    def _update_trust(self, state: ClientTrustState, current_feature: np.ndarray) -> float:
        lr = (
            self.trust_update_rate_cold
            if state.participation_count <= self.cold_start_rounds
            else self.trust_update_rate_normal
        )

        adjusted_trusts: Dict[str, float] = {}
        for scale in SCALE_ORDER:
            window = self.windows[scale]
            recent_features = self._recent_feature_mean(state, window, current_feature)
            anomaly_score = float(np.dot(self.feature_weights[scale], recent_features))
            base_trust = self._sigmoid(self.trust_score_scale * (0.5 - anomaly_score))
            prev = state.scale_trust.get(scale, state.final_trust)
            smoothed = (1.0 - lr) * prev + lr * base_trust
            adjusted_trusts[scale] = float(np.clip(smoothed, 0.0, 1.0))

        self._apply_deep_penalties(state, adjusted_trusts)

        attention_weights = self._attention_weights(state, current_feature, adjusted_trusts)
        transition = self._transition_factor(state.participation_count)
        weights = (1.0 - transition) * self.fixed_scale_weights + transition * attention_weights
        weights = weights / (weights.sum() + self.eps)

        final_trust = 0.0
        for idx, scale in enumerate(SCALE_ORDER):
            trust = float(np.clip(adjusted_trusts[scale], 0.0, 1.0))
            state.scale_trust[scale] = trust
            state.scale_history.setdefault(scale, []).append(trust)
            self._trim(state.scale_history[scale])
            final_trust += float(weights[idx]) * trust

        state.final_trust = float(np.clip(final_trust, 0.0, 1.0))
        state.final_trust_history.append(state.final_trust)
        self._trim(state.final_trust_history)
        return state.final_trust

    def _recent_feature_mean(
        self,
        state: ClientTrustState,
        window: int,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if not state.feature_history:
            return fallback
        recent = state.feature_history[-window:]
        return np.mean(np.vstack(recent), axis=0).astype(np.float32)

    def _apply_deep_penalties(self, state: ClientTrustState, trusts: Dict[str, float]) -> None:
        if len(state.feature_history) >= self.direction_window:
            recent_dir = [float(f[0]) for f in state.feature_history[-self.direction_window:]]
            if all(v >= self.direction_threshold for v in recent_dir):
                trusts["short"] *= self.direction_penalty
                trusts["mid"] *= self.direction_penalty

        if len(state.feature_history) > self.offset_window:
            current_offset = float(state.feature_history[-1][2])
            past_offset = float(state.feature_history[-self.offset_window - 1][2])
            if current_offset - past_offset >= self.offset_growth_threshold:
                trusts["mid"] *= self.offset_penalty

        if len(state.feature_history) >= self.windows["short"]:
            low_entropy = float(state.feature_history[-1][4])
            freqs = np.asarray(state.dominant_freq_history[-self.windows["short"]:], dtype=np.float32)
            nonzero_freqs = freqs[freqs > 0.0]
            if (
                low_entropy >= self.periodic_entropy_threshold
                and nonzero_freqs.size >= 3
                and float(np.var(nonzero_freqs)) <= self.periodic_freq_var_threshold
            ):
                trusts["long"] *= self.periodic_penalty

        for scale in SCALE_ORDER:
            trusts[scale] = float(np.clip(trusts[scale], 0.0, 1.0))

    def _attention_weights(
        self,
        state: ClientTrustState,
        current_feature: np.ndarray,
        trusts: Dict[str, float],
    ) -> np.ndarray:
        query = current_feature.astype(np.float32)
        logits = []
        for scale in SCALE_ORDER:
            key = self._recent_feature_mean(state, self.windows[scale], query)
            logits.append(float(np.dot(query, key) / math.sqrt(FEATURE_DIM)))

        base = self._softmax(np.asarray(logits, dtype=np.float32))
        factors = []
        for scale in SCALE_ORDER:
            derivative = self._derivative_factor(state, scale)
            uncertainty = self._uncertainty_factor(state, scale)
            collaboration = self._collaboration_factor(state, scale, trusts[scale])
            factor = (
                derivative ** self.derivative_exp
                * uncertainty ** self.uncertainty_exp
                * collaboration ** self.collaboration_exp
            )
            factors.append(float(np.clip(factor, self.eps, 10.0)))

        combined = base * np.asarray(factors, dtype=np.float32)
        if combined.sum() <= self.eps or not np.isfinite(combined).all():
            combined = np.ones(len(SCALE_ORDER), dtype=np.float32)
        return combined / (combined.sum() + self.eps)

    def _derivative_factor(self, state: ClientTrustState, scale: str) -> float:
        history = state.scale_history.get(scale, [])
        window = min(self.windows[scale], len(history))
        if window < 2:
            return 1.0

        y = np.asarray(history[-window:], dtype=np.float32)
        x = np.arange(window, dtype=np.float32)
        x_center = x - float(np.mean(x))
        y_center = y - float(np.mean(y))
        denom = float(np.sum(x_center ** 2)) + self.eps
        slope = float(np.sum(x_center * y_center) / denom)
        if slope >= 0:
            return 1.0
        return float(math.exp(min(self.derivative_max, self.derivative_beta * (-slope))))

    def _uncertainty_factor(self, state: ClientTrustState, scale: str) -> float:
        history = state.scale_history.get(scale, [])
        window = min(self.windows[scale], len(history))
        if window < 2:
            return 1.0
        variance = float(np.var(np.asarray(history[-window:], dtype=np.float32)))
        return float(math.exp(-self.uncertainty_gamma * variance))

    def _collaboration_factor(self, state: ClientTrustState, scale: str, trust: float) -> float:
        history = state.scale_history.get(scale, [])
        window = min(self.windows[scale], len(history))
        if window < 3:
            return 1.0

        reference = np.asarray(history[-window:], dtype=np.float32)
        if float(np.std(reference)) <= self.eps:
            return 1.0

        boosts = []
        for other in self._states.values():
            if other is state or other.final_trust < self.high_trust_threshold:
                continue
            other_history = other.scale_history.get(scale, [])
            if len(other_history) < window:
                continue
            other_ref = np.asarray(other_history[-window:], dtype=np.float32)
            if float(np.std(other_ref)) <= self.eps:
                continue
            corr = float(np.corrcoef(reference, other_ref)[0, 1])
            if np.isfinite(corr) and corr > self.correlation_threshold and trust >= self.high_trust_threshold:
                boosts.append(corr - self.correlation_threshold)

        if not boosts:
            return 1.0
        return float(1.0 + self.collaboration_coeff * np.mean(boosts))

    def _transition_factor(self, count: int) -> float:
        start = self.transition_start_rounds
        end = max(start + 1, self.transition_end_rounds)
        return float(np.clip((count - start) / (end - start), 0.0, 1.0))

    def _aggregate_with_effective_weights(
        self,
        updates: UpdateList,
        effective_weights: Sequence[float],
    ) -> List[np.ndarray]:
        weights = np.asarray(effective_weights, dtype=np.float64)
        weights = np.maximum(weights, self.min_effective_weight)
        total = float(weights.sum())
        if total <= self.eps or not np.isfinite(total):
            weights = np.ones(len(updates), dtype=np.float64)
            total = float(weights.sum())

        assert self._global_params is not None
        result: List[np.ndarray] = []
        for param_idx, global_param in enumerate(self._global_params):
            dtype = global_param.dtype
            if np.issubdtype(dtype, np.floating):
                delta = np.zeros_like(global_param, dtype=np.float32)
                for weight, (params, _) in zip(weights, updates):
                    delta += (weight / total) * (
                        params[param_idx].astype(np.float32) - global_param.astype(np.float32)
                    )
                result.append((global_param.astype(np.float32) + delta).astype(dtype, copy=False))
            else:
                acc = np.zeros_like(global_param, dtype=np.float32)
                for weight, (params, _) in zip(weights, updates):
                    acc += (weight / total) * params[param_idx].astype(np.float32)
                result.append(np.round(acc).astype(dtype))
        return result

    def _record_round_metrics(
        self,
        client_ids: Sequence[str],
        trust_scores: Sequence[float],
        effective_weights: Sequence[float],
    ) -> None:
        trusts = np.asarray(trust_scores, dtype=np.float32)
        weights = np.asarray(effective_weights, dtype=np.float32)
        self.last_client_trusts = {cid: float(score) for cid, score in zip(client_ids, trust_scores)}
        self.last_client_weights = {cid: float(weight) for cid, weight in zip(client_ids, effective_weights)}
        self.last_round_metrics = {
            "time_consistency_trust_mean": float(np.mean(trusts)) if trusts.size else 0.0,
            "time_consistency_trust_min": float(np.min(trusts)) if trusts.size else 0.0,
            "time_consistency_trust_max": float(np.max(trusts)) if trusts.size else 0.0,
            "time_consistency_effective_weight_min": float(np.min(weights)) if weights.size else 0.0,
            "time_consistency_effective_weight_max": float(np.max(weights)) if weights.size else 0.0,
        }

    def _trim(self, values: List) -> None:
        overflow = len(values) - self.max_history
        if overflow > 0:
            del values[:overflow]

    def _sigmoid(self, x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + math.exp(-x))

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        shifted = values - float(np.max(values))
        exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
        total = float(exp_values.sum())
        if total <= self.eps:
            return np.ones_like(exp_values) / len(exp_values)
        return exp_values / total
