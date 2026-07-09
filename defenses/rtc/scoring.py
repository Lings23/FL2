"""Risk scoring for RTC-v2."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .history import ClientHistory, RoundRecord


class RiskScorer:
    """Round-aware, explainable RTC-v2 risk scorer."""

    def __init__(self, params: Mapping[str, object]):
        self.eps = float(params.get("eps", 1e-9))
        self.min_history_for_self_score = int(params.get("min_history_for_self_score", 5))
        self.min_temporal_points = int(params.get("min_temporal_points", 8))
        self.high_risk_window = int(params.get("high_risk_window", 6))
        self.high_risk_threshold = float(params.get("high_risk_threshold", 0.70))
        self.medium_risk_threshold = float(params.get("medium_risk_threshold", 0.40))
        self.robust_z_threshold = float(params.get("robust_z_threshold", 3.0))
        self.enable_direction = bool(params.get("enable_direction_features", params.get("enable_direction", True)))
        self.enable_temporal = bool(params.get("enable_temporal_features", params.get("enable_fft_features", True)))
        self.enable_influence = bool(params.get("enable_influence_features", True))
        self.periodic_gap_tolerance = float(params.get("periodic_gap_tolerance", 1.0))
        self.min_periodic_events = int(params.get("min_periodic_events", 4))
        self.risk_weights = self._risk_weights(params.get("risk_weights"))

    def score_records(
        self,
        records: Sequence[RoundRecord],
        histories: Mapping[str, ClientHistory],
        server_round: int,
    ) -> None:
        if not records:
            return
        norms = np.asarray([r.norm for r in records], dtype=np.float64)
        signatures = np.vstack([r.signature for r in records]).astype(np.float32, copy=False)
        raw_weights = np.asarray([r.raw_weight for r in records], dtype=np.float64)

        norm_sum = float(np.sum(norms)) + self.eps
        weight_sum = float(np.sum(raw_weights)) + self.eps
        group_center_sig = np.median(signatures, axis=0).astype(np.float32)
        group_norm_history = norms.tolist()

        for record in records:
            history = histories[record.client_id]
            record.raw_weight_share = float(record.raw_weight / weight_sum)
            record.norm_share = float(record.norm / norm_sum)
            record.impact_proxy = float(record.raw_weight_share * record.norm)
            record.magnitude_risk = self._magnitude_risk(record, history, group_norm_history)
            record.direction_risk = self._direction_risk(record, history, group_center_sig)
            record.influence_risk = self._influence_risk(record, records)
            # Evaluate temporal evidence only after the instantaneous signals
            # exist. Otherwise the current event is always observed as zero.
            record.event_risk = self._instantaneous_risk(record)
            record.temporal_risk, record.dominant_frequency = self._temporal_risk(
                record, history, server_round, current_event_risk=record.event_risk
            )
            record.total_risk = self._combine(record)

    def update_state_and_trust(
        self,
        record: RoundRecord,
        history: ClientHistory,
        params: Mapping[str, object],
        server_round: int | None = None,
    ) -> None:
        beta = float(params.get("trust_beta", params.get("beta", 0.75)))
        min_trust = float(params.get("min_trust", 0.05))
        watch_threshold = float(params.get("watch_threshold", 0.25))
        restricted_threshold = float(params.get("restricted_threshold", 0.45))
        quarantine_threshold = float(params.get("quarantine_threshold", 0.80))
        cooldown_rounds = int(params.get("quarantine_cooldown_rounds", 2))

        observation_trust = 1.0 - float(np.clip(record.total_risk, 0.0, 1.0))
        history.final_trust = float(np.clip(beta * history.final_trust + (1.0 - beta) * observation_trust, min_trust, 1.0))

        recent = list(history.risk_history[-max(1, self.high_risk_window - 1):]) + [record.total_risk]
        high_count = sum(1 for risk in recent if risk >= self.high_risk_threshold)
        medium_count = sum(1 for risk in recent if risk >= self.medium_risk_threshold)

        current_round = int(server_round) if server_round is not None else max(1, history.last_seen_round + 1)
        state = "normal"
        if record.total_risk >= watch_threshold or medium_count >= 2:
            state = "watch"
        if record.total_risk >= restricted_threshold or high_count >= 2:
            state = "restricted"
        if (
            record.total_risk >= quarantine_threshold
            and (record.temporal_risk >= restricted_threshold or record.influence_risk >= restricted_threshold)
        ):
            state = "quarantined"
            history.cooldown_until = max(history.cooldown_until, current_round + cooldown_rounds)
        elif current_round <= history.cooldown_until:
            state = "quarantined"

        history.state = state
        record.state = state
        record.trust = history.final_trust
        record.quarantined = state == "quarantined"

    def _magnitude_risk(self, record: RoundRecord, history: ClientHistory, group_norms: Sequence[float]) -> float:
        group_risk = self._robust_z_risk(record.norm, group_norms)
        own_risk = 0.0
        if len(history.norm_history) >= self.min_history_for_self_score:
            own_risk = self._robust_z_risk(record.norm, history.norm_history)
        return max(group_risk, own_risk)

    def _direction_risk(self, record: RoundRecord, history: ClientHistory, group_center_sig: np.ndarray) -> float:
        if not self.enable_direction:
            return 0.0
        group_shift = self._cosine_distance(record.signature, group_center_sig)
        own_shift = 0.0
        if history.signature_history:
            recent = np.mean(np.vstack(history.signature_history[-min(5, len(history.signature_history)):]), axis=0)
            own_shift = self._cosine_distance(record.signature, recent)
        if history.signature_history:
            return float(np.clip(0.5 * own_shift + 0.5 * group_shift, 0.0, 1.0))
        return float(np.clip(group_shift, 0.0, 1.0))

    def _temporal_risk(
        self,
        record: RoundRecord,
        history: ClientHistory,
        server_round: int,
        current_event_risk: float | None = None,
    ) -> tuple[float, float]:
        if not self.enable_temporal:
            return 0.0, 0.0
        event_risk = float(record.total_risk) if current_event_risk is None else float(current_event_risk)
        periodic, dominant_frequency = self._periodic_gap_risk(history, event_risk, server_round)
        return float(np.clip(periodic, 0.0, 1.0)), dominant_frequency

    def _periodic_gap_risk(self, history: ClientHistory, current_risk: float, server_round: int) -> tuple[float, float]:
        # Historical periodicity may amplify a *current* anomaly, but it must
        # not latch forever after the current client update returns to normal.
        if current_risk < self.high_risk_threshold:
            return 0.0, 0.0
        historical_events = (
            history.event_risk_history
            if len(history.event_risk_history) == len(history.rounds) and history.event_risk_history
            else history.risk_history
        )
        risk_pairs = list(zip(history.rounds, historical_events))
        risk_pairs.append((int(server_round), float(current_risk)))
        high_rounds = [rnd for rnd, risk in risk_pairs if risk >= self.high_risk_threshold]
        if len(high_rounds) < self.min_periodic_events:
            return 0.0, 0.0
        gaps = np.diff(np.asarray(high_rounds[-self.min_periodic_events:], dtype=np.float64))
        if gaps.size < self.min_periodic_events - 1 or np.any(gaps <= 0):
            return 0.0, 0.0
        median_gap = float(np.median(gaps))
        mad_gap = float(np.median(np.abs(gaps - median_gap)))
        if median_gap <= 0:
            return 0.0, 0.0
        if mad_gap >= self.periodic_gap_tolerance:
            return 0.0, 0.0
        else:
            consistency = 1.0 - mad_gap / max(self.periodic_gap_tolerance, self.eps)
        dominant_frequency = 1.0 / median_gap
        return float(np.clip(consistency, 0.0, 1.0)), float(dominant_frequency)

    def _influence_risk(self, record: RoundRecord, records: Sequence[RoundRecord]) -> float:
        if not self.enable_influence:
            return 0.0
        n = max(1, len(records))
        fair_share = 1.0 / n
        weight_risk = max(0.0, record.raw_weight_share - fair_share) / (fair_share + self.eps)
        norm_risk = max(0.0, record.norm_share - fair_share) / (fair_share + self.eps)
        impacts = [r.impact_proxy for r in records]
        impact_risk = self._robust_z_risk(record.impact_proxy, impacts)
        return float(np.clip(max(weight_risk, norm_risk, impact_risk), 0.0, 1.0))

    def _combine(self, record: RoundRecord) -> float:
        weights = self.risk_weights
        value = (
            weights["magnitude"] * record.magnitude_risk
            + weights["direction"] * record.direction_risk
            + weights["temporal"] * record.temporal_risk
            + weights["influence"] * record.influence_risk
        )
        return float(np.clip(value, 0.0, 1.0))

    def _instantaneous_risk(self, record: RoundRecord) -> float:
        """Fuse non-temporal evidence without diluting it by temporal weight."""
        weights = self.risk_weights
        enabled = {
            "magnitude": record.magnitude_risk,
            "direction": record.direction_risk,
            "influence": record.influence_risk,
        }
        denominator = sum(weights[name] for name in enabled)
        if denominator <= self.eps:
            return 0.0
        value = sum(weights[name] * risk for name, risk in enabled.items()) / denominator
        return float(np.clip(value, 0.0, 1.0))

    def _robust_z_risk(self, value: float, values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return 0.0
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        scale = 1.4826 * mad
        if scale <= self.eps:
            scale = max(float(np.std(arr)), self.eps)
        z = abs(float(value) - median) / scale
        return float(np.clip(z / self.robust_z_threshold, 0.0, 1.0))

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= self.eps:
            return 0.0
        cosine = float(np.dot(a, b) / denom)
        return float((1.0 - np.clip(cosine, -1.0, 1.0)) / 2.0)

    @staticmethod
    def _risk_weights(raw: object) -> Dict[str, float]:
        defaults = {"magnitude": 0.30, "direction": 0.25, "temporal": 0.30, "influence": 0.15}
        if isinstance(raw, Mapping):
            for key in list(defaults):
                if key in raw:
                    defaults[key] = max(0.0, float(raw[key]))
        total = sum(defaults.values())
        if total <= 0.0 or not math.isfinite(total):
            return {"magnitude": 0.30, "direction": 0.25, "temporal": 0.30, "influence": 0.15}
        return {key: value / total for key, value in defaults.items()}
