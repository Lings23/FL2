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
        self.peak_risk_weight = float(np.clip(params.get("peak_risk_weight", 0.75), 0.0, 1.0))
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

        weight_sum = float(np.sum(np.maximum(raw_weights, 0.0)))
        group_center_sig = np.median(signatures, axis=0).astype(np.float32)
        group_norm_history = norms.tolist()

        if weight_sum > self.eps:
            shares = np.maximum(raw_weights, 0.0) / weight_sum
        else:
            shares = np.full(len(records), 1.0 / len(records), dtype=np.float64)
        if records[0].delta_flat.size:
            nominal_update = np.zeros_like(records[0].delta_flat, dtype=np.float64)
            for share, item in zip(shares, records):
                nominal_update += float(share) * item.delta_flat.astype(np.float64, copy=False)
        else:
            nominal_update = np.zeros(0, dtype=np.float64)

        influence_attempts: List[float] = []
        for share, item in zip(shares, records):
            if len(records) <= 1 or share >= 1.0 - self.eps:
                attempt = 0.0
            else:
                residual = item.delta_flat.astype(np.float64, copy=False) - nominal_update
                attempt = float(share / max(1.0 - share, self.eps) * np.linalg.norm(residual))
            influence_attempts.append(max(0.0, attempt))

        for record, attempt, share in zip(records, influence_attempts, shares):
            history = histories[record.client_id]
            record.raw_weight_share = float(share)
            record.norm_share = float(record.norm / (float(np.sum(norms)) + self.eps))
            record.influence_attempt = float(attempt)
            record.impact_proxy = float(attempt)
            record.magnitude_risk = self._magnitude_risk(record, history, group_norm_history)
            record.direction_risk = self._direction_risk(record, history, group_center_sig)
            record.influence_risk = self._influence_risk(record, history, influence_attempts)
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
        watch_threshold = float(params.get("watch_threshold", 0.55))
        restricted_threshold = float(params.get("restricted_threshold", 0.70))
        quarantine_threshold = float(params.get("quarantine_threshold", 0.80))
        immediate_quarantine_threshold = float(params.get("immediate_quarantine_threshold", 0.95))
        quarantine_window = max(1, int(params.get("quarantine_window", 4)))
        quarantine_strikes = max(1, int(params.get("quarantine_strikes", 2)))
        recovery_threshold = float(params.get("recovery_threshold", 0.35))
        recovery_observations = max(1, int(params.get("recovery_observations", 3)))
        cooldown_rounds = int(params.get("quarantine_cooldown_rounds", 2))

        observation_trust = 1.0 - float(np.clip(record.total_risk, 0.0, 1.0))
        history.final_trust = float(np.clip(beta * history.final_trust + (1.0 - beta) * observation_trust, min_trust, 1.0))

        recent = list(history.risk_history[-max(0, quarantine_window - 1):]) + [record.total_risk]
        quarantine_count = sum(1 for risk in recent if risk >= quarantine_threshold)
        if record.total_risk < recovery_threshold:
            history.low_risk_streak += 1
        else:
            history.low_risk_streak = 0
        recovered = history.low_risk_streak >= recovery_observations

        current_round = int(server_round) if server_round is not None else max(1, history.last_seen_round + 1)
        if record.total_risk >= restricted_threshold:
            desired_state = "restricted"
        elif record.total_risk >= watch_threshold:
            desired_state = "watch"
        else:
            desired_state = "normal"

        if record.total_risk >= immediate_quarantine_threshold or quarantine_count >= quarantine_strikes:
            state = "quarantined"
            history.cooldown_until = max(history.cooldown_until, current_round + cooldown_rounds)
        elif history.state == "quarantined" and (current_round <= history.cooldown_until or not recovered):
            state = "quarantined"
        elif history.state == "quarantined":
            state = "restricted"
        else:
            ranks = {"normal": 0, "watch": 1, "restricted": 2, "quarantined": 3}
            previous_rank = ranks.get(history.state, 0)
            desired_rank = ranks[desired_state]
            if desired_rank < previous_rank:
                desired_rank = previous_rank - 1
            state = ("normal", "watch", "restricted", "quarantined")[desired_rank]

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

    def _influence_risk(
        self,
        record: RoundRecord,
        history: ClientHistory,
        influence_attempts: Sequence[float],
    ) -> float:
        if not self.enable_influence:
            return 0.0
        value = math.log(max(record.influence_attempt, 0.0) + self.eps)
        group_values = [math.log(max(item, 0.0) + self.eps) for item in influence_attempts]
        group_risk = self._robust_upper_z_risk(value, group_values)
        own_risk = 0.0
        if len(history.impact_history) >= self.min_history_for_self_score:
            own_values = [math.log(max(item, 0.0) + self.eps) for item in history.impact_history]
            own_risk = self._robust_upper_z_risk(value, own_values)
        return float(max(group_risk, own_risk))

    def _combine(self, record: RoundRecord) -> float:
        weights = self.risk_weights
        weighted = (
            weights["magnitude"] * record.magnitude_risk
            + weights["direction"] * record.direction_risk
            + weights["temporal"] * record.temporal_risk
            + weights["influence"] * record.influence_risk
        )
        peak = max(
            record.magnitude_risk,
            record.direction_risk,
            record.temporal_risk,
            record.influence_risk,
        )
        value = (1.0 - self.peak_risk_weight) * weighted + self.peak_risk_weight * peak
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
        weighted = sum(weights[name] * risk for name, risk in enabled.items()) / denominator
        peak = max(enabled.values())
        value = (1.0 - self.peak_risk_weight) * weighted + self.peak_risk_weight * peak
        return float(np.clip(value, 0.0, 1.0))

    def _robust_upper_z_risk(self, value: float, values: Sequence[float]) -> float:
        """One-sided robust score: only unusually large attempted influence is risky."""
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2 or not np.isfinite(value):
            return 0.0
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        scale = 1.4826 * mad
        if scale <= self.eps:
            scale = max(abs(median) * 0.1, 1e-6)
        z = max(0.0, float(value) - median) / scale
        return float(np.clip(z / self.robust_z_threshold, 0.0, 1.0))

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
