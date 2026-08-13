"""Metrics helpers for RTC-v2."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .history import RoundRecord


def build_round_metrics(
    records: Sequence[RoundRecord],
    *,
    clip_norm: float,
    fallback_used: bool,
    fallback_reason: str,
    weight_sum: float = 0.0,
    zero_mass: float = 1.0,
    max_cap_violation: float = 0.0,
    max_budget_violation: float = 0.0,
) -> Dict[str, float | str]:
    if not records:
        return {}
    trusts = np.asarray([r.trust for r in records], dtype=np.float64)
    weights = np.asarray([r.effective_weight for r in records], dtype=np.float64)
    metrics: Dict[str, float | str] = {
        "time_consistency_version": "rtc_v2",
        "time_consistency_trust_mean": float(np.mean(trusts)),
        "time_consistency_trust_min": float(np.min(trusts)),
        "time_consistency_trust_max": float(np.max(trusts)),
        "time_consistency_effective_weight_min": float(np.min(weights)),
        "time_consistency_effective_weight_max": float(np.max(weights)),
        "time_consistency_clipped_clients": float(sum(r.clipped for r in records)),
        "time_consistency_clip_norm": float(clip_norm),
        "time_consistency_quarantined_clients": float(sum(r.quarantined for r in records)),
        "time_consistency_capped_clients": float(sum(r.capped for r in records)),
        "time_consistency_fallback_used": float(fallback_used),
        "time_consistency_fallback_reason": fallback_reason,
        "rtc_magnitude_risk_mean": float(np.mean([r.magnitude_risk for r in records])),
        "rtc_direction_risk_mean": float(np.mean([r.direction_risk for r in records])),
        "rtc_temporal_risk_mean": float(np.mean([r.temporal_risk for r in records])),
        "rtc_influence_risk_mean": float(np.mean([r.influence_risk for r in records])),
        "rtc_influence_attempt_mean": float(np.mean([r.influence_attempt for r in records])),
        "rtc_influence_effective_mean": float(np.mean([r.influence_effective for r in records])),
        "rtc_total_risk_mean": float(np.mean([r.total_risk for r in records])),
        "rtc_event_risk_mean": float(np.mean([r.event_risk for r in records])),
        "rtc_num_effective_clients": float(sum(r.aggregation_weight > 1e-12 for r in records)),
        "rtc_num_watch": float(sum(r.state == "watch" for r in records)),
        "rtc_num_restricted": float(sum(r.state == "restricted" for r in records)),
        "rtc_num_quarantined": float(sum(r.state == "quarantined" for r in records)),
        "rtc_aggregation_weight_sum": float(weight_sum),
        "rtc_zero_update_mass": float(zero_mass),
        "rtc_max_weight_cap_violation": float(max_cap_violation),
        "rtc_max_exposure_budget_violation": float(max_budget_violation),
        "rtc_client_budget_remaining_min": _finite_min([r.client_budget_remaining for r in records]),
        "rtc_direction_budget_remaining_min": _finite_min([r.direction_budget_remaining for r in records]),
        # Backward-compatible smoke columns.
        "fft_periodic_penalty_clients": float(sum(r.temporal_risk >= 0.7 for r in records)),
        "direction_penalty_clients": float(sum(r.direction_risk >= 0.7 for r in records)),
        "mean_dominant_frequency_energy": float(np.mean([r.dominant_frequency for r in records])),
        "mean_direction_anomaly": float(np.mean([r.direction_risk for r in records])),
        "mean_low_frequency_power": float(np.mean([r.temporal_risk for r in records])),
    }
    return metrics


def _finite_min(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else -1.0
