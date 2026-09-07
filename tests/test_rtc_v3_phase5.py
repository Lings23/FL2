"""Phase 5 gates for optional one-sided principal cumulative evidence."""

from __future__ import annotations

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import build_manifest
from defenses.rtc.v3 import RTCv3Defense


MAPPING = {"attacker": "p", "good-a": "a", "good-b": "b"}


def _params(value=0.0):
    return [np.asarray([value], dtype=np.float32)]


def _manifest(*, cumulative_enabled):
    extra = {}
    if cumulative_enabled:
        extra["cumulative"] = {
            "enabled": True,
            "resolutions": {
                "full": {
                    "scale_center": 1.0,
                    "scale_lower": 1.0,
                    "scale_upper": 1.0,
                    "kappa": 0.5,
                    "threshold": 0.5,
                    "eta": 2.0,
                    "q_min": 0.2,
                }
            },
        }
    return build_manifest(
        params=_params(),
        clip_lower=10.0,
        clip_upper=10.0,
        resolutions=("full",),
        residual_scales={"full": 1.0},
        server_windows=(1,),
        server_budgets={
            "full": {
                "1": {
                    "anchor": 100.0,
                    "residual": 100.0,
                    "total": 100.0,
                }
            }
        },
        principal_windows=(1,),
        principal_betas={"full": {"1": 1.0}},
        **extra,
    )


def _defense(*, cumulative_enabled, phase=5, cumulative_q_cap_power=0):
    return RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": _manifest(
                    cumulative_enabled=cumulative_enabled
                ),
                "implementation_phase": phase,
                "principal_map": MAPPING,
                "principal_first_sampling_verified": True,
                "cumulative_q_cap_power": cumulative_q_cap_power,
            },
        )
    )


def _round(defense, server_round, global_value, attacker_delta, ids=None):
    ids = ids or ["attacker", "good-a", "good-b"]
    deltas = {
        "attacker": attacker_delta,
        "attacker-2": attacker_delta,
        "good-a": 0.0,
        "good-b": 0.0,
    }
    defense.set_context(
        server_round,
        ids,
        _params(global_value),
        server_optimizer="fedavg",
    )
    result = defense.aggregate(
        [(_params(global_value + deltas[client_id]), 1) for client_id in ids]
    )
    return float(result[0][0])


def test_cumulative_evidence_adds_restriction_beyond_phase4_window():
    phase4 = _defense(cumulative_enabled=False, phase=4)
    phase5 = _defense(cumulative_enabled=True, phase=5)
    value4 = value5 = 0.0
    for server_round in (1, 2, 3):
        value4 = _round(phase4, server_round, value4, 0.8)
        value5 = _round(phase5, server_round, value5, 0.8)

    assert phase4.last_client_aggregation_weights["attacker"] == pytest.approx(
        1.0 / 3.0
    )
    assert phase5.last_client_aggregation_weights["attacker"] < 1.0 / 3.0
    assert phase5.last_round_metrics["rtc_v3_cumulative_q_min"] < 1.0
    assert value5 < value4


def test_linear_cumulative_q_client_cap_is_applied_without_renormalization():
    baseline = _defense(cumulative_enabled=True)
    capped = _defense(cumulative_enabled=True, cumulative_q_cap_power=1)
    baseline_value = capped_value = 0.0
    for server_round in (1, 2, 3):
        baseline_value = _round(baseline, server_round, baseline_value, 0.8)
        capped_value = _round(capped, server_round, capped_value, 0.8)

    q = capped._last_client_q_cap[0]
    assert q < 1.0
    assert capped._last_client_q_cap[0] == pytest.approx(q)
    assert capped.last_client_aggregation_weights["attacker"] <= (
        (1.0 / 3.0) * q + 1e-8
    )
    assert capped.last_round_metrics["rtc_v3_cumulative_q_cap_power"] == 1.0
    assert capped.last_round_metrics["rtc_v3_client_q_cap_active_count"] >= 1.0
    assert capped_value <= baseline_value


@pytest.mark.parametrize("value", [-1, 1.5, 3, float("nan")])
def test_cumulative_q_cap_power_is_validated(value):
    with pytest.raises((ValueError, OverflowError), match="cumulative_q_cap_power"):
        _defense(
            cumulative_enabled=True,
            cumulative_q_cap_power=value,
        )


def test_cumulative_q_cap_requires_enabled_cumulative_evidence():
    with pytest.raises(ValueError, match="requires cumulative evidence"):
        _defense(cumulative_enabled=False, cumulative_q_cap_power=1)


def test_one_ordinary_tail_observation_does_not_jump_to_q_min():
    defense = _defense(cumulative_enabled=True)
    _round(defense, 1, 0.0, 0.8)

    observation_q = defense.last_round_metrics["rtc_v3_cumulative_q_min"]
    assert observation_q == pytest.approx(1.0)
    assert observation_q > 0.2


def test_absence_does_not_reduce_or_advance_cumulative_state():
    defense = _defense(cumulative_enabled=True)
    value = _round(defense, 1, 0.0, 0.8)
    before = defense._cumulative.statistic("p", "full")
    value = _round(defense, 2, value, 0.0, ids=["good-a", "good-b"])
    value = _round(defense, 3, value, 0.0, ids=["good-a", "good-b"])

    assert defense._cumulative.statistic("p", "full") == pytest.approx(before)


def test_cumulative_is_default_off_and_cannot_be_enabled_before_phase5():
    defense = _defense(cumulative_enabled=False)
    _round(defense, 1, 0.0, 0.8)
    assert "rtc_v3_cumulative_q_min" not in defense.last_round_metrics

    with pytest.raises(ValueError, match="only be enabled in Phase 5"):
        _defense(cumulative_enabled=True, phase=4)
