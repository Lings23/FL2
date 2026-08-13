"""Phase 2 gates for real server-round multi-window accounting."""

from __future__ import annotations

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import build_manifest
from defenses.rtc.v3 import RTCv3Defense


def _params(value: float = 0.0) -> list[np.ndarray]:
    return [np.asarray([value], dtype=np.float32)]


def _defense(manifest: dict) -> RTCv3Defense:
    return RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": manifest,
                "implementation_phase": 2,
            },
        )
    )


def _manifest(**extra) -> dict:
    return build_manifest(
        params=_params(),
        clip_lower=10.0,
        clip_upper=10.0,
        residual_scales={"full": extra.pop("residual_scale", 1.0)},
        server_windows=(1, 3),
        server_budgets=extra.pop(
            "server_budgets",
            {
                "full": {
                    "1": {"anchor": 10.0, "residual": 10.0, "total": 10.0},
                    "3": {"anchor": 1.5, "residual": 10.0, "total": 1.5},
                }
            },
        ),
        **extra,
    )


def _round(
    defense: RTCv3Defense,
    server_round: int,
    global_value: float,
    delta: float = 1.0,
) -> tuple[float, float]:
    global_params = _params(global_value)
    defense.set_context(
        server_round,
        ["client"],
        global_params,
        server_optimizer="fedavg",
    )
    result = defense.aggregate([(_params(global_value + delta), 1)])
    return float(result[0][0]), defense.last_client_aggregation_weights["client"]


def test_long_window_limits_repeated_subthreshold_updates():
    defense = _defense(_manifest())
    value, weight1 = _round(defense, 1, 0.0)
    value, weight2 = _round(defense, 2, value)
    value, weight3 = _round(defense, 3, value)

    assert weight1 == pytest.approx(1.0)
    assert weight2 == pytest.approx(0.5, abs=1e-7)
    assert weight3 == pytest.approx(0.0, abs=1e-7)
    assert value == pytest.approx(1.5, abs=1e-7)
    assert defense.last_round_metrics["rtc_v3_full_total_used_w3"] <= 1.5 + 1e-8


def test_server_window_advances_by_real_round_not_append_count():
    defense = _defense(_manifest())
    value, _ = _round(defense, 1, 0.0)
    value, weight = _round(defense, 4, value)

    assert weight == pytest.approx(1.0)
    assert value == pytest.approx(2.0)
    assert defense.last_round_metrics["rtc_v3_full_total_history_w3"] == 0.0


def test_training_phase_boundary_uses_minimum_budget_without_clearing_history():
    budgets = {
        "full": {
            "1": {
                "anchor": {"early": 10.0, "late": 10.0},
                "residual": {"early": 10.0, "late": 10.0},
                "total": {"early": 10.0, "late": 10.0},
            },
            "3": {
                "anchor": {"early": 2.0, "late": 0.75},
                "residual": {"early": 10.0, "late": 10.0},
                "total": {"early": 2.0, "late": 0.75},
            },
        }
    }
    defense = _defense(
        _manifest(
            residual_scale={"early": 1.0, "late": 1.0},
            server_budgets=budgets,
            training_phases=[
                {"start_round": 1, "profile": "early"},
                {"start_round": 2, "profile": "late"},
            ],
        )
    )
    value, first_weight = _round(defense, 1, 0.0)
    value, second_weight = _round(defense, 2, value)

    assert first_weight == pytest.approx(1.0)
    assert second_weight == pytest.approx(0.0)
    assert value == pytest.approx(1.0)
    assert defense.last_round_metrics["rtc_v3_full_total_history_w3"] == pytest.approx(
        1.0
    )
    assert defense.last_round_metrics["rtc_v3_full_total_budget_w3"] == pytest.approx(
        0.75
    )


def test_clip_bounds_follow_frozen_training_profile():
    defense = _defense(
        build_manifest(
            params=_params(),
            profile="early",
            clip_lower={"early": 1.0, "late": 2.0},
            clip_upper={"early": 1.0, "late": 2.0},
            residual_scales={"full": {"early": 1.0, "late": 1.0}},
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
            training_phases=[
                {"start_round": 1, "profile": "early"},
                {"start_round": 2, "profile": "late"},
            ],
        )
    )
    value, _ = _round(defense, 1, 0.0, delta=10.0)
    assert value == pytest.approx(1.0)
    value, _ = _round(defense, 2, value, delta=10.0)

    assert value == pytest.approx(3.0)
    assert defense.last_round_metrics["rtc_v3_clip_norm"] == pytest.approx(2.0)
