"""Phase 6 gates for conjunctive direction-persistence tightening."""

from __future__ import annotations

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import build_manifest
from defenses.rtc.v3 import RTCv3Defense


MAPPING = {"attacker": "p", "good-a": "a", "good-b": "b"}


def _params(value=0.0):
    return [np.asarray([value], dtype=np.float32)]


def _manifest(
    *,
    direction_enabled=True,
    z_threshold=0.5,
    min_matches=2,
    server_usage_threshold=0.1,
    principal_usage_threshold=0.5,
):
    stable = {
        "enabled": True,
        "dimension": 32,
        "seed": 23,
        "max_prototypes": 16,
        "match_threshold": 0.85,
        "momentum": 0.8,
        "retirement_rounds": 16,
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
                "1": {"anchor": 100.0, "residual": 2.0, "total": 100.0}
            }
        },
        principal_windows=(1,),
        principal_betas={"full": {"1": 1.0}},
        stable_cones=stable,
        cone_budgets={"full": {"1": 2.0}},
        cumulative={
            "enabled": False,
            "resolutions": {
                "full": {
                    "scale_center": 1.0,
                    "scale_lower": 1.0,
                    "scale_upper": 1.0,
                    "kappa": 100.0,
                    "threshold": 100.0,
                    "eta": 0.0,
                    "q_min": 1.0,
                }
            },
        },
        direction_persistence={
            "enabled": direction_enabled,
            "resolutions": {
                "full": {
                    "z_threshold": z_threshold,
                    "history_length": 3,
                    "min_matches": min_matches,
                    "server_window": 1,
                    "principal_window": 1,
                    "server_usage_threshold": server_usage_threshold,
                    "principal_usage_threshold": principal_usage_threshold,
                    "q_value": 0.5,
                }
            },
        },
    )


def _defense(manifest, phase=6):
    return RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": manifest,
                "implementation_phase": phase,
                "principal_map": MAPPING,
                "principal_first_sampling_verified": True,
            },
        )
    )


def _round(defense, server_round, global_value, attacker_delta, ids=None):
    ids = ids or ["attacker", "good-a", "good-b"]
    deltas = {"attacker": attacker_delta, "good-a": 0.0, "good-b": 0.0}
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


def _three_rounds(defense, deltas=(0.8, 0.8, 0.8)):
    value = 0.0
    for server_round, delta in enumerate(deltas, start=1):
        value = _round(defense, server_round, value, delta)
    return value


def test_direction_tightens_only_after_all_five_conditions_hold():
    enabled = _defense(_manifest(direction_enabled=True))
    baseline = _defense(_manifest(direction_enabled=False))
    enabled_value = _three_rounds(enabled)
    baseline_value = _three_rounds(baseline)

    assert enabled.last_round_metrics["rtc_v3_direction_q_min"] == pytest.approx(0.5)
    assert enabled.last_round_metrics["rtc_v3_direction_active_count"] >= 1.0
    assert enabled.last_client_aggregation_weights["attacker"] < 1.0 / 3.0
    assert baseline.last_client_aggregation_weights["attacker"] == pytest.approx(
        1.0 / 3.0
    )
    assert enabled_value < baseline_value


@pytest.mark.parametrize(
    "manifest_kwargs,deltas",
    [
        ({"z_threshold": 0.9}, (0.8, 0.8, 0.8)),
        ({"min_matches": 3}, (0.8, 0.8, 0.8)),
        ({"server_usage_threshold": 0.9}, (0.8, 0.8, 0.8)),
        ({"principal_usage_threshold": 0.9}, (0.8, 0.8, 0.8)),
        ({}, (0.8, -0.8, 0.8)),
    ],
)
def test_each_missing_joint_condition_keeps_direction_q_at_one(
    manifest_kwargs, deltas
):
    defense = _defense(_manifest(**manifest_kwargs))
    _three_rounds(defense, deltas)

    assert defense.last_round_metrics["rtc_v3_direction_q_min"] == pytest.approx(1.0)
    assert defense.last_round_metrics["rtc_v3_direction_active_count"] == 0.0


def test_absence_does_not_create_direction_repetition():
    defense = _defense(_manifest())
    value = _round(defense, 1, 0.0, 0.8)
    value = _round(defense, 2, value, 0.0, ids=["good-a", "good-b"])
    value = _round(defense, 3, value, 0.0, ids=["good-a", "good-b"])
    _round(defense, 4, value, 0.8)

    assert defense.last_round_metrics["rtc_v3_direction_q_min"] == pytest.approx(1.0)


def test_direction_is_default_off_and_rejected_before_phase6():
    with pytest.raises(ValueError, match="only be enabled in Phase 6"):
        _defense(_manifest(direction_enabled=True), phase=5)


def test_phase6_checkpoint_roundtrip_preserves_all_temporal_state():
    manifest = _manifest()
    original = _defense(manifest)
    value = _round(original, 1, 0.0, 0.8)
    value = _round(original, 2, value, 0.8)
    state = original.state_dict()

    restored = _defense(manifest)
    restored.load_state_dict(state)
    original_value = _round(original, 3, value, 0.8)
    restored_value = _round(restored, 3, value, 0.8)

    assert restored_value == pytest.approx(original_value)
    assert restored.last_round_metrics["rtc_v3_direction_q_min"] == pytest.approx(
        original.last_round_metrics["rtc_v3_direction_q_min"]
    )
    assert restored._server_ledger.snapshot() == original._server_ledger.snapshot()
    assert (
        restored._principal_ledger.snapshot()
        == original._principal_ledger.snapshot()
    )

    mismatched = dict(state)
    mismatched["calibration_hash"] = "wrong"
    with pytest.raises(ValueError, match="calibration_hash"):
        restored.load_state_dict(mismatched)
