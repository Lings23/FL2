"""Phase 3 gates for fixed multi-resolution residuals and stable cones."""

from __future__ import annotations

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import build_manifest
from defenses.rtc.prototypes import StablePrototypeBank
from defenses.rtc.resolutions import ResolutionLayout
from defenses.rtc.v3 import RTCv3Defense


ROLES = {"0": "body", "1": "classifier_head", "2": "body"}
RESOLUTIONS = ("full", "head", "block:0", "overflow")


def _params(
    body=(0.0, 0.0, 0.0, 0.0),
    head=(0.0, 0.0),
    tail=(0.0,),
    counter=3,
):
    return [
        np.asarray(body, dtype=np.float32),
        np.asarray(head, dtype=np.float32),
        np.asarray(tail, dtype=np.float32),
        np.asarray([counter], dtype=np.int64),
    ]


def _budget(value=100.0):
    return {
        resolution: {
            "1": {"anchor": value, "residual": value, "total": value}
        }
        for resolution in RESOLUTIONS
    }


def _manifest(*, budgets=None, cones=False, cone_budget=100.0):
    extra = {}
    if cones:
        extra = {
            "stable_cones": {
                "enabled": True,
                "dimension": 32,
                "seed": 17,
                "max_prototypes": 16,
                "match_threshold": 0.85,
                "momentum": 0.8,
                "retirement_rounds": 16,
            },
            "cone_budgets": {
                resolution: {"1": cone_budget}
                for resolution in RESOLUTIONS
            },
        }
    return build_manifest(
        params=_params(),
        roles=ROLES,
        clip_lower=100.0,
        clip_upper=100.0,
        resolutions=RESOLUTIONS,
        residual_scales={resolution: 1.0 for resolution in RESOLUTIONS},
        server_windows=(1,),
        server_budgets=budgets or _budget(),
        **extra,
    )


def _defense(manifest):
    return RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": manifest,
                "implementation_phase": 3,
                "parameter_roles": ROLES,
            },
        )
    )


def _run(defense, round_value, global_params, updates, ids=None):
    ids = ids or [f"c{index}" for index in range(len(updates))]
    defense.set_context(
        round_value,
        ids,
        global_params,
        server_optimizer="fedavg",
    )
    return defense.aggregate([(update, 1) for update in updates])


def test_fixed_layout_assigns_head_largest_block_and_overflow():
    defense = _defense(_manifest())
    defense.set_context(1, ["c"], _params(), server_optimizer="fedavg")

    assert defense._layout is not None
    assert defense._layout.indices == {
        "full": (0, 1, 2),
        "head": (1,),
        "block:0": (0,),
        "overflow": (2,),
    }
    with pytest.raises(ValueError, match="must be ordered"):
        ResolutionLayout.build(
            _params(),
            ROLES,
            ("full", "overflow", "block:0"),
        )


def test_head_budget_limits_head_only_residual_outlier():
    budgets = _budget()
    budgets["head"]["1"]["residual"] = 0.1
    defense = _defense(_manifest(budgets=budgets))
    updates = [
        _params(head=(0.0, 0.0)),
        _params(head=(0.0, 0.0)),
        _params(head=(2.0, 0.0)),
    ]
    result = _run(defense, 1, _params(), updates)

    assert defense.last_client_aggregation_weights["c2"] <= 0.05 + 1e-8
    assert defense.last_round_metrics["rtc_v3_head_residual_exposure"] <= 0.1 + 1e-8
    assert float(result[1][0]) <= 0.1 + 1e-7
    assert result[3].tolist() == [3]


def test_overflow_is_one_fixed_resolution_not_per_tensor_capacity():
    budgets = _budget()
    budgets["overflow"]["1"]["residual"] = 0.2
    defense = _defense(_manifest(budgets=budgets))
    updates = [
        _params(tail=(0.0,)),
        _params(tail=(0.0,)),
        _params(tail=(4.0,)),
    ]
    _run(defense, 1, _params(), updates)

    assert (
        defense.last_round_metrics["rtc_v3_overflow_residual_exposure"]
        <= 0.2 + 1e-8
    )


def test_prototypes_are_assigned_from_frozen_previous_round_then_committed():
    defense = _defense(_manifest(cones=True, cone_budget=100.0))
    first_updates = [
        _params(body=(-1.0, 0.0, 0.0, 0.0)),
        _params(body=(0.0, 0.0, 0.0, 0.0)),
        _params(body=(1.0, 0.0, 0.0, 0.0)),
    ]
    first = _run(defense, 1, _params(), first_updates)

    assert set(defense._last_cone_assignments["full"]) == {"overflow"}
    assert len(defense._prototype_banks["full"].prototypes) == 2

    second_updates = [
        _params(body=(float(first[0][0] - 1.0), 0.0, 0.0, 0.0)),
        _params(body=(float(first[0][0]), 0.0, 0.0, 0.0)),
        _params(body=(float(first[0][0] + 1.0), 0.0, 0.0, 0.0)),
    ]
    _run(defense, 2, first, second_updates)
    assignments = defense._last_cone_assignments["full"]

    assert assignments[0].startswith("cone:")
    assert assignments[1] == "overflow"
    assert assignments[2].startswith("cone:")
    assert assignments[0] != assignments[2]


def test_cone_spreading_cannot_exceed_parent_full_residual_budget():
    budgets = _budget()
    budgets["full"]["1"]["residual"] = 0.2
    defense = _defense(
        _manifest(budgets=budgets, cones=True, cone_budget=100.0)
    )
    updates = [
        _params(body=(1.0, 0.0, 0.0, 0.0)),
        _params(body=(0.0, 1.0, 0.0, 0.0)),
        _params(body=(-1.0, 0.0, 0.0, 0.0)),
        _params(body=(0.0, -1.0, 0.0, 0.0)),
    ]
    _run(defense, 1, _params(), updates)

    assert defense.last_round_metrics["rtc_v3_full_residual_exposure"] <= 0.2 + 1e-8


def test_prototype_creation_is_client_arrival_order_independent():
    sketches = [
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([-1.0, 0.0]),
    ]
    masses = [0.2, 0.5, 0.3]
    ids = ["a", "b", "c"]
    first = StablePrototypeBank()
    transition_a = first.prepare(
        server_round=1,
        sketches=sketches,
        nominal_masses=masses,
        client_ids=ids,
    )
    first.commit(transition_a)

    order = [2, 0, 1]
    second = StablePrototypeBank()
    transition_b = second.prepare(
        server_round=1,
        sketches=[sketches[index] for index in order],
        nominal_masses=[masses[index] for index in order],
        client_ids=[ids[index] for index in order],
    )
    second.commit(transition_b)

    assert first.snapshot().keys() == second.snapshot().keys()
    for cone_id, vector in first.snapshot().items():
        assert second.snapshot()[cone_id] == pytest.approx(vector)
