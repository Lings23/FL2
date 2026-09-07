"""Phase 1 gates for RTC-v3's robust anchor and one-round hard budgets."""

from __future__ import annotations

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import build_manifest
from defenses.rtc.solver import (
    LinearBudget,
    SolverResult,
    feasible_scale,
    max_violation,
    solve_subprobability_qp,
)
from defenses.rtc.v3 import RTCv3Defense
from defenses.rtc.weighted_stats import weighted_coordinate_median


def _global() -> list[np.ndarray]:
    return [
        np.zeros(2, dtype=np.float32),
        np.asarray([7], dtype=np.int64),
    ]


def _manifest(
    *,
    clip_lower: float = 10.0,
    clip_upper: float = 10.0,
    anchor: float = 1e6,
    residual: float = 1e6,
    total: float = 1e6,
) -> dict:
    return build_manifest(
        params=_global(),
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        residual_scales={"full": 1.0},
        server_budgets={
            "full": {
                "1": {
                    "anchor": anchor,
                    "residual": residual,
                    "total": total,
                }
            }
        },
    )


def _defense(manifest: dict, **custom) -> RTCv3Defense:
    return RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": manifest,
                "implementation_phase": 1,
                **custom,
            },
        )
    )


def _update(x: float, y: float = 0.0, counter: int = 999):
    return [
        np.asarray([x, y], dtype=np.float32),
        np.asarray([counter], dtype=np.int64),
    ]


def _run(
    defense: RTCv3Defense,
    updates,
    *,
    client_ids=None,
    principal_ids=None,
):
    client_ids = client_ids or [str(index) for index in range(len(updates))]
    defense.set_context(
        1,
        client_ids,
        _global(),
        principal_ids=principal_ids,
        server_optimizer="fedavg",
    )
    return defense.aggregate([(update, 100) for update in updates])


def test_weighted_coordinate_median_is_invariant_to_fixed_mass_split():
    unsplit = weighted_coordinate_median(
        [np.asarray([0.0]), np.asarray([10.0])],
        [0.6, 0.4],
        principal_ids=["p", "q"],
        client_ids=["p0", "q0"],
    )
    split = weighted_coordinate_median(
        [np.asarray([0.0]), np.asarray([0.0]), np.asarray([10.0])],
        [0.3, 0.3, 0.4],
        principal_ids=["p", "p", "q"],
        client_ids=["p0", "p1", "q0"],
    )
    assert unsplit == pytest.approx([0.0])
    assert split == pytest.approx(unsplit)
    assert np.median([0.0, 10.0]) != np.median([0.0, 0.0, 10.0])


def test_w1_hard_constraints_are_subprobability_and_never_renormalized():
    defense = _defense(
        _manifest(clip_lower=100.0, clip_upper=100.0, anchor=0.5, total=0.5)
    )
    result = _run(defense, [_update(1.0), _update(1.0)])

    weight_sum = sum(defense.last_client_aggregation_weights.values())
    assert weight_sum == pytest.approx(0.5, abs=1e-8)
    assert result[0] == pytest.approx([0.5, 0.0], abs=1e-7)
    assert defense.last_round_metrics["rtc_v3_zero_update_mass"] == pytest.approx(0.5)
    assert defense.last_round_metrics["rtc_v3_full_anchor_exposure"] <= 0.5 + 1e-8
    assert defense.last_round_metrics["rtc_v3_max_constraint_violation"] <= 1e-8


def test_anchor_recycle_restores_missing_mass_with_coordinate_median():
    manifest = _manifest(
        clip_lower=100.0,
        clip_upper=100.0,
        anchor=10.0,
        residual=0.5,
        total=10.0,
    )
    baseline = _defense(manifest)
    recycled = _defense(manifest, anchor_recycle_fraction=1.0)

    baseline_result = _run(baseline, [_update(1.0), _update(3.0)])
    recycled_result = _run(recycled, [_update(1.0), _update(3.0)])

    assert baseline.last_round_metrics["rtc_v3_weight_sum"] == pytest.approx(0.75)
    assert baseline_result[0] == pytest.approx([1.25, 0.0], abs=1e-7)
    assert recycled_result[0] == pytest.approx([1.5, 0.0], abs=1e-7)
    assert recycled.last_round_metrics["rtc_v3_anchor_recycle_mass"] == pytest.approx(0.25)
    assert recycled.last_round_metrics["rtc_v3_effective_update_mass"] == pytest.approx(1.0)
    assert recycled.last_round_metrics["rtc_v3_zero_update_mass"] == pytest.approx(0.0)
    assert recycled.last_round_metrics["rtc_v3_full_anchor_exposure"] == pytest.approx(1.0)
    assert recycled.last_round_metrics["rtc_v3_full_total_exposure"] == pytest.approx(1.5)


def test_accepted_weight_anchor_excludes_zero_weight_outlier(monkeypatch):
    monkeypatch.setattr(
        "defenses.rtc.v3.solve_subprobability_qp",
        lambda *args, **kwargs: SolverResult(
            weights=np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0]),
            status="optimized",
            optimized=True,
            max_violation=0.0,
            objective=0.0,
        ),
    )
    manifest = _manifest(
        clip_lower=1000.0,
        clip_upper=1000.0,
        anchor=1e6,
        residual=1e6,
        total=1e6,
    )
    nominal = _defense(manifest, anchor_recycle_fraction=1.0)
    accepted = _defense(
        manifest,
        anchor_recycle_fraction=1.0,
        anchor_recycle_weighting="accepted",
    )
    updates = [_update(0.0), _update(1.0), _update(100.0)]

    nominal_result = _run(nominal, updates)
    accepted_result = _run(accepted, updates)

    assert nominal_result[0] == pytest.approx([2.0 / 3.0, 0.0], abs=1e-7)
    assert accepted_result[0] == pytest.approx([1.0 / 3.0, 0.0], abs=1e-7)
    assert accepted.last_round_metrics["rtc_v3_anchor_recycle_mass"] == pytest.approx(
        1.0 / 3.0
    )
    assert accepted.last_round_metrics["rtc_v3_anchor_recycle_weighting"] == "accepted"
    assert accepted.last_round_metrics[
        "rtc_v3_anchor_recycle_source_available"
    ] == pytest.approx(1.0)


def test_accepted_weight_anchor_recomputes_hard_budget_coefficient(monkeypatch):
    monkeypatch.setattr(
        "defenses.rtc.v3.solve_subprobability_qp",
        lambda *args, **kwargs: SolverResult(
            weights=np.asarray([0.0, 0.0, 1.0 / 3.0]),
            status="optimized",
            optimized=True,
            max_violation=0.0,
            objective=0.0,
        ),
    )
    defense = _defense(
        _manifest(
            clip_lower=1000.0,
            clip_upper=1000.0,
            anchor=10.0,
            residual=1e6,
            total=1e6,
        ),
        anchor_recycle_fraction=1.0,
        anchor_recycle_weighting="accepted",
    )

    _run(defense, [_update(0.0), _update(1.0), _update(100.0)])

    assert defense.last_round_metrics["rtc_v3_anchor_recycle_mass"] == pytest.approx(
        (10.0 - 1.0 / 3.0) / 100.0,
        abs=1e-8,
    )
    assert defense.last_round_metrics["rtc_v3_full_anchor_exposure"] == pytest.approx(
        10.0,
        abs=1e-8,
    )
    assert defense.last_round_metrics["rtc_v3_max_constraint_violation"] <= 1e-8


def test_accepted_weight_anchor_does_not_fallback_when_no_weight_is_accepted(
    monkeypatch,
):
    monkeypatch.setattr(
        "defenses.rtc.v3.solve_subprobability_qp",
        lambda *args, **kwargs: SolverResult(
            weights=np.zeros(2, dtype=np.float64),
            status="optimized",
            optimized=True,
            max_violation=0.0,
            objective=0.0,
        ),
    )
    defense = _defense(
        _manifest(clip_lower=100.0, clip_upper=100.0),
        anchor_recycle_fraction=1.0,
        anchor_recycle_weighting="accepted",
    )

    result = _run(defense, [_update(1.0), _update(3.0)])

    assert result[0] == pytest.approx(_global()[0])
    assert defense.last_round_metrics["rtc_v3_anchor_recycle_target_mass"] == pytest.approx(1.0)
    assert defense.last_round_metrics["rtc_v3_anchor_recycle_mass"] == pytest.approx(0.0)
    assert defense.last_round_metrics["rtc_v3_zero_update_mass"] == pytest.approx(1.0)
    assert defense.last_round_metrics[
        "rtc_v3_anchor_recycle_source_available"
    ] == pytest.approx(0.0)


def test_anchor_recycle_cannot_bypass_zero_hard_budgets():
    defense = _defense(
        _manifest(anchor=0.0, residual=0.0, total=0.0),
        anchor_recycle_fraction=1.0,
    )
    result = _run(defense, [_update(1.0), _update(-1.0)])

    assert result[0] == pytest.approx(_global()[0])
    assert defense.last_round_metrics["rtc_v3_anchor_recycle_mass"] == pytest.approx(0.0)
    assert defense.last_round_metrics["rtc_v3_zero_update_mass"] == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_anchor_recycle_fraction_is_validated(value):
    with pytest.raises(ValueError, match="anchor_recycle_fraction"):
        _defense(_manifest(), anchor_recycle_fraction=value)


def test_anchor_recycle_weighting_is_validated():
    with pytest.raises(ValueError, match="anchor_recycle_weighting"):
        _defense(_manifest(), anchor_recycle_weighting="unknown")


def test_norm_clip_mad_k_defaults_to_2_5_and_can_be_lowered_to_2_25():
    manifest = _manifest(clip_lower=0.0, clip_upper=100.0)
    baseline = _defense(manifest)
    candidate = _defense(manifest, norm_clip_mad_k=2.25)
    updates = [_update(1.0), _update(2.0), _update(10.0)]

    _run(baseline, updates)
    _run(candidate, updates)

    assert baseline.norm_clip_mad_k == pytest.approx(2.5)
    assert candidate.norm_clip_mad_k == pytest.approx(2.25)
    assert baseline.last_round_metrics["rtc_v3_norm_clip_mad_k"] == pytest.approx(2.5)
    assert candidate.last_round_metrics["rtc_v3_norm_clip_mad_k"] == pytest.approx(2.25)
    assert candidate.last_round_metrics["rtc_v3_clip_norm"] < baseline.last_round_metrics[
        "rtc_v3_clip_norm"
    ]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_norm_clip_mad_k_is_validated(value):
    with pytest.raises(ValueError, match="norm_clip_mad_k"):
        _defense(_manifest(), norm_clip_mad_k=value)


def test_residual_rank_cap_routes_only_its_removed_mass_fully_to_accepted_anchor():
    defense = _defense(
        _manifest(clip_lower=100.0, clip_upper=100.0),
        anchor_recycle_fraction=0.51,
        anchor_recycle_weighting="accepted",
        residual_rank_cap_top_k=2,
        residual_rank_cap_factor=0.5,
        residual_rank_recycle_fraction=1.0,
    )

    _run(
        defense,
        [_update(0.0), _update(1.0), _update(2.0), _update(20.0)],
    )

    assert defense.last_client_aggregation_weights["0"] == pytest.approx(0.125)
    assert defense.last_client_aggregation_weights["3"] == pytest.approx(0.125)
    assert defense.last_client_aggregation_weights["1"] == pytest.approx(0.25)
    assert defense.last_client_aggregation_weights["2"] == pytest.approx(0.25)
    metrics = defense.last_round_metrics
    assert metrics["rtc_v3_residual_rank_cap_client_ids"] == "0|3"
    assert metrics["rtc_v3_residual_rank_cap_active_count"] == pytest.approx(2.0)
    assert metrics["rtc_v3_residual_rank_reference_valid"] == pytest.approx(1.0)
    assert metrics["rtc_v3_residual_rank_reference_weight_sum"] == pytest.approx(1.0)
    assert metrics["rtc_v3_residual_rank_removed_mass"] == pytest.approx(0.25)
    assert metrics["rtc_v3_anchor_recycle_base_target_mass"] == pytest.approx(0.0)
    assert metrics["rtc_v3_residual_rank_recycle_target_mass"] == pytest.approx(0.25)
    assert metrics["rtc_v3_anchor_recycle_target_mass"] == pytest.approx(0.25)
    assert metrics["rtc_v3_anchor_recycle_mass"] == pytest.approx(0.25)
    assert metrics["rtc_v3_zero_update_mass"] == pytest.approx(0.0)


def test_residual_rank_cap_ties_are_broken_by_server_identity():
    defense = _defense(
        _manifest(),
        anchor_recycle_weighting="accepted",
        residual_rank_cap_top_k=2,
        residual_rank_cap_factor=0.5,
    )

    _run(
        defense,
        [_update(0.0), _update(0.0), _update(0.0)],
        client_ids=["z", "a", "m"],
        principal_ids=["z", "a", "m"],
    )

    assert defense.last_round_metrics["rtc_v3_residual_rank_cap_client_ids"] == "a|m"


@pytest.mark.parametrize("value", [-1, 1.5, float("nan")])
def test_residual_rank_cap_top_k_is_validated(value):
    with pytest.raises(ValueError, match="residual_rank_cap_top_k"):
        _defense(_manifest(), residual_rank_cap_top_k=value)


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.1, float("nan")])
def test_enabled_residual_rank_cap_factor_is_validated(value):
    with pytest.raises(ValueError, match="residual_rank_cap_factor"):
        _defense(
            _manifest(),
            residual_rank_cap_top_k=2,
            residual_rank_cap_factor=value,
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_residual_rank_recycle_fraction_is_validated(value):
    with pytest.raises(ValueError, match="residual_rank_recycle_fraction"):
        _defense(_manifest(), residual_rank_recycle_fraction=value)


def test_residual_rank_recycle_requires_accepted_anchor():
    with pytest.raises(ValueError, match="anchor_recycle_weighting='accepted'"):
        _defense(
            _manifest(),
            residual_rank_cap_top_k=2,
            residual_rank_cap_factor=0.5,
            residual_rank_recycle_fraction=1.0,
        )


def test_zero_budgets_produce_zero_update_without_fedavg_fallback():
    defense = _defense(_manifest(anchor=0.0, residual=0.0, total=0.0))
    result = _run(defense, [_update(1.0), _update(-1.0)])

    assert result[0] == pytest.approx(_global()[0])
    assert sum(defense.last_client_aggregation_weights.values()) == pytest.approx(0.0)
    assert defense.last_round_metrics["rtc_v3_zero_update_mass"] == pytest.approx(1.0)


def test_nonfloat_buffer_always_preserves_server_value():
    defense = _defense(_manifest())
    result = _run(defense, [_update(1.0, counter=111), _update(2.0, counter=222)])

    assert result[1].dtype == np.int64
    assert result[1].tolist() == [7]


def test_principal_internal_identity_split_preserves_result_and_exposure():
    manifest = _manifest(clip_lower=100.0, clip_upper=100.0, total=1.2)
    unsplit = _defense(manifest)
    result_a = _run(
        unsplit,
        [_update(1.0), _update(3.0)],
        client_ids=["p0", "q0"],
        principal_ids=["p", "q"],
    )

    split = _defense(manifest)
    result_b = _run(
        split,
        [_update(1.0), _update(1.0), _update(3.0)],
        client_ids=["p0", "p1", "q0"],
        principal_ids=["p", "p", "q"],
    )

    assert result_b[0] == pytest.approx(result_a[0], abs=1e-7)
    assert (
        split.last_client_aggregation_weights["p0"]
        + split.last_client_aggregation_weights["p1"]
    ) == pytest.approx(unsplit.last_client_aggregation_weights["p0"], abs=1e-7)
    for exposure_type in ("anchor", "residual", "total"):
        assert split.last_round_metrics[
            f"rtc_v3_full_{exposure_type}_exposure"
        ] == pytest.approx(
            unsplit.last_round_metrics[
                f"rtc_v3_full_{exposure_type}_exposure"
            ],
            abs=1e-7,
        )


def test_first_round_model_replacement_is_clipped_and_hard_limited():
    defense = _defense(
        _manifest(
            clip_lower=2.0,
            clip_upper=2.0,
            anchor=1.0,
            residual=0.1,
            total=1.0,
        )
    )
    result = _run(
        defense,
        [_update(1.0), _update(1.0), _update(100.0)],
        client_ids=["good-a", "good-b", "replacement"],
    )

    assert defense._last_clipped_mask == [False, False, True]
    assert defense._last_records[2].norm == pytest.approx(100.0)
    assert defense.last_round_metrics["rtc_v3_full_residual_exposure"] <= 0.1 + 1e-8
    assert defense.last_round_metrics["rtc_v3_full_total_exposure"] <= 1.0 + 1e-8
    assert float(result[0][0]) < 1.1


def test_solver_failure_uses_verified_feasible_scale(monkeypatch):
    import defenses.rtc.solver as solver_module

    monkeypatch.setattr(
        solver_module,
        "minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("solver down")),
    )
    result = solve_subprobability_qp(
        [0.5, 0.5],
        [0.5, 0.5],
        [LinearBudget("limit", np.asarray([1.0, 1.0]), 0.25)],
    )
    assert result.status == "feasible_scale"
    assert result.weights.sum() == pytest.approx(0.25)
    assert result.max_violation <= 1e-8


def test_solver_and_warm_start_failure_returns_zero(monkeypatch):
    import defenses.rtc.solver as solver_module

    monkeypatch.setattr(
        solver_module,
        "minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("solver down")),
    )
    monkeypatch.setattr(
        solver_module,
        "feasible_scale",
        lambda *args, **kwargs: np.asarray([np.nan, np.nan]),
    )
    result = solve_subprobability_qp(
        [0.5, 0.5],
        [0.5, 0.5],
        [LinearBudget("limit", np.asarray([1.0, 1.0]), 0.25)],
    )
    assert result.status == "zero_fallback"
    assert result.weights == pytest.approx([0.0, 0.0])


def test_random_qp_instances_never_exceed_caps_mass_or_budgets():
    rng = np.random.default_rng(20260729)
    for _ in range(50):
        nominal = rng.dirichlet(np.ones(8))
        caps = nominal * rng.uniform(0.2, 1.0, size=8)
        coefficients = rng.uniform(0.0, 3.0, size=(5, 8))
        scale = rng.uniform(0.0, 1.0)
        reference = scale * caps
        budgets = [
            LinearBudget(f"b{index}", row, float(np.dot(row, reference)))
            for index, row in enumerate(coefficients)
        ]
        result = solve_subprobability_qp(nominal, caps, budgets)

        assert max_violation(result.weights, caps, budgets) <= 1e-8
        assert result.weights.sum() <= 1.0 + 1e-8
        assert np.all(result.weights >= -1e-8)
        assert np.all(result.weights <= caps + 1e-8)
        assert max_violation(feasible_scale(caps, budgets), caps, budgets) <= 1e-8


def test_independent_validation_turns_invalid_solver_output_into_zero(monkeypatch):
    import defenses.rtc.v3 as v3_module

    monkeypatch.setattr(
        v3_module,
        "solve_subprobability_qp",
        lambda *args, **kwargs: SolverResult(
            weights=np.asarray([np.nan, np.nan]),
            status="optimized",
            optimized=True,
            max_violation=0.0,
            objective=0.0,
        ),
    )
    defense = _defense(_manifest())
    result = _run(defense, [_update(1.0), _update(-1.0)])

    assert result[0] == pytest.approx(_global()[0])
    assert defense.last_round_metrics["rtc_v3_weight_sum"] == pytest.approx(0.0)
    snapshot = defense._server_ledger.snapshot()
    assert snapshot["full"]["residual"][1] == pytest.approx(0.0)


def test_server_supplied_mass_still_cannot_exceed_principal_cap():
    defense = _defense(_manifest())
    defense.set_context(
        1,
        ["a", "b"],
        _global(),
        principal_ids=["p", "q"],
        validated_masses=[100.0, 1.0],
        server_optimizer="fedavg",
    )
    defense.aggregate([(_update(1.0), 1), (_update(0.0), 1)])

    assert defense.last_client_weights == pytest.approx({"a": 0.5, "b": 0.5})
