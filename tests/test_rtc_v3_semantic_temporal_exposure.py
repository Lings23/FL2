from __future__ import annotations

import numpy as np
import pytest
import time
from copy import deepcopy

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import (
    CalibrationManifest,
    SCHEMA_VERSION_V3,
    build_manifest,
)
from defenses.rtc.semantic import (
    ClassifierHeadLayout,
    extract_semantic_batch,
    gate_semantic_interventions,
    hard_exposure_coefficients,
    ordered_class_pairs,
    select_top_pair_indices,
)
from defenses.rtc.semantic_temporal import SemanticTemporalEvidence
from defenses.rtc.v3 import RTCv3Defense
from experiments.rtc_v3_formal_calibration import (
    _apply_parent_budget_floor,
    _collection_observation_contract,
)
from defenses.rtc.calibration import content_hash


ROLES = {"1": "classifier_weight", "2": "classifier_bias"}
NAMES = {"0": "body.weight", "1": "fc.weight", "2": "fc.bias"}


def _params():
    return [
        np.zeros(4, dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    ]


def _semantic_config():
    pair_count = 3 * 2
    return {
        "enabled": True,
        "num_classes": 3,
        "phases": {
            "steady": {
                "row_scales": [1.0, 1.0, 1.0],
                "pair_centers": [0.0] * pair_count,
                "pair_scales": [0.1] * pair_count,
                "head_exposure_scale": 1.0,
            }
        },
        "temporal": {
            "decay_rate": 0.1,
            "kappa": 0.0,
            "threshold": 0.0,
            "eta": 1.0,
            "q_min": 0.2,
            "recovery_threshold": 0.1,
            "recovery_observations": 2,
            "recovery_factor": 0.5,
            "synchronization": {
                "enabled": True,
                "min_distinct_principals": 2,
                "evidence_threshold": 0.5,
                "similarity_threshold": 0.9,
                "multiplier": 1.5,
            },
        },
        "risk_lambda": 0.5,
        "state_thresholds": {"watch": 0.10, "restricted": 0.50, "quarantined": 0.80},
        "hard_exposure_risk_floor": 0.10,
        "top_k_pairs": 2,
        "global_windows": [1, 4, 8],
        "global_budgets": {"1": 1e6, "4": 1e6, "8": 1e6},
        "principal_windows": [1, 4, 8],
        "principal_budgets": {"1": 1e6, "4": 1e6, "8": 1e6},
        "pair_windows": [1, 4, 8],
        "pair_budgets": {"1": 1e6, "4": 1e6, "8": 1e6},
        "performance_budget": {
            "semantic_mean_seconds": 0.15,
            "semantic_p95_seconds": 0.25,
            "defense_mean_increase_fraction": 0.05,
            "defense_p95_increase_fraction": 0.10,
        },
    }


def _manifest(*, frozen_cones: bool = False):
    params = _params()
    layout = ClassifierHeadLayout.build(params, ROLES)
    stable_cones = {"enabled": False, "online_updates_enabled": False}
    extra = {}
    if frozen_cones:
        stable_cones = {
            "enabled": True,
            "online_updates_enabled": False,
            "mode": "offline_controlled_v1",
            "dimension": 8,
            "seed": 7,
            "sketch_algorithm_version": "splitmix64_v1",
            "max_prototypes": 2,
            "momentum": 0.98,
            "min_update_principals": 2,
            "phases": {
                "steady": {
                    "match_threshold": -1.0,
                    "update_threshold": -1.0,
                    "max_angular_drift": 0.2,
                    "prototypes": [
                        {
                            "cone_id": "cone-0",
                            "vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        }
                    ],
                }
            },
        }
        extra["cone_budgets"] = {
            "full": {"1": {"residual": {"steady": 1e9}}}
        }
    return build_manifest(
        params=params,
        roles=ROLES,
        schema_version=SCHEMA_VERSION_V3,
        clip_lower=100.0,
        clip_upper=100.0,
        training_phases=[{"start_round": 1, "profile": "steady"}],
        stable_cones=stable_cones,
        classifier_head_contract=layout.contract(params, NAMES),
        semantic_temporal_exposure=_semantic_config(),
        **extra,
    )


def _defense(
    *,
    frozen_cones: bool = False,
    semantic_ablation: str = "exposure",
    semantic_intervention_risk_floor: float = 0.0,
):
    cfg = DefenseConfig(
        enabled=True,
        type="rtc_v3_candidate",
        custom_params={
            "calibration_manifest": _manifest(frozen_cones=frozen_cones),
            "implementation_phase": 3 if frozen_cones else 1,
            "parameter_roles": ROLES,
            "semantic_ablation": semantic_ablation,
            "semantic_intervention_risk_floor": semantic_intervention_risk_floor,
        },
    )
    return RTCv3Defense(cfg, num_clients=3)


def test_clean_collection_observation_contract_ignores_reporting_only_metadata():
    payload = deepcopy(_manifest())
    semantic = payload["semantic_temporal_exposure"]
    semantic["risk_lambda"] = 0.0
    semantic["temporal"]["q_min"] = 0.999999
    semantic["temporal"]["synchronization"]["enabled"] = False
    first = content_hash(_collection_observation_contract(payload))

    payload["classifier_head_contract"].pop("semantic_resolutions", None)
    payload["classifier_head_contract"]["contract_hash"] = "legacy"
    semantic["top_k_pairs"] = 1
    semantic.pop("state_thresholds", None)
    semantic.pop("performance_budget", None)
    semantic["temporal"].pop("min_history_observations", None)
    second = content_hash(_collection_observation_contract(payload))
    assert first == second

    semantic["risk_lambda"] = 0.1
    with pytest.raises(ValueError, match="neutral semantic risk"):
        _collection_observation_contract(payload)


def test_parent_budget_floor_prevents_recursive_feasible_set_shrinkage():
    candidate = {"full": {"1": {"anchor": {"warmup": 0.3}, "residual": {"warmup": 2.0}}}}
    parent = {"full": {"1": {"anchor": {"warmup": 4.2}, "residual": {"warmup": 1.9}}}}
    result, raised = _apply_parent_budget_floor(candidate, parent)
    assert result["full"]["1"]["anchor"]["warmup"] == pytest.approx(4.2)
    assert result["full"]["1"]["residual"]["warmup"] == pytest.approx(2.0)
    assert raised == ["full/1/anchor/warmup"]


def test_ordered_pairs_are_complete_and_stable():
    assert ordered_class_pairs(3) == (
        (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    )


def test_hard_exposure_floor_ignores_sub_watch_jitter_only():
    risks = np.asarray([0.0, 0.0025, 0.10, 0.90])
    heads = np.asarray([1.0, 2.0, 3.0, 4.0])
    coefficients = hard_exposure_coefficients(risks, heads, 0.10)
    np.testing.assert_allclose(coefficients, [0.0, 0.0, 0.0, 3.2])
    # Continuous soft controls are intentionally not modified by the floor.
    np.testing.assert_allclose(np.maximum(0.3, 1.0 - risks), [1.0, 0.9975, 0.9, 0.3])


def test_semantic_intervention_gate_keeps_observation_but_removes_sub_watch_controls():
    risks = np.asarray([0.0, 0.059, 0.10, 0.93])
    q_values = np.asarray([1.0, 0.941, 0.90, 0.30])

    optimization_risk, optimization_q, active = gate_semantic_interventions(
        risks, q_values, 0.10
    )

    np.testing.assert_allclose(optimization_risk, [0.0, 0.0, 0.0, 0.93])
    np.testing.assert_allclose(optimization_q, [1.0, 1.0, 1.0, 0.30])
    np.testing.assert_array_equal(active, [False, False, False, True])


def test_zero_semantic_intervention_floor_is_backward_compatible():
    risks = np.asarray([0.0, 0.059, 0.93])
    q_values = np.asarray([1.0, 0.941, 0.30])

    optimization_risk, optimization_q, _ = gate_semantic_interventions(
        risks, q_values, 0.0
    )

    np.testing.assert_allclose(optimization_risk, risks)
    np.testing.assert_allclose(optimization_q, q_values)


def test_intervention_floor_also_controls_hard_exposure_coefficients():
    params = _params()
    updates = []
    for client in range(3):
        local = [value.copy() for value in params]
        if client == 2:
            local[1][1, 0] = 1.0
        updates.append((local, 1))

    influence = {}
    for floor in (0.10, 0.50):
        defense = _defense(semantic_intervention_risk_floor=floor)
        defense.set_context(
            1,
            ["a", "b", "c"],
            params,
            parameter_names=NAMES,
            server_optimizer="fedavg",
        )
        defense.aggregate(updates)
        influence[floor] = defense._last_records[-1].influence_risk
        assert defense.last_round_metrics[
            "rtc_v3_semantic_hard_exposure_risk_floor"
        ] == pytest.approx(floor)

    assert influence[0.50] < influence[0.10]


def test_semantic_features_are_vectorized_for_unknown_source_target():
    params = _params()
    layout = ClassifierHeadLayout.build(params, ROLES)
    residuals = [
        [
            np.zeros(4),
            np.asarray([[0.0, 0.0], [2.0, 0.0], [0.5, 0.0]]),
            np.zeros(3),
        ]
    ]
    batch = extract_semantic_batch(
        layout=layout,
        residual_arrays=residuals,
        floating_indices=(0, 1, 2),
        phase_config=_semantic_config()["phases"]["steady"],
    )
    assert batch.z_values.shape == (1, 6)
    assert batch.pairs[int(batch.top_pair_indices[0])] == (0, 1)
    assert batch.z_values[0, batch.pairs.index((0, 1))] == pytest.approx(20.0)
    scalar = []
    normalized_rows = np.asarray([0.0, 2.0, 0.5])
    for source, target in batch.pairs:
        scalar.append(max(0.0, (normalized_rows[target] - normalized_rows[source]) / 0.1))
    np.testing.assert_allclose(batch.z_values[0], scalar)


def test_top_k_and_other_partition_are_order_independent_with_stable_ties():
    pairs = ordered_class_pairs(3)
    selected = select_top_pair_indices(
        client_pair_indices=[3, 0, 3, 0],
        exposure_coefficients=[1.0, 1.0, 1.0, 1.0],
        nominal_masses=[0.25] * 4,
        pairs=pairs,
        top_k=2,
    )
    permuted = select_top_pair_indices(
        client_pair_indices=[0, 3, 0, 3],
        exposure_coefficients=[1.0] * 4,
        nominal_masses=[0.25] * 4,
        pairs=pairs,
        top_k=2,
    )
    assert selected == permuted == (0, 3)


def test_missing_rounds_decay_lazily_and_never_count_as_recovery():
    config = _semantic_config()["temporal"]
    evidence = SemanticTemporalEvidence(config, pair_count=2)
    first = evidence.prepare(
        server_round=1,
        principal_ids=["p"],
        z_values=np.asarray([[2.0, 0.0]]),
        top_pair_indices=[0],
        top_pair_signatures=[np.asarray([1.0, 0.0])],
    )
    assert evidence.state_dict() == {}
    evidence.commit(first)
    before = evidence.state_dict()["p\u001f0"]
    assert before["low_evidence_streak"] == 0
    fourth = evidence.prepare(
        server_round=4,
        principal_ids=["p"],
        z_values=np.asarray([[0.0, 0.0]]),
        top_pair_indices=[0],
        top_pair_signatures=[np.asarray([1.0, 0.0])],
    )
    state = fourth.next_states[("p", 0)]
    assert state.statistic == pytest.approx(2.0 * np.exp(-0.3))
    assert state.low_evidence_streak == 1
    assert state.observation_count == 2


def test_cross_principal_sync_requires_distinct_principals_and_similarity():
    evidence = SemanticTemporalEvidence(_semantic_config()["temporal"], pair_count=2)
    same_principal = evidence.prepare(
        server_round=1,
        principal_ids=["p", "p"],
        z_values=np.asarray([[2.0, 0.0], [2.0, 0.0]]),
        top_pair_indices=[0, 0],
        top_pair_signatures=[np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])],
    )
    assert same_principal.synchronized_pairs == ()
    distinct = evidence.prepare(
        server_round=1,
        principal_ids=["p", "q"],
        z_values=np.asarray([[2.0, 0.0], [2.0, 0.0]]),
        top_pair_indices=[0, 0],
        top_pair_signatures=[np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])],
    )
    assert distinct.synchronized_pairs == (0,)
    assert distinct.next_states[("p", 0)].statistic == pytest.approx(3.0)
    first_order = evidence.prepare(
        server_round=1,
        principal_ids=["p", "p", "q"],
        client_ids=["endpoint-a", "endpoint-b", "endpoint-q"],
        z_values=np.asarray([[1.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        top_pair_indices=[0, 0, 0],
        top_pair_signatures=[
            np.asarray([-1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
        ],
    )
    second_order = evidence.prepare(
        server_round=1,
        principal_ids=["q", "p", "p"],
        client_ids=["endpoint-q", "endpoint-b", "endpoint-a"],
        z_values=np.asarray([[2.0, 0.0], [2.0, 0.0], [1.0, 0.0]]),
        top_pair_indices=[0, 0, 0],
        top_pair_signatures=[
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
        ],
    )
    assert first_order.synchronized_pairs == second_order.synchronized_pairs == (0,)
    assert first_order.next_states == second_order.next_states


def test_reordering_clients_preserves_principal_state_and_controlled_recovery():
    config = dict(_semantic_config()["temporal"])
    first = SemanticTemporalEvidence(config, pair_count=2)
    second = SemanticTemporalEvidence(config, pair_count=2)
    a = first.prepare(
        server_round=1,
        principal_ids=["p", "q"],
        z_values=np.asarray([[2.0, 0.0], [1.0, 0.0]]),
        top_pair_indices=[0, 0],
        top_pair_signatures=[np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])],
    )
    b = second.prepare(
        server_round=1,
        principal_ids=["q", "p"],
        z_values=np.asarray([[1.0, 0.0], [2.0, 0.0]]),
        top_pair_indices=[0, 0],
        top_pair_signatures=[np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])],
    )
    assert a.next_states == b.next_states
    first.commit(a)
    low1 = first.prepare(
        server_round=2,
        principal_ids=["p"],
        z_values=np.zeros((1, 2)),
        top_pair_indices=[0],
        top_pair_signatures=[np.asarray([1.0, 0.0])],
    )
    first.commit(low1)
    low2 = first.prepare(
        server_round=3,
        principal_ids=["p"],
        z_values=np.zeros((1, 2)),
        top_pair_indices=[0],
        top_pair_signatures=[np.asarray([1.0, 0.0])],
    )
    assert low2.next_states[("p", 0)].statistic < low1.next_states[("p", 0)].statistic


def test_v3_manifest_tamper_and_head_mismatch_fail_fast():
    payload = _manifest()
    CalibrationManifest.load(payload)
    tampered = dict(payload)
    tampered["classifier_head_contract"] = dict(payload["classifier_head_contract"])
    tampered["classifier_head_contract"]["num_classes"] = 4
    with pytest.raises(ValueError, match="content hash|contract hash"):
        CalibrationManifest.load(tampered)

    defense = _defense()
    params = _params()
    with pytest.raises(ValueError, match="model_metadata_hash"):
        defense.set_context(
            1,
            ["a", "b", "c"],
            params,
            parameter_roles={"0": "classifier_weight"},
            parameter_names=NAMES,
            server_optimizer="fedavg",
        )


def test_semantic_risk_caps_client_and_checkpoint_round_trips():
    defense = _defense()
    params = _params()
    updates = []
    for client in range(3):
        local = [value.copy() for value in params]
        if client == 2:
            local[1][1, 0] = 2.0
        updates.append((local, 1))
    defense.set_context(
        1,
        ["a", "b", "c"],
        params,
        parameter_names=NAMES,
        server_optimizer="fedavg",
    )
    defense.aggregate(updates)
    assert defense.last_round_metrics["rtc_v3_semantic_risk_max"] > 0.0
    assert defense.last_client_aggregation_weights["c"] < 1.0 / 3.0
    assert defense.last_round_metrics["rtc_v3_max_constraint_violation"] <= 1e-8
    semantic_ledger = defense.state_dict()["server_ledger"]["semantic"]
    assert "risk" in semantic_ledger
    assert "semantic:other" in semantic_ledger
    assert any(key.startswith("semantic:pair:") for key in semantic_ledger)
    principal_ledger = defense.state_dict()["principal_ledger"]
    assert all(
        "semantic" in events[-1].exposures for events in principal_ledger.values()
    )

    state = defense.state_dict()
    restored = _defense()
    restored.load_state_dict(state)
    assert restored.state_dict()["semantic_temporal"] == state["semantic_temporal"]


def test_semantic_vectorization_meets_server_cost_budget():
    rng = np.random.default_rng(42)
    params = [
        np.zeros(8, dtype=np.float32),
        np.zeros((10, 512), dtype=np.float32),
        np.zeros(10, dtype=np.float32),
    ]
    roles = {"1": "classifier_weight", "2": "classifier_bias"}
    layout = ClassifierHeadLayout.build(params, roles)
    residuals = [
        [
            rng.normal(size=8),
            rng.normal(size=(10, 512)),
            rng.normal(size=10),
        ]
        for _ in range(10)
    ]
    phase = {
        "row_scales": [1.0] * 10,
        "pair_centers": [0.0] * 90,
        "pair_scales": [1.0] * 90,
        "head_exposure_scale": 1.0,
    }
    for _ in range(5):
        extract_semantic_batch(
            layout=layout,
            residual_arrays=residuals,
            floating_indices=(0, 1, 2),
            phase_config=phase,
        )
    samples = []
    for _ in range(40):
        started = time.perf_counter()
        extract_semantic_batch(
            layout=layout,
            residual_arrays=residuals,
            floating_indices=(0, 1, 2),
            phase_config=phase,
        )
        samples.append(time.perf_counter() - started)
    assert float(np.mean(samples)) <= 0.15
    assert float(np.quantile(samples, 0.95)) <= 0.25


def test_schema_v3_frozen_cone_vectors_never_update_from_semantic_samples():
    defense = _defense(frozen_cones=True)
    params = _params()
    updates = []
    for client in range(3):
        local = [value.copy() for value in params]
        local[0][client] = 0.25 + client
        local[1][client, 0] = 1.0 + client
        updates.append((local, 1))
    defense.set_context(
        1,
        ["a", "b", "c"],
        params,
        parameter_names=NAMES,
        server_optimizer="fedavg",
    )
    before = [value.vector.copy() for value in defense._prototype_banks["full"].prototypes]
    defense.aggregate(updates)
    after = [value.vector.copy() for value in defense._prototype_banks["full"].prototypes]
    assert len(before) == len(after) == 1
    np.testing.assert_array_equal(before[0], after[0])
    assert defense.last_round_metrics["rtc_v3_cone_update_applied_count"] == 0.0


def test_semantic_ablation_layers_enable_cap_only_from_temporal_stage():
    params = _params()
    updates = []
    for client in range(3):
        local = [value.copy() for value in params]
        if client == 2:
            local[1][1, 0] = 2.0
        updates.append((local, 1))
    observed_weights = {}
    for mode in ("observe", "soft", "temporal", "exposure"):
        defense = _defense(semantic_ablation=mode)
        defense.set_context(
            1,
            ["a", "b", "c"],
            params,
            parameter_names=NAMES,
            server_optimizer="fedavg",
        )
        defense.aggregate(updates)
        observed_weights[mode] = defense.last_client_aggregation_weights["c"]
    assert observed_weights["observe"] == pytest.approx(1.0 / 3.0)
    assert observed_weights["soft"] <= observed_weights["observe"]
    assert observed_weights["temporal"] <= (1.0 / 3.0) * 0.2 + 1e-8
    assert observed_weights["exposure"] <= (1.0 / 3.0) * 0.2 + 1e-8
