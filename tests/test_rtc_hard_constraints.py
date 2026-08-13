"""Focused invariants for RTC LOO influence and hard aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from defenses.rtc import ClientHistory, InfluenceAggregator, RiskScorer, RoundRecord


def _record(client_id: str, delta: float, raw_weight: float = 1.0) -> RoundRecord:
    flat = np.asarray([delta], dtype=np.float32)
    return RoundRecord(
        client_id=client_id,
        params_index=0,
        num_examples=int(raw_weight),
        delta_flat=flat,
        signature=flat.copy(),
        norm=abs(float(delta)),
        raw_weight=float(raw_weight),
    )


def _histories(records):
    return {
        record.client_id: ClientHistory(record.client_id, max_history=30)
        for record in records
    }


def test_loo_attempt_matches_remove_one_aggregate_displacement():
    records = [_record("a", 0.0), _record("b", 0.0), _record("attack", 10.0)]
    scorer = RiskScorer({"enable_temporal_features": False})

    scorer.score_records(records, _histories(records), server_round=1)

    nominal = (0.0 + 0.0 + 10.0) / 3.0
    without_attack = 0.0
    assert records[2].influence_attempt == pytest.approx(abs(nominal - without_attack))
    assert records[2].influence_attempt > records[0].influence_attempt
    assert records[2].influence_risk > records[0].influence_risk


def test_large_sample_consensus_update_has_zero_loo_influence():
    records = [_record("large", 1.0, 1000.0), _record("small-a", 1.0), _record("small-b", 1.0)]
    scorer = RiskScorer({"enable_temporal_features": False})

    scorer.score_records(records, _histories(records), server_round=1)

    assert [record.influence_attempt for record in records] == pytest.approx([0.0, 0.0, 0.0])
    assert [record.influence_risk for record in records] == pytest.approx([0.0, 0.0, 0.0])


def test_model_replacement_attempt_grows_and_first_event_is_restricted():
    scorer = RiskScorer({
        "enable_temporal_features": False,
        "robust_z_threshold": 10.0,
    })
    attempts = []
    influence_risks = []
    magnitude_risks = []
    final_record = None
    final_history = None
    for boost in (2.0, 5.0, 10.0, 50.0):
        records = [_record(f"good-{idx}", 0.1) for idx in range(4)] + [_record("attack", 0.1 * boost)]
        histories = _histories(records)
        histories["attack"].impact_history.extend([0.02] * 5)
        histories["attack"].norm_history.extend([0.1] * 5)
        scorer.score_records(records, histories, server_round=1)
        attack = records[-1]
        attempts.append(attack.influence_attempt)
        influence_risks.append(attack.influence_risk)
        magnitude_risks.append(attack.magnitude_risk)
        final_record = attack
        final_history = histories["attack"]

    assert attempts == sorted(attempts)
    assert influence_risks == sorted(influence_risks)
    assert magnitude_risks == sorted(magnitude_risks)
    assert final_record is not None and final_history is not None
    scorer.update_state_and_trust(final_record, final_history, {}, server_round=1)
    assert final_record.state in {"restricted", "quarantined"}
    assert final_record.clipped is False  # clipping is an aggregation decision


def test_peak_enhanced_total_risk_and_state_use_only_total():
    scorer = RiskScorer({})
    record = _record("client", 1.0)
    record.magnitude_risk = 1.0
    record.direction_risk = 0.0
    record.temporal_risk = 0.0
    record.influence_risk = 0.0

    record.total_risk = scorer._combine(record)
    assert record.total_risk == pytest.approx(0.75 + 0.25 * 0.30)

    # Component fields are deliberately absent: state transition reads total only.
    state_record = type("StateRecord", (), {
        "total_risk": 0.96,
        "state": "normal",
        "trust": 1.0,
        "quarantined": False,
    })()
    history = ClientHistory("client", max_history=30)
    scorer.update_state_and_trust(state_record, history, {}, server_round=1)
    assert state_record.state == "quarantined"


def test_hard_caps_preserve_zero_mass_and_server_integer_buffers():
    records = [_record("dominant", 1.0, 100.0), _record("a", 1.0), _record("b", 1.0)]
    aggregator = InfluenceAggregator({"enable_exposure_budgets": False})
    aggregator.assign_weights(records)
    assigned_sum = sum(record.effective_weight for record in records)

    assert records[0].effective_weight == pytest.approx(2.0 / 3.0)
    assert assigned_sum < 1.0

    updates = [([np.asarray([1.0], dtype=np.float32), np.asarray([99], dtype=np.int64)], int(r.raw_weight)) for r in records]
    global_params = [np.asarray([0.0], dtype=np.float32), np.asarray([7], dtype=np.int64)]
    result = aggregator.aggregate(updates, records, global_params, server_round=1)

    assert sum(record.aggregation_weight for record in records) == pytest.approx(assigned_sum)
    assert result[0][0] == pytest.approx(assigned_sum)
    assert result[1][0] == 7
    assert aggregator._last_zero_mass == pytest.approx(1.0 - assigned_sum)
    assert aggregator._last_max_cap_violation <= 1e-8


def test_client_window_budget_rejects_excess_without_renormalizing():
    aggregator = InfluenceAggregator({
        "client_exposure_budget_multiplier": 1.0,
        "direction_exposure_budget_multiplier": 100.0,
        "exposure_window": 4,
    })
    global_params = [np.asarray([0.0], dtype=np.float32)]
    updates = [([np.asarray([1.0], dtype=np.float32)], 1), ([np.asarray([-1.0], dtype=np.float32)], 1)]

    first = [_record("a", 1.0), _record("b", -1.0)]
    aggregator.assign_weights(first)
    result_one = aggregator.aggregate(updates, first, global_params, server_round=1)
    assert aggregator._last_weight_sum == pytest.approx(1.0)
    assert result_one[0][0] == pytest.approx(0.0)

    second = [_record("a", 1.0), _record("b", -1.0)]
    aggregator.assign_weights(second)
    result_two = aggregator.aggregate(updates, second, global_params, server_round=2)

    np.testing.assert_array_equal(result_two[0], global_params[0])
    assert aggregator._last_fallback_reason == "insufficient_effective_clients"
    assert aggregator._last_weight_sum == 0.0
    assert aggregator._last_zero_mass == 1.0
    assert all(record.aggregation_weight == 0.0 for record in second)


def test_direction_group_budget_caps_repeated_residual_direction():
    aggregator = InfluenceAggregator({
        "client_exposure_budget_multiplier": 100.0,
        "direction_exposure_budget_multiplier": 1.0,
        "exposure_window": 4,
        "min_effective_clients": 1,
    })
    global_params = [np.asarray([0.0], dtype=np.float32)]
    updates = [
        ([np.asarray([-2.0], dtype=np.float32)], 1),
        ([np.asarray([2.0], dtype=np.float32)], 1),
        ([np.asarray([2.0], dtype=np.float32)], 1),
    ]

    first = [_record("residual", -2.0), _record("a", 2.0), _record("b", 2.0)]
    aggregator.assign_weights(first)
    aggregator.aggregate(updates, first, global_params, server_round=1)
    assert first[0].direction_group != "consensus"
    assert first[0].aggregation_weight == pytest.approx(1.0 / 3.0)

    second = [_record("residual", -2.0), _record("a", 2.0), _record("b", 2.0)]
    aggregator.assign_weights(second)
    aggregator.aggregate(updates, second, global_params, server_round=2)

    assert second[0].aggregation_weight == pytest.approx(0.0)
    assert sum(record.aggregation_weight for record in second) == pytest.approx(2.0 / 3.0)
    assert aggregator._last_zero_mass == pytest.approx(1.0 / 3.0)
    assert aggregator._last_max_budget_violation <= 1e-8
