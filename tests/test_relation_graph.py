"""Fixed upload and graph regressions for the relation graph defense."""

import json
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from config.config_loader import DefenseConfig, StrategyConfig
from defenses.defense_base import FedAvgDefense, get_defense
from defenses.relation_graph.graph import EdgeObservation, RoundObservation, assess
from defenses.relation_graph.settings import Settings


IDS = tuple(str(index) for index in range(8))
BASE = [np.zeros(2, dtype=np.float32), np.zeros(9, dtype=np.float32),
        np.zeros(1, dtype=np.float32), np.array(0, dtype=np.int64)]
NAMES = {"0": "body.weight", "1": "classifier.weight", "2": "running_mean", "3": "counter"}
ROLES = {"1": "classifier_weight"}


def make_defense(**settings):
    return get_defense(DefenseConfig(enabled=True, type="relation_graph", custom_params={"settings": settings}),
                       num_clients=len(IDS))


def uploads(ids=IDS, *, coordinated=(), natural_groups=False):
    results = []
    for identity in ids:
        index = int(identity)
        backbone = np.eye(2, dtype=np.float32)[index // 4 if natural_groups else 0]
        task_index = index // 4 if natural_groups else 8 if identity in coordinated else index
        task = np.eye(9, dtype=np.float32)[task_index]
        results.append(([backbone, task, np.array([1000 * index], dtype=np.float32),
                         np.array(index, dtype=np.int64)], 3 + index))
    return results


def set_context(defense, number, ids=IDS):
    defense.set_context(number, ids, BASE, principal_ids=ids, server_optimizer="fedavg",
                        parameter_roles=ROLES, parameter_names=NAMES, trainable_parameter_indices={0, 1})


def step(defense, number, ids=IDS, **fixture):
    set_context(defense, number, ids)
    result = defense.aggregate(uploads(ids, **fixture))
    return result, defense.diagnostics()


def assert_same_model(left, right):
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)


def test_missing_reference_and_single_upload_preserve_fedavg():
    defense = make_defense()
    for number, ids in enumerate((IDS, ("2",)), start=1):
        result, diag = step(defense, number, ids)
        assert_same_model(result, FedAvgDefense(DefenseConfig()).aggregate(uploads(ids)))
        assert diag["metrics"]["relation_graph_state"] == "insufficient_evidence"
        assert diag["metrics"]["relation_graph_anchor_weight"] == 0
    assert defense.state_dict()["history"][-1]["edges"] == ()


def test_normal_non_iid_groups_keep_their_sample_weights():
    defense = make_defense()
    for number in range(1, 5):
        result, diag = step(defense, number, natural_groups=True)
    assert diag["metrics"]["relation_graph_state"] == "no_anomaly_evidence"
    assert diag["metrics"]["relation_graph_calibrated_fraction"] == 1
    assert {edge["reference_center"] for edge in diag["edges"]} == {0, 1}
    assert not diag["groups"]
    assert_same_model(result, FedAvgDefense(DefenseConfig()).aggregate(uploads(natural_groups=True)))


@pytest.mark.parametrize("attack_start", [1, 4])
def test_persistent_local_group_loses_weight_without_amplifying_other_clients(attack_start):
    defense = make_defense()
    group = ("5", "6", "7")
    for number in (1, 2, 3):
        step(defense, number, coordinated=group if number >= attack_start else ())
    _, first = step(defense, 4, coordinated=group)
    assert first["metrics"]["relation_graph_state"] == "insufficient_evidence"
    edge = next(edge for edge in first["edges"] if (edge["left"], edge["right"]) == ("5", "6"))
    expected_shift = 0.15 if attack_start == 1 else 0.0
    assert edge["reference_center"] == pytest.approx(expected_shift)
    assert edge["residual"] == pytest.approx(1.0 - expected_shift)
    result, second = step(defense, 5, coordinated=group)
    assert second["metrics"]["relation_graph_state"] == "anomalous_structure"
    assert [row["principals"] for row in second["groups"]] == [list(group)]
    for row in second["clients"]:
        if row["principal_id"] in group:
            assert 0 < row["aggregation_weight"] < row["nominal_weight"]
        else:
            assert row["aggregation_weight"] == row["nominal_weight"]
    weights = [row["aggregation_weight"] for row in second["clients"]]
    anchor = second["metrics"]["relation_graph_anchor_weight"]
    assert sum(weights) + anchor == pytest.approx(1)
    uploaded = uploads(coordinated=group)
    for index in (0, 1, 2):
        expected = sum(weight * values[index] for weight, (values, _) in zip(weights, uploaded)) + anchor * BASE[index]
        np.testing.assert_allclose(result[index], expected, rtol=2e-7)
    resumed, clean = step(defense, 6)
    assert clean["metrics"]["relation_graph_state"] == "no_anomaly_evidence"
    assert_same_model(resumed, FedAvgDefense(DefenseConfig()).aggregate(uploads()))


def test_group_zero_variant_removes_confirmed_members():
    from defenses.relation_graph.exclusion import GroupExclusionRelationGraphDefense

    defense = GroupExclusionRelationGraphDefense(
        DefenseConfig(enabled=True, type="relation_graph",
                      custom_params={"settings": {}}),
        num_clients=len(IDS))
    group = ("5", "6", "7")
    for number in (1, 2, 3):
        step(defense, number, coordinated=group)
    _, first = step(defense, 4, coordinated=group)
    assert first["metrics"]["relation_graph_state"] == "insufficient_evidence"
    _, second = step(defense, 5, coordinated=group)
    assert second["metrics"]["relation_graph_state"] == "anomalous_structure"
    assert [row["principals"] for row in second["groups"]] == [list(group)]
    expected_anchor = 0.0
    for row in second["clients"]:
        if row["principal_id"] in group:
            assert row["aggregation_weight"] == 0.0
            expected_anchor += row["nominal_weight"]
        else:
            assert row["aggregation_weight"] == row["nominal_weight"]
    anchor = second["metrics"]["relation_graph_anchor_weight"]
    assert anchor == pytest.approx(expected_anchor)
    weights = sum(row["aggregation_weight"] for row in second["clients"])
    assert weights + anchor == pytest.approx(1)


def test_round_wide_drift_without_boundary_evidence_keeps_aggregation():
    defense = make_defense()
    for number in (1, 2, 3):
        step(defense, number)
    step(defense, 4, coordinated=IDS)
    result, diag = step(defense, 5, coordinated=IDS)
    assert all(edge["strength"] > 0 for edge in diag["edges"])
    assert not diag["groups"]
    assert diag["metrics"]["relation_graph_state"] == "insufficient_evidence"
    assert_same_model(result, FedAvgDefense(DefenseConfig()).aggregate(uploads(coordinated=IDS)))


def test_graph_topology_changes_decision_with_identical_degrees_and_edge_values():
    identities = IDS[:7]
    pairs = tuple(combinations(identities, 2))
    settings = Settings(min_reference_rounds=1, min_edge_confirmations=1)
    baseline = tuple(EdgeObservation(i, j, 1., 0.) for i, j in pairs)
    reference = assess(1, identities, baseline, (), settings).observation
    triangles = set(combinations(identities[:3], 2)) | set(combinations(identities[3:6], 2))
    cycle = {(str(i), str(i+1)) for i in range(5)} | {("0", "5")}
    scores = []
    for structure in (triangles, cycle):
        edges = tuple(EdgeObservation(i, j, 1., float((i, j) in structure)) for i, j in pairs)
        scores.append(assess(2, identities, edges, (reference,), settings))
    assert len(scores[0].groups) == 2
    assert scores[1].state == "insufficient_evidence"
    assert not any(scores[1].risks.values())


def test_partial_participation_only_counts_joint_observations_and_expires_history():
    defense = make_defense()
    for number in (1, 2, 3):
        step(defense, number)
    step(defense, 4, IDS[2:])
    _, later = step(defense, 6, IDS[:4])
    edge = next(edge for edge in later["edges"] if (edge["left"], edge["right"]) == ("0", "1"))
    assert edge["pair_reference_observations"] == 3
    assert edge["effective_observations"] == 1
    _, expired = step(defense, 19, IDS[:4])
    assert expired["metrics"]["relation_graph_state"] == "insufficient_evidence"
    assert [frame["round_number"] for frame in defense.state_dict()["history"]] == [19]


def test_json_resume_and_failed_round_preserve_graph_state():
    original = make_defense()
    group = ("5", "6", "7")
    for number in (1, 2, 3):
        step(original, number)
    step(original, 4, coordinated=group)
    saved = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    restored = make_defense()
    restored.load_state_dict(saved)
    before = restored.state_dict()
    set_context(restored, 5)
    bad = uploads(coordinated=group)
    bad[0][0][1][0] = np.nan
    with pytest.raises(FloatingPointError):
        restored.aggregate(bad)
    assert restored.state_dict() == before
    expected, expected_diag = step(original, 5, coordinated=group)
    actual = restored.aggregate(uploads(coordinated=group))
    assert_same_model(actual, expected)
    assert restored.state_dict() == original.state_dict()
    assert restored.diagnostics() == expected_diag
    with pytest.raises(ValueError, match="fresh context"):
        restored.aggregate(uploads(coordinated=group))


def test_flower_aggregate_uses_server_partition_ids():
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from strategies.fed_strategy import FedSecStrategy

    strategy = FedSecStrategy(
        StrategyConfig(name="fedavg"), DefenseConfig(enabled=True, type="relation_graph"), BASE,
        num_clients=8, parameter_roles=ROLES, parameter_names=NAMES, scalable_parameter_indices={0, 1},
    )
    results = [
        (SimpleNamespace(cid=f"endpoint-{identity}", partition_id=int(identity)),
         FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), count, {"client_id": 999}))
        for identity, (params, count) in zip(IDS, uploads())
    ]
    output, metrics = strategy.aggregate_fit(1, results, [])
    assert_same_model(parameters_to_ndarrays(output), FedAvgDefense(DefenseConfig()).aggregate(uploads()))
    assert set(strategy.defense.last_client_aggregation_weights) == set(IDS)
    assert metrics["relation_graph_state"] == "insufficient_evidence"


def _frame(round_number, edges):
    return RoundObservation(round_number, tuple(sorted(edges, key=lambda edge: edge.pair)))


def test_high_similarity_ceiling_detects_aligned_clique():
    """Near the +1 cosine ceiling, headroom-aware scaling must still exceed z=3."""
    identities = tuple(str(index) for index in range(8))
    clique = ("0", "1", "2", "3")
    benign = ("4", "5", "6", "7")
    history = []
    for number in (1, 2, 3):
        edges = [EdgeObservation(left, right, 0.9, 0.84)
                 for left, right in combinations(clique, 2)]
        edges += [EdgeObservation(left, right, 0.9, 0.84)
                  for left, right in combinations(benign, 2)]
        edges += [EdgeObservation(*sorted((left, right)), 0.6, 0.50)
                  for left in clique for right in benign]
        history.append(_frame(number, edges))
    current = [EdgeObservation(left, right, 0.9, 0.99)
               for left, right in combinations(clique, 2)]
    current += [EdgeObservation(left, right, 0.9, 0.84)
                for left, right in combinations(benign, 2)]
    current += [EdgeObservation(*sorted((left, right)), 0.6, 0.50)
                for left in clique for right in benign]
    result = assess(4, identities, _frame(4, current).edges, tuple(history),
                    Settings(min_edge_confirmations=1))
    edge = next(row for row in result.edges
                if (row["left"], row["right"]) == ("0", "1"))
    assert edge["reference_center"] == pytest.approx(0.84)
    assert edge["score"] > 3.0
    assert result.state == "anomalous_structure"
    assert [group["principals"] for group in result.groups] == [["0", "1", "2", "3"]]
    assert all(result.risks[identity] > 0 for identity in clique)
    assert all(result.risks[identity] == 0 for identity in benign)


def test_two_node_single_edge_does_not_confirm_a_group():
    """A 2-node 1-edge excess is an edge event, not subgraph collusion."""
    identities = tuple(str(index) for index in range(8))
    history = [
        _frame(number, [EdgeObservation(left, right, 0.9, 0.0)
                        for left, right in combinations(identities, 2)])
        for number in (1, 2, 3)
    ]
    current = [EdgeObservation(left, right, 0.9,
                               1.0 if (left, right) == ("0", "1") else 0.0)
               for left, right in combinations(identities, 2)]
    result = assess(4, identities, _frame(4, current).edges, tuple(history),
                    Settings(min_edge_confirmations=1))
    edge = next(row for row in result.edges
                if (row["left"], row["right"]) == ("0", "1"))
    assert edge["score"] > 3.0
    assert result.state == "insufficient_evidence"
    assert not result.groups
    assert all(value == 0.0 for value in result.risks.values())


def test_bridge_node_is_pruned_from_dense_core():
    """A single bridge to a benign node must not join the confirmed clique."""
    identities = tuple(str(index) for index in range(8))
    history = [
        _frame(number, [EdgeObservation(left, right, 0.9, 0.0)
                        for left, right in combinations(identities, 2)])
        for number in (1, 2, 3)
    ]
    clique = ("0", "1", "2", "3")
    anomalous = {tuple(sorted(pair)) for pair in combinations(clique, 2)}
    anomalous.add(("0", "4"))
    current = [EdgeObservation(left, right, 0.9,
                               1.0 if (left, right) in anomalous else 0.0)
               for left, right in combinations(identities, 2)]
    result = assess(4, identities, _frame(4, current).edges, tuple(history),
                    Settings(min_edge_confirmations=1, min_group_density=0.8))
    assert result.state == "anomalous_structure"
    assert [group["principals"] for group in result.groups] == [["0", "1", "2", "3"]]
    assert result.groups[0]["edges"] == 6
    assert result.groups[0]["density"] == pytest.approx(1.0)
    assert all(result.risks[identity] > 0 for identity in clique)
    assert result.risks["4"] == 0.0
