from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from attacks.attack_client import (
    DBAClient,
    GaussianNoiseClient,
    RandomNoiseClient,
    SignFlipClient,
    get_dba_trigger_coords,
    stamp_dba_coords_,
    validate_attack_config,
)
from attacks.coordinator import AttackCoordinator, _flatten_delta, _pairwise_squared
from attacks.spec import attack_contract_payload, get_attack_spec
from client.fl_client import assign_dba_fragments
from config.config_loader import AttackConfig
from experiments import periodic_attack
from experiments import run as experiment_run
from experiments.rtc_v3 import byzantine as byzantine_experiment
from experiments.rtc_v3 import byzantine_analysis
from experiments.rtc_v3 import byzantine_agr_report
from experiments.rtc_v3 import byzantine_screen_report
from server.fl_server import evaluate_targeted_asr, label_flip_metrics_from_confusion
from strategies.fed_strategy import _weighted_avg_metrics


def _bare_client(cls, config: AttackConfig, global_params: list[np.ndarray]):
    client = object.__new__(cls)
    client.client_id = 7
    client.attack_cfg = config
    client._attack_active = True
    client._global_params_cache = [value.copy() for value in global_params]
    return client


def test_sign_flip_operates_on_delta_and_preserves_integer_buffers():
    global_params = [np.array([1.0, 2.0], np.float32), np.array([3], np.int64)]
    local = [np.array([2.0, 0.0], np.float32), np.array([4], np.int64)]
    client = _bare_client(SignFlipClient, AttackConfig(sign_flip_scale=2.0), global_params)
    attacked = client.on_after_fit(local, {})
    np.testing.assert_allclose(attacked[0], np.array([-1.0, 6.0], np.float32))
    np.testing.assert_array_equal(attacked[1], local[1])


def test_random_noise_is_norm_matched_for_rademacher():
    global_params = [np.zeros(4, np.float32)]
    local = [np.array([1.0, 2.0, 2.0, 1.0], np.float32)]
    config = AttackConfig(random_noise_scale=1.5, random_noise_distribution="rademacher")
    client = _bare_client(RandomNoiseClient, config, global_params)
    np.random.seed(123)
    attacked = client.on_after_fit(local, {})
    assert np.linalg.norm(attacked[0]) == pytest.approx(
        1.5 * np.linalg.norm(local[0]), rel=1e-6
    )


def test_gaussian_zero_variance_adds_configured_mean():
    params = [np.array([1.0, 2.0], np.float32)]
    config = AttackConfig(gaussian_noise_mean=0.25, gaussian_noise_std=0.0)
    client = _bare_client(GaussianNoiseClient, config, params)
    attacked = client.on_after_fit(params, {})
    np.testing.assert_allclose(attacked[0], np.array([1.25, 2.25], np.float32))


@pytest.mark.parametrize("attack", ["lie", "alie", "min_max", "min_sum"])
def test_coordinated_attacks_replace_only_active_malicious_updates(attack: str):
    global_params = [np.zeros(2, np.float32), np.array([10], np.int64)]
    vectors = [
        np.array([0.9, 1.0], np.float32),
        np.array([1.0, 1.1], np.float32),
        np.array([1.1, 0.9], np.float32),
        np.array([9.0, 9.0], np.float32),
        np.array([8.0, 8.0], np.float32),
    ]
    updates = [([value, np.array([index], np.int64)], 1) for index, value in enumerate(vectors)]
    original = [[parameter.copy() for parameter in item[0]] for item in updates]
    coordinator = AttackCoordinator(
        AttackConfig(type=attack, enabled=True, coordinated_attack_knowledge="all_updates"),
        num_clients=5,
    )
    result = coordinator.transform(
        server_round=1,
        updates=updates,
        global_params=global_params,
        client_ids=[str(index) for index in range(5)],
        malicious=[False, False, False, True, True],
        attack_active=[False, False, False, True, True],
    )
    assert result.metrics["attack_coordinator_applied"] is True
    for index in range(3):
        np.testing.assert_array_equal(result.updates[index][0][0], original[index][0])
    np.testing.assert_allclose(result.updates[3][0][0], result.updates[4][0][0])
    np.testing.assert_array_equal(result.updates[3][0][1], original[3][1])
    for index, parameters in enumerate(original):
        np.testing.assert_array_equal(updates[index][0][0], parameters[0])


@pytest.mark.parametrize("attack", ["lie", "alie", "min_max", "min_sum"])
def test_coordinated_attacks_preserve_unselected_floating_buffers(attack: str):
    global_params = [
        np.zeros(2, np.float32),
        np.zeros(2, np.float32),
        np.ones(2, np.float32),
        np.array([0], np.int64),
    ]
    trainable = [
        np.array([0.9, 1.0], np.float32),
        np.array([1.0, 1.1], np.float32),
        np.array([1.1, 0.9], np.float32),
        np.array([9.0, 9.0], np.float32),
        np.array([8.0, 8.0], np.float32),
    ]
    updates = [
        (
            [
                value,
                np.full(2, 10.0 + index, np.float32),
                np.full(2, 0.2 + 0.1 * index, np.float32),
                np.array([index], np.int64),
            ],
            1,
        )
        for index, value in enumerate(trainable)
    ]
    original = [[parameter.copy() for parameter in params] for params, _ in updates]
    result = AttackCoordinator(
        AttackConfig(
            type=attack,
            enabled=True,
            coordinated_attack_knowledge="all_updates",
        ),
        num_clients=5,
        scalable_parameter_indices={0},
    ).transform(
        server_round=1,
        updates=updates,
        global_params=global_params,
        client_ids=[str(index) for index in range(5)],
        malicious=[False, False, False, True, True],
        attack_active=[False, False, False, True, True],
    )

    np.testing.assert_allclose(result.updates[3][0][0], result.updates[4][0][0])
    for client_index in (3, 4):
        for parameter_index in (1, 2, 3):
            np.testing.assert_array_equal(
                result.updates[client_index][0][parameter_index],
                original[client_index][parameter_index],
            )
        assert np.all(result.updates[client_index][0][2] > 0.0)


@pytest.mark.parametrize("attack", ["min_max", "min_sum"])
def test_optimized_poison_respects_published_distance_constraint(attack: str):
    global_params = [np.zeros(3, np.float32)]
    benign = np.array(
        [[0.9, 1.0, 1.1], [1.0, 1.2, 0.8], [1.1, 0.9, 1.0]],
        dtype=np.float32,
    )
    updates = [([row.copy()], 1) for row in benign]
    updates.extend([([np.full(3, 8.0, np.float32)], 1) for _ in range(2)])
    result = AttackCoordinator(
        AttackConfig(type=attack, enabled=True), num_clients=5
    ).transform(
        server_round=1,
        updates=updates,
        global_params=global_params,
        client_ids=[str(index) for index in range(5)],
        malicious=[False, False, False, True, True],
        attack_active=[False, False, False, True, True],
    )
    crafted = _flatten_delta(result.updates[3][0], global_params)
    distances = _pairwise_squared(benign.astype(np.float64))
    observed = np.sum((benign.astype(np.float64) - crafted) ** 2, axis=1)
    if attack == "min_max":
        assert np.max(observed) <= np.max(distances) + 1e-6
    else:
        assert np.sum(observed) <= np.max(np.sum(distances, axis=1)) + 1e-6
    assert result.metrics["attack_coordinator_r_max"] == pytest.approx(
        np.sqrt(np.max(observed) / np.max(distances))
    )
    assert result.metrics["attack_coordinator_r_sum"] == pytest.approx(
        np.sum(observed) / np.max(np.sum(distances, axis=1))
    )


def test_lie_records_actual_trainable_coordinate_sign_flippability():
    global_params = [np.zeros(4, np.float32)]
    benign = np.array(
        [[0.5, -2.0, 0.0, 1.0], [1.5, -2.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    updates = [([row.copy()], 1) for row in benign]
    updates.extend([([np.full(4, 8.0, np.float32)], 1) for _ in range(2)])
    result = AttackCoordinator(
        AttackConfig(type="lie", enabled=True, lie_z=3.0), num_clients=4
    ).transform(
        server_round=1,
        updates=updates,
        global_params=global_params,
        client_ids=[str(index) for index in range(4)],
        malicious=[False, False, True, True],
        attack_active=[False, False, True, True],
    )

    # mean=[1,-2,0,2], std=[.5,0,0,1], crafted=[-.5,-2,0,-1]
    assert result.metrics["attack_coordinator_sign_flippable_ratio"] == pytest.approx(0.5)


def test_dba_rank_mapping_covers_fragments_for_nonconsecutive_ids():
    mapping = assign_dba_fragments({2, 7, 11, 19}, 4)
    assert mapping == {2: 0, 7: 1, 11: 2, 19: 3}


def test_dba_applies_exactly_one_declared_delta_scaling_step():
    global_params = [np.array([1.0, -1.0], np.float32)]
    local = [np.array([2.0, 1.0], np.float32)]
    config = AttackConfig(
        type="dba", dba_scale_update=True, dba_boost_factor=10.0
    )
    client = _bare_client(DBAClient, config, global_params)
    metrics = {}
    attacked = client.on_after_fit(local, metrics)
    np.testing.assert_allclose(
        attacked[0], global_params[0] + 10.0 * (local[0] - global_params[0])
    )
    assert metrics["dba_update_scaled"] is True


def test_dba_boost_scales_parameters_but_preserves_batchnorm_buffers():
    model = torch.nn.BatchNorm1d(2)
    global_params = [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]
    local = [value.copy() for value in global_params]
    # weight and bias are learned parameters; running statistics are buffers.
    local[0] += 0.5
    local[1] += 0.25
    local[2] += 2.0
    local[3] = np.full_like(local[3], 0.25)
    local[4] += 1

    client = _bare_client(
        DBAClient,
        AttackConfig(type="dba", dba_scale_update=True, dba_boost_factor=10.0),
        global_params,
    )
    client.model = model
    attacked = client.on_after_fit(local, {})

    np.testing.assert_allclose(attacked[0], global_params[0] + 10.0 * 0.5)
    np.testing.assert_allclose(attacked[1], global_params[1] + 10.0 * 0.25)
    np.testing.assert_array_equal(attacked[2], local[2])
    np.testing.assert_array_equal(attacked[3], local[3])
    np.testing.assert_array_equal(attacked[4], local[4])
    assert np.min(attacked[3]) >= 0.0


def test_string_client_metrics_are_not_numerically_aggregated():
    metrics = _weighted_avg_metrics([
        (2, {"train_loss": 1.0, "dba_scaling_mode": "fixed_client_boost"}),
        (3, {"train_loss": 2.0, "custom_note": "replication"}),
    ])
    assert metrics == {"train_loss": pytest.approx(1.6)}


def test_nonfinite_numeric_client_metric_fails_fast():
    with pytest.raises(FloatingPointError, match="Non-finite client metric"):
        _weighted_avg_metrics([(1, {"train_loss": float("nan")})])


def test_aggregation_aware_backdoor_scaling_compensates_actual_fedavg_share():
    global_params = [np.zeros(2, np.float32)]
    updates = [
        ([np.full(2, value, np.float32)], weight)
        for value, weight in ((1.0, 2), (2.0, 3), (3.0, 1), (3.0, 4))
    ]
    result = AttackCoordinator(
        AttackConfig(
            enabled=True,
            type="dba",
            dba_scale_update=True,
            aggregation_aware_scaling=True,
            replacement_gain=0.8,
        ),
        num_clients=4,
    ).transform(
        server_round=1,
        updates=updates,
        global_params=global_params,
        client_ids=["0", "1", "2", "3"],
        malicious=[False, False, True, True],
        attack_active=[False, False, True, True],
    )

    np.testing.assert_array_equal(result.updates[0][0][0], updates[0][0][0])
    np.testing.assert_array_equal(result.updates[1][0][0], updates[1][0][0])
    np.testing.assert_allclose(result.updates[2][0][0], np.full(2, 4.8))
    np.testing.assert_allclose(result.updates[3][0][0], np.full(2, 4.8))
    assert result.metrics["attack_scaling_malicious_weight_share"] == pytest.approx(0.5)
    assert result.metrics["attack_scaling_client_multiplier"] == pytest.approx(1.6)
    assert result.metrics["attack_scaling_effective_gain"] == pytest.approx(0.8)
    assert result.metrics["attack_scaling_raw_trainable_delta_norm"] == pytest.approx(
        np.sqrt(36.0)
    )
    assert result.metrics["attack_scaling_scaled_trainable_delta_norm"] == pytest.approx(
        1.6 * np.sqrt(36.0)
    )


def test_server_scaling_preserves_unselected_floating_buffers():
    global_params = [
        np.array([1.0], np.float32),
        np.array([1.0], np.float32),
        np.array([4], np.int64),
    ]
    local = [
        np.array([2.0], np.float32),
        np.array([0.25], np.float32),
        np.array([5], np.int64),
    ]
    result = AttackCoordinator(
        AttackConfig(
            enabled=True,
            type="dba",
            dba_scale_update=True,
            aggregation_aware_scaling=True,
            replacement_gain=1.0,
        ),
        num_clients=1,
        scalable_parameter_indices={0},
    ).transform(
        server_round=1,
        updates=[(local, 1)],
        global_params=global_params,
        client_ids=["0"],
        malicious=[True],
        attack_active=[True],
    )

    np.testing.assert_array_equal(result.updates[0][0][1], local[1])
    np.testing.assert_array_equal(result.updates[0][0][2], local[2])


def test_dba_cifar_fragments_match_official_geometry_and_union():
    fragments = [
        set(
            get_dba_trigger_coords(
                fragment_index=index,
                image_shape=(3, 32, 32),
                trigger_size=3,
                dba_trigger_num=4,
                gap=3,
                base_row=0,
                base_col=0,
            )
        )
        for index in range(4)
    ]
    assert fragments == [
        {(0, col) for col in range(0, 6)},
        {(0, col) for col in range(9, 15)},
        {(4, col) for col in range(0, 6)},
        {(4, col) for col in range(9, 15)},
    ]
    assert all(len(fragment) == 6 for fragment in fragments)
    assert sum(len(fragment) for fragment in fragments) == len(set().union(*fragments))


def test_dba_legacy_square_geometry_is_explicit_only():
    coords = get_dba_trigger_coords(
        1,
        (3, 32, 32),
        trigger_size=3,
        dba_trigger_num=4,
        pattern_mode="legacy_blocks",
        gap=3,
    )
    assert len(coords) == 9
    assert min(coords) == (0, 6)


def test_dba_official_white_value_matches_cifar_normalization():
    tensor = torch.zeros(3, 32, 32)
    stamp_dba_coords_(
        tensor,
        [(0, 0)],
        trigger_value=1.0,
        value_mode="cifar10_normalized_white",
    )
    expected = torch.tensor(
        [
            (1.0 - 0.4914) / 0.2023,
            (1.0 - 0.4822) / 0.1994,
            (1.0 - 0.4465) / 0.2010,
        ]
    )
    torch.testing.assert_close(tensor[:, 0, 0], expected)
    assert torch.count_nonzero(tensor[:, 0, 1:]) == 0


def test_attack_config_validation_rejects_ambiguous_random_distribution():
    with pytest.raises(ValueError, match="random_noise_distribution"):
        validate_attack_config(
            AttackConfig(type="random_noise", random_noise_distribution="uniform"),
            num_classes=10,
        )


def test_alie_alias_is_numerically_identical_to_lie():
    global_params = [np.zeros(2, np.float32)]
    updates = [
        ([np.array([0.9, 1.0], np.float32)], 1),
        ([np.array([1.0, 1.1], np.float32)], 1),
        ([np.array([1.1, 0.9], np.float32)], 1),
        ([np.array([8.0, 8.0], np.float32)], 1),
        ([np.array([9.0, 9.0], np.float32)], 1),
    ]
    outputs = []
    for name in ("lie", "alie"):
        result = AttackCoordinator(
            AttackConfig(type=name, enabled=True, lie_z=0.5), num_clients=5
        ).transform(
            server_round=1,
            updates=updates,
            global_params=global_params,
            client_ids=[str(index) for index in range(5)],
            malicious=[False, False, False, True, True],
            attack_active=[False, False, False, True, True],
        )
        outputs.append(result.updates[3][0][0])
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_byzantine_matrix_is_defense_paired_and_excludes_alias_by_default():
    args = experiment_run.parse_args(
        [
            "--profile",
            "rtc-byzantine",
            "--seeds",
            "42",
            "--malicious-fractions",
            "0.2",
            "--dry-run",
        ]
    )
    args.malicious_fraction = 0.2
    rows = byzantine_experiment.build_matrix(args)
    assert len(rows) == len(byzantine_experiment.CANONICAL_ATTACKS) * len(
        byzantine_experiment.DEFAULT_DEFENSES
    )
    assert {row["attack"] for row in rows} == set(
        byzantine_experiment.CANONICAL_ATTACKS
    )
    for attack in byzantine_experiment.CANONICAL_ATTACKS:
        group = [row for row in rows if row["attack"] == attack]
        assert len({row["attack_contract_hash"] for row in group}) == 1
        assert len({row["attack_implementation_hash"] for row in group}) == 1


def test_clean_counterfactual_contract_binds_runner_materialized_boost(tmp_path):
    args = experiment_run.parse_args(
        [
            "--profile",
            "rtc-byzantine",
            "--attacks",
            "random_noise",
            "--defenses",
            "fedavg",
            "--seeds",
            "42",
            "--malicious-fractions",
            "0.2",
            "--smoke",
            "--dry-run",
            "--output",
            str(tmp_path),
        ]
    )
    args.malicious_fraction = 0.2
    byzantine_experiment.run(args)
    matrix = pd.read_csv(tmp_path / "experiment_matrix.csv")
    clean = matrix[matrix["attack"] == "none"].iloc[0]
    parameters = {
        key: clean[key]
        for key in byzantine_experiment._ATTACK_PARAMETER_KEYS
        if key in clean and not pd.isna(clean[key])
    }
    runtime = byzantine_experiment._runtime_attack_config(
        "none",
        parameters,
        malicious_fraction=0.0,
        attack_start_round=int(clean["attack_start_round"]),
        attack_end_round=int(clean["attack_end_round"]),
    )
    expected = attack_contract_payload("none", vars(runtime))["implementation_hash"]
    assert clean["attack_implementation_hash"] == expected


def test_runtime_attack_config_maps_label_flip_cli_names():
    runtime = byzantine_experiment._runtime_attack_config(
        "label_flip_targeted",
        {
            "label_flip_source_label": 5,
            "label_flip_target_label": 3,
            "label_flip_poison_fraction": 0.75,
        },
        malicious_fraction=0.2,
        attack_start_round=11,
        attack_end_round=-1,
    )
    assert runtime.source_label == 5
    assert runtime.target_label == 3
    assert runtime.label_flip_poison_fraction == 0.75


def test_dba_evaluation_reports_full_and_each_local_trigger_asr():
    class AlwaysTarget(torch.nn.Module):
        def forward(self, values):
            logits = torch.zeros((values.shape[0], 10), device=values.device)
            logits[:, 3] = 1.0
            return logits

    loader = DataLoader(
        TensorDataset(
            torch.zeros((7, 3, 32, 32)),
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        ),
        batch_size=3,
    )
    metrics = evaluate_targeted_asr(
        AlwaysTarget(),
        loader,
        torch.device("cpu"),
        AttackConfig(enabled=True, type="dba", backdoor_target_label=3),
    )
    assert metrics is not None
    assert metrics["asr_valid"] is True
    assert metrics["asr_total"] == 6
    assert metrics["dba_full_trigger_asr"] == 1.0
    assert metrics["dba_local_asr_mean"] == 1.0
    assert metrics["dba_local_asr_min"] == 1.0
    assert metrics["dba_local_asr_max"] == 1.0
    assert {
        key for key in metrics if key.startswith("dba_local_asr_fragment_")
    } == {f"dba_local_asr_fragment_{index}" for index in range(4)}


def test_dba_evaluation_rejects_nonfinite_logits_before_argmax():
    class NonFiniteModel(torch.nn.Module):
        def forward(self, values):
            return torch.full(
                (values.shape[0], 10), float("nan"), device=values.device
            )

    loader = DataLoader(
        TensorDataset(torch.zeros((4, 3, 32, 32)), torch.tensor([1, 2, 3, 4])),
        batch_size=2,
    )
    metrics = evaluate_targeted_asr(
        NonFiniteModel(),
        loader,
        torch.device("cpu"),
        AttackConfig(enabled=True, type="dba", backdoor_target_label=0),
    )

    assert metrics is not None
    assert metrics["asr_valid"] is False
    assert metrics["asr_invalid_reason"] == "nonfinite_logits"
    assert np.isnan(metrics["asr"])
    assert np.isnan(metrics["dba_full_trigger_asr"])


def test_targeted_label_flip_reports_source_recall_and_source_to_target_rate():
    confusion = np.zeros((10, 10), dtype=np.int64)
    confusion[5, 5] = 6
    confusion[5, 3] = 4
    confusion[3, 3] = 10
    metrics = label_flip_metrics_from_confusion(
        confusion,
        AttackConfig(
            enabled=True,
            type="label_flip_targeted",
            source_label=5,
            target_label=3,
        ),
    )
    assert metrics["source_recall"] == pytest.approx(0.6)
    assert metrics["asr"] == pytest.approx(0.4)
    assert metrics["source_to_target_count"] == 4


def test_targeted_label_flip_paired_summary_reports_source_recall_drop():
    clean = {
        "attack": "none",
        "defense": "fedavg",
        "trial_plan_hash": "plan",
        "final_accuracy": 0.8,
        "final_class_recall_5": 0.75,
    }
    attacked = {
        "attack": "label_flip_targeted",
        "defense": "fedavg",
        "trial_plan_hash": "plan",
        "final_accuracy": 0.78,
        "label_flip_source_label": 5,
        "final_source_recall": 0.40,
    }
    summary = periodic_attack.add_accuracy_drop(pd.DataFrame([clean, attacked]))
    row = summary[summary["attack"] == "label_flip_targeted"].iloc[0]
    assert row["accuracy_drop"] == pytest.approx(0.02)
    assert row["source_recall_drop"] == pytest.approx(0.35)


def test_dba_summary_separates_window_participating_and_residual_asr():
    rounds = pd.DataFrame({
        "round": [1, 2, 3, 4],
        "planned_attack_active": [1, 1, 0, 1],
        "fit_selected_active_attackers": [1, 0, 0, 1],
        "server_asr": [0.8, 0.4, 0.2, np.nan],
        "server_asr_valid": [True, True, True, False],
        "server_loss": [1.0, 0.9, 0.8, np.nan],
        "server_accuracy": [0.7, 0.72, 0.74, 0.1],
    })
    spec = {
        "attack": "dba",
        "attack_group": "targeted",
        "period": "continuous_1_0",
        "on_rounds": 1,
        "off_rounds": 0,
        "malicious_fraction": 0.2,
        "seed": 42,
        "defense": "fedavg",
        "defense_type": "none",
        "custom_params": {},
        "attack_start_round": 1,
    }

    summary = periodic_attack.summarize_run(rounds, spec)

    assert summary["window_asr"] == pytest.approx(0.6)
    assert summary["participating_asr"] == pytest.approx(0.8)
    assert summary["residual_asr"] == pytest.approx(0.2)
    assert summary["participating_asr_rounds"] == 1
    assert summary["window_asr_rounds"] == 2
    assert summary["invalid_asr_rounds"] == 1
    assert summary["nan_rounds"] == 1
    assert summary["asr_valid"] is False
    assert summary["collapsed_invalid"] is True


def test_summary_ignores_not_applicable_round_zero_update_validity_cells():
    rounds = pd.DataFrame({
        "round": [0, 1],
        "planned_attack_active": [0, 1],
        "fit_received_updates_finite": [np.nan, True],
        "fit_coordinated_updates_finite": [np.nan, True],
        "fit_aggregate_parameters_finite": [np.nan, True],
        "server_loss": [2.0, 1.0],
        "server_accuracy": [0.1, 0.5],
    })
    spec = {
        "attack": "lie", "attack_group": "untargeted",
        "period": "continuous_1_0", "malicious_fraction": 0.2,
        "seed": 42, "defense": "fedavg", "defense_type": "none",
        "custom_params": {}, "attack_start_round": 1,
    }
    summary = periodic_attack.summarize_run(rounds, spec)
    assert summary["invalid_update_rounds"] == 0
    assert summary["numerical_divergence"] is False


def test_screening_matrix_has_three_fedavg_strengths_per_attack():
    args = experiment_run.parse_args(
        [
            "--profile",
            "rtc-byzantine-screen",
            "--seeds",
            "42",
            "--malicious-fractions",
            "0.2",
            "--dry-run",
        ]
    )
    args.malicious_fraction = 0.2
    rows = byzantine_experiment.build_screening_matrix(args)
    assert len(rows) == 3 * len(byzantine_experiment.CANONICAL_ATTACKS)
    assert {row["defense"] for row in rows} == {"fedavg"}
    for attack in byzantine_experiment.CANONICAL_ATTACKS:
        levels = {
            row["strength_level"] for row in rows if row["attack"] == attack
        }
        assert levels == {"weak", "medium", "strong"}
    for row in rows:
        expected_start = 11 if row["attack"] in {"dba", "scaling_backdoor"} else 1
        assert row["attack_start_round"] == expected_start


def test_screening_matrix_can_select_only_strong_strength():
    args = experiment_run.parse_args(
        [
            "--profile", "rtc-byzantine-screen",
            "--attacks", "lie,min_max,min_sum",
            "--strength-levels", "strong",
            "--seeds", "42",
            "--malicious-fractions", "0.3",
            "--dry-run",
        ]
    )
    args.malicious_fraction = 0.3
    rows = byzantine_experiment.build_screening_matrix(args)
    assert len(rows) == 3
    assert {row["attack"] for row in rows} == {"lie", "min_max", "min_sum"}
    assert {row["strength_level"] for row in rows} == {"strong"}


def test_strength_selector_uses_only_fedavg_and_weakest_effective_level():
    frame = pd.DataFrame(
        [
            {
                "attack": "sign_flip",
                "defense": "fedavg",
                "attack_group": "untargeted",
                "strength_level": level,
                "accuracy_drop": value,
                "numerical_divergence": False,
            }
            for level, value in (("weak", 0.01), ("medium", 0.04), ("strong", 0.09))
        ]
        + [
            {
                "attack": "sign_flip",
                "defense": "rtc_full",
                "attack_group": "untargeted",
                "strength_level": "weak",
                "accuracy_drop": 0.99,
                "numerical_divergence": False,
            }
        ]
    )
    recommendations, payload, gates = byzantine_experiment.select_screened_strengths(
        frame
    )
    assert recommendations.iloc[0]["selected_strength"] == "weak"
    assert payload["attacks"]["sign_flip"]["strength_level"] == "weak"
    assert "scale=1" in recommendations.iloc[0]["selection_reason"]
    assert gates[0]["passed"] is True


def test_dba_strength_selector_rejects_collapsed_configuration():
    frame = pd.DataFrame([
        {
            "attack": "dba", "defense": "fedavg", "attack_group": "targeted",
            "strength_level": "weak", "participating_asr": 0.35,
            "active_accuracy": 0.70, "asr_valid": True,
            "accuracy_drop": 0.05, "numerical_divergence": False,
        },
        {
            "attack": "dba", "defense": "fedavg", "attack_group": "targeted",
            "strength_level": "medium", "participating_asr": 1.0,
            "active_accuracy": 0.10, "asr_valid": False,
            "accuracy_drop": 0.20, "numerical_divergence": True,
        },
        {
            "attack": "dba", "defense": "fedavg", "attack_group": "targeted",
            "strength_level": "strong", "participating_asr": 0.80,
            "active_accuracy": 0.65, "asr_valid": True,
            "accuracy_drop": 0.05, "numerical_divergence": False,
        },
    ])

    recommendations, payload, gates = byzantine_experiment.select_screened_strengths(
        frame
    )

    result = recommendations.iloc[0]
    assert result["metric"] == "participating_asr"
    assert result["selected_strength"] == "strong"
    assert result["medium_status"] == "COLLAPSED/INVALID"
    assert result["invalid_strengths"] == "medium"
    assert payload["attacks"]["dba"]["strength_level"] == "strong"
    assert gates[0]["passed"] is True


def test_dba_calibration_grid_avoids_known_collapsing_strength():
    grid = byzantine_experiment.STRENGTH_GRID["dba"]
    assert (grid["weak"]["replacement_gain"], grid["weak"]["poison_fraction"]) == (0.4, 0.1)
    assert (grid["medium"]["replacement_gain"], grid["medium"]["poison_fraction"]) == (0.7, 0.2)
    assert (grid["strong"]["replacement_gain"], grid["strong"]["poison_fraction"]) == (1.0, 0.3)
    assert all(cell["aggregation_aware_scaling"] for cell in grid.values())


def test_multi_krum_row_selects_more_than_one_candidate():
    args = experiment_run.parse_args([
        "--profile", "rtc-byzantine", "--attacks", "lie",
        "--defenses", "multi_krum", "--seeds", "42",
        "--malicious-fractions", "0.2", "--dry-run",
    ])
    args.malicious_fraction = 0.2
    row = byzantine_experiment.build_matrix(args)[0]
    assert row["defense"] == "multi_krum"
    assert row["krum_num_to_select"] > 1


def test_robust_agr_report_requires_complete_matrix_and_marks_collapse(tmp_path):
    rows = []
    for attack in byzantine_experiment.ROBUST_AGR_REPRODUCTION_ATTACKS:
        for defense in byzantine_experiment.ROBUST_AGR_REPRODUCTION_DEFENSES:
            rows.append({
                "run_id": f"{attack}-{defense}",
                "attack": attack,
                "defense": defense,
                "final_accuracy": 0.7,
                "accuracy_drop": 0.1,
                "numerical_divergence": attack == "min_sum" and defense == "krum",
                "collapsed_invalid": False,
                "model_random_guess": False,
            })
    pd.DataFrame(rows).to_csv(tmp_path / "byzantine_attack_runs.csv", index=False)
    detailed, aggregate = byzantine_agr_report.summarize(tmp_path)
    invalid = detailed[
        (detailed["attack"] == "min_sum") & (detailed["defense"] == "krum")
    ].iloc[0]
    assert invalid["reproduction_status"] == "COLLAPSED/INVALID"
    assert len(aggregate) == 12
    comparison = byzantine_agr_report.optimization_comparison(aggregate)
    assert set(comparison["attack"]) == {"min_max", "min_sum"}

    pd.DataFrame(rows[:-1]).to_csv(
        tmp_path / "byzantine_attack_runs.csv", index=False
    )
    with pytest.raises(ValueError, match="matrix is incomplete"):
        byzantine_agr_report.summarize(tmp_path)


def test_screening_outcome_distinguishes_too_weak_from_collapsed():
    summary = pd.DataFrame([
        {
            "attack": "lie", "defense": "fedavg", "strength_level": "weak",
            "accuracy_drop": 0.01,
        },
        {
            "attack": "lie", "defense": "fedavg", "strength_level": "medium",
            "accuracy_drop": 0.05,
        },
        {
            "attack": "lie", "defense": "fedavg", "strength_level": "strong",
            "accuracy_drop": 0.30,
        },
    ])
    recommendations = pd.DataFrame([{
        "attack": "lie", "metric": "accuracy_drop", "threshold": 0.03,
        "weak_status": "VALID", "medium_status": "VALID",
        "strong_status": "COLLAPSED/INVALID",
    }])
    annotated = byzantine_experiment.annotate_screening_outcomes(
        summary, recommendations
    ).set_index("strength_level")
    assert bool(annotated.loc["weak", "attack_effect_too_weak"]) is True
    assert bool(annotated.loc["medium", "successful_attack"]) is True
    assert annotated.loc["strong", "reproduction_status"] == "COLLAPSED/INVALID"


def test_all_reverse_screen_requires_non_decreasing_mapping_effect():
    frame = pd.DataFrame([
        {
            "attack": "label_flip_all_reverse", "defense": "fedavg",
            "attack_group": "untargeted", "strength_level": level,
            "participating_asr": value, "active_accuracy": 0.7,
            "asr_valid": True, "numerical_divergence": False,
        }
        for level, value in (("weak", 0.20), ("medium", 0.10), ("strong", 0.05))
    ])
    recommendations, payload, gates = byzantine_experiment.select_screened_strengths(
        frame
    )
    assert recommendations.iloc[0]["selected_strength"] == ""
    assert "label_flip_all_reverse" not in payload["attacks"]
    assert gates[0]["passed"] is False


def test_plan_clean_execution_is_interleaved_after_last_defense():
    matrix = [
        {"pairing_group_id": "a", "defense": "fedavg"},
        {"pairing_group_id": "a", "defense": "rtc_full"},
        {"pairing_group_id": "b", "defense": "fedavg"},
    ]
    clean = [
        {"pairing_group_id": "a", "counterfactual_for": "a", "defense": "clean"},
        {"pairing_group_id": "b", "counterfactual_for": "b", "defense": "clean"},
    ]
    ordered = byzantine_experiment._interleave_plan_clean(matrix, clean)
    assert [(row["pairing_group_id"], row["defense"]) for row in ordered] == [
        ("a", "fedavg"),
        ("a", "rtc_full"),
        ("a", "clean"),
        ("b", "fedavg"),
        ("b", "clean"),
    ]


def test_byzantine_analysis_paired_effects_uses_plan_and_positive_is_better():
    rows = []
    for seed, fedavg, rtc in ((42, 0.30, 0.10), (43, 0.20, 0.15)):
        for defense, value in (("fedavg", fedavg), ("rtc_full", rtc)):
            rows.append(
                {
                    "attack": "dba",
                    "defense": defense,
                    "seed": seed,
                    "trial_plan_hash": f"plan-{seed}",
                    "active_asr_auc": value,
                    "accuracy_drop_auc": 0.0,
                }
            )
    effects = byzantine_analysis.paired_effects(pd.DataFrame(rows))
    rtc = effects.loc[effects["defense"] == "rtc_full"].iloc[0]
    assert rtc["metric"] == "active_asr_auc"
    assert rtc["n"] == 2
    assert rtc["mean_improvement"] == pytest.approx(0.125)


def test_byzantine_analysis_three_seed_statistics_are_descriptive():
    rows = []
    for seed, fedavg, rtc in ((42, 0.30, 0.10), (43, 0.20, 0.15), (44, 0.40, 0.20)):
        for defense, value in (("fedavg", fedavg), ("rtc_full", rtc)):
            rows.append(
                {
                    "attack": "dba",
                    "defense": defense,
                    "seed": seed,
                    "trial_plan_hash": f"plan-{seed}",
                    "active_asr_auc": value,
                }
            )
    result = byzantine_analysis.paired_rtc_statistics(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["n"] == 3
    assert bool(row["descriptive_only"]) is True
    assert pd.isna(row["p_value"])
    assert row["mean_improvement"] == pytest.approx(0.15)


def test_byzantine_analysis_complete_matrix_rejects_missing_defense():
    summary = pd.DataFrame(
        [
            {
                "attack": "sign_flip",
                "defense": "fedavg",
                "seed": 42,
                "trial_plan_hash": "plan",
            }
        ]
    )
    with pytest.raises(ValueError, match="formal matrix is incomplete"):
        byzantine_analysis.validate_complete_matrix(
            summary,
            attacks=("sign_flip",),
            defenses=("fedavg", "rtc_full"),
            seeds=(42,),
        )


def test_random_attack_provenance_uses_the_verified_primary_paper_slug():
    for attack in ("gaussian_noise", "random_noise"):
        assert get_attack_spec(attack).source_url.endswith("/huang24u.html")


def test_both_label_flip_variants_use_the_data_poisoning_source():
    targeted = get_attack_spec("label_flip_targeted").source_url
    reverse = get_attack_spec("label_flip_all_reverse").source_url
    assert reverse == targeted
    assert "2007.08432" in reverse


def test_screen_report_allows_strength_rejection_but_rejects_pairing_failure(tmp_path):
    pd.DataFrame(
        [
            {
                "attack": "sign_flip",
                "defense": "fedavg",
                "strength_level": level,
                "trial_plan_hash": "same-plan",
                "accuracy_drop": value,
            }
            for level, value in (("weak", 0.01), ("medium", 0.02), ("strong", 0.03))
        ]
    ).to_csv(tmp_path / "byzantine_attack_runs.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "trial_plan_hash_match", "passed": True},
            {"gate": "attack_strength:sign_flip", "passed": False},
        ]
    ).to_csv(tmp_path / "quality_gates.csv", index=False)
    pd.DataFrame(
        [{"attack": "sign_flip", "selected_strength": "", "passed": False}]
    ).to_csv(tmp_path / "attack_strength_screening.csv", index=False)
    attacked, _ = byzantine_screen_report.load_screen(tmp_path)
    assert len(attacked) == 3
    gates = pd.read_csv(tmp_path / "quality_gates.csv")
    gates.loc[gates["gate"] == "trial_plan_hash_match", "passed"] = False
    gates.to_csv(tmp_path / "quality_gates.csv", index=False)
    with pytest.raises(ValueError, match="non-strength screening gates failed"):
        byzantine_screen_report.load_screen(tmp_path)


def test_screen_report_plots_partial_dba_matrix_with_participating_asr(tmp_path):
    pd.DataFrame([
        {
            "attack": "dba",
            "defense": "fedavg",
            "strength_level": level,
            "trial_plan_hash": "same-plan",
            "participating_asr": value,
        }
        for level, value in (("weak", 0.3), ("medium", 0.5), ("strong", 0.7))
    ]).to_csv(tmp_path / "byzantine_attack_runs.csv", index=False)
    pd.DataFrame([
        {"gate": "trial_plan_hash_match", "passed": True},
        {"gate": "attack_strength:dba", "passed": True},
    ]).to_csv(tmp_path / "quality_gates.csv", index=False)
    pd.DataFrame([
        {"attack": "dba", "selected_strength": "weak", "passed": True}
    ]).to_csv(tmp_path / "attack_strength_screening.csv", index=False)

    path = byzantine_screen_report.run(tmp_path)

    assert path.is_file()
    assert path.name == "attack_strength_screening.png"


def test_defense_assumption_audit_counts_variable_malicious_participation(tmp_path):
    round_file = tmp_path / "krum-run.csv"
    pd.DataFrame(
        {
            "round": [0, 1, 2, 3],
            "fit_selected_malicious_clients": [np.nan, 1, 2, 3],
            "fit_selected_benign_clients": [np.nan, 9, 8, 7],
        }
    ).to_csv(round_file, index=False)
    summary = pd.DataFrame(
        [
            {
                "run_id": "krum-run",
                "attack": "sign_flip",
                "seed": 42,
                "defense": "krum",
                "krum_num_malicious": 2,
                "trim_fraction": 0.2,
            }
        ]
    )
    audit = byzantine_analysis.defense_assumption_audit(
        summary, {"krum-run": round_file}
    )
    assert audit.iloc[0]["violating_rounds"] == 1
    assert audit.iloc[0]["violation_rate"] == pytest.approx(1 / 3)
    assert audit.iloc[0]["max_selected_malicious"] == 3


def test_rtc_mechanism_summary_uses_attack_active_rounds_only(tmp_path):
    path = tmp_path / "rtc-run.csv"
    pd.DataFrame(
        {
            "round": [0, 1, 2],
            "attack": ["sign_flip"] * 3,
            "planned_attack_active": [0, 1, 1],
            "fit_selected_active_attackers": [np.nan, 2, 1],
            "fit_selected_benign_clients": [np.nan, 8, 9],
            "fit_malicious_clipped_clients": [np.nan, 1, 1],
            "fit_active_attacker_weight_share": [np.nan, 0.1, 0.0],
            "fit_rtc_v3_cone_overflow_assignments": [np.nan, 1, 0],
            "fit_rtc_v3_semantic_active_w1": [np.nan, 1, 0],
            "fit_rtc_v3_cumulative_active_count": [np.nan, 0, 1],
            "fit_rtc_v3_direction_active_count": [np.nan, 1, 1],
            "fit_rtc_v3_max_constraint_violation": [np.nan, 0.0, 1e-10],
            "fit_rtc_v3_semantic_q_min_round": [np.nan, 0.8, 0.7],
            "fit_rtc_v3_cumulative_q_min": [np.nan, 1.0, 0.9],
            "fit_rtc_v3_direction_q_min": [np.nan, 0.6, 0.5],
            "fit_rtc_v3_principal_budget_active_count": [np.nan, 0, 1],
            "fit_total_defense_seconds": [np.nan, 2.0, 4.0],
        }
    ).to_csv(path, index=False)
    summary = pd.DataFrame(
        [{"run_id": "rtc-run", "attack": "sign_flip", "defense": "rtc_full"}]
    )
    result = byzantine_analysis.rtc_mechanism_summary(summary, {"rtc-run": path})
    row = result.iloc[0]
    assert row["active_round_observations"] == 2
    assert row["clip_recall_active_attackers"] == pytest.approx(2 / 3)
    assert row["semantic_active_round_rate"] == pytest.approx(0.5)
    assert row["direction_active_round_rate"] == pytest.approx(1.0)
    assert row["mean_total_defense_seconds"] == pytest.approx(3.0)
    assert row["semantic_q_min"] == pytest.approx(0.7)
    assert row["principal_budget_active_round_rate"] == pytest.approx(0.5)


def test_aggregate_update_distance_joins_exact_plan_and_round(tmp_path):
    attack_path = tmp_path / "attack.csv"
    clean_path = tmp_path / "clean.csv"
    common = {
        "round": [0, 1, 2],
        "planned_attack_active": [0, 1, 1],
        "fit_aggregate_update_sketch_version": ["v1"] * 3,
        "fit_aggregate_update_sketch_dimension": [2] * 3,
        "fit_aggregate_update_sketch_seed": [7] * 3,
    }
    pd.DataFrame(
        {
            **common,
            "fit_aggregate_update_norm": [0.0, 2.0, 1.0],
            "fit_aggregate_update_sketch_json": ["[0,0]", "[1,0]", "[0,1]"],
        }
    ).to_csv(attack_path, index=False)
    pd.DataFrame(
        {
            **common,
            "fit_aggregate_update_norm": [0.0, 1.0, 1.0],
            "fit_aggregate_update_sketch_json": ["[0,0]", "[1,0]", "[1,0]"],
        }
    ).to_csv(clean_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "run_id": "attack",
                "attack": "sign_flip",
                "defense": "rtc_full",
                "seed": 42,
                "trial_plan_hash": "plan",
            },
            {
                "run_id": "clean",
                "attack": "none",
                "defense": "fedavg",
                "seed": 42,
                "trial_plan_hash": "plan",
            },
        ]
    )
    result = byzantine_analysis.aggregate_update_reference_distances(
        summary, {"attack": attack_path, "clean": clean_path}
    )
    row = result.iloc[0]
    assert row["active_rounds"] == 2
    assert row["mean_cosine_distance_to_clean"] == pytest.approx(0.5)
    assert row["mean_update_norm_ratio_to_clean"] == pytest.approx(1.5)
