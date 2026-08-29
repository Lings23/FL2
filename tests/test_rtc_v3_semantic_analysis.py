from __future__ import annotations

import pandas as pd

from experiments.rtc_v3_semantic_analysis import (
    analyze_run,
    clean_false_persistence_metrics,
    principal_recovery_metrics,
    replay_semantic_budget_triggers,
)
from defenses.rtc.calibration import CalibrationManifest


def test_detection_metrics_include_benign_clients_from_attack_active_rounds() -> None:
    spec = {
        "trial_plan_hash": "plan",
        "seed": 43,
        "attack": "label_flip_targeted",
        "period": "continuous_1_0",
        "attack_start_round": 2,
        "attack_end_round": -1,
        "defense": "rtc_v3",
    }
    rounds = pd.DataFrame(
        {
            "round": [1, 2],
            "planned_attack_active": [0, 1],
            "server_accuracy": [0.5, 0.6],
            "server_asr": [0.1, 0.2],
            "fit_selected_active_attackers": [0, 1],
        }
    )
    clients = pd.DataFrame(
        {
            "round": [1, 1, 2, 2],
            "is_malicious": [True, False, True, False],
            "attack_active": [False, False, True, False],
            "clipped": [False, False, True, False],
            "semantic_risk": [0.0, 0.0, 0.9, 0.1],
        }
    )

    result = analyze_run(spec, rounds, clients)

    assert result["clip_recall"] == 1.0
    assert result["clip_fpr"] == 0.0
    assert result["semantic_detection_auc"] == 1.0
    assert result["asr_auc_raw"] == 0.2
    assert result["asr_auc_normalized"] == 0.2


def test_asr_auc_reports_raw_and_normalized_forms() -> None:
    spec = {
        "trial_plan_hash": "plan",
        "seed": 43,
        "attack": "label_flip_targeted",
        "period": "continuous_1_0",
        "attack_start_round": 1,
        "attack_end_round": -1,
        "defense": "rtc_v3",
    }
    rounds = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "planned_attack_active": [1, 1, 1],
            "server_accuracy": [0.5, 0.6, 0.7],
            "server_asr": [0.1, 0.2, 0.3],
            "fit_selected_active_attackers": [1, 1, 1],
        }
    )
    clients = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "is_malicious": [True, True, True],
            "attack_active": [True, True, True],
            "clipped": [False, False, False],
        }
    )
    result = analyze_run(spec, rounds, clients)
    assert result["asr_auc"] == result["asr_auc_raw"] == 0.4
    assert result["asr_auc_normalized"] == 0.2


def test_semantic_trigger_count_uses_only_hard_budget_activity() -> None:
    spec = {
        "trial_plan_hash": "plan",
        "seed": 43,
        "attack": "label_flip_targeted",
        "period": "continuous_1_0",
        "attack_start_round": 2,
        "attack_end_round": -1,
        "defense": "rtc_v3",
    }
    rounds = pd.DataFrame(
        {
            "round": [1, 2],
            "planned_attack_active": [0, 1],
            "server_accuracy": [0.5, 0.6],
            "server_asr": [0.1, 0.2],
            "fit_selected_active_attackers": [0, 1],
            "fit_rtc_v3_semantic_active_pair_count": [0.0, 7.0],
            "fit_rtc_v3_semantic_active_w1": [0.0, 1.0],
            "fit_rtc_v3_semantic_active_w4": [0.0, 0.0],
            "fit_rtc_v3_principal_budget_active_count": [0.0, 2.0],
            "fit_rtc_v3_full_residual_active_w1": [0.0, 1.0],
            "fit_solver_seconds": [0.01, 0.03],
        }
    )
    clients = pd.DataFrame(
        {
            "round": [2, 2],
            "is_malicious": [True, False],
            "attack_active": [True, False],
            "clipped": [True, False],
            "semantic_risk": [0.9, 0.1],
        }
    )

    result = analyze_run(spec, rounds, clients)

    assert result["semantic_trigger_count"] == 1
    assert result["principal_trigger_count"] == 2
    assert result["server_trigger_count"] == 1
    assert result["solver_seconds_mean"] == 0.02


def test_validation_only_clean_reports_false_persistence_by_schedule() -> None:
    spec = {
        "trial_plan_hash": "clean-plan",
        "seed": 42,
        "attack": "label_flip_targeted",
        "period": "continuous_1_0",
        "attack_start_round": 61,
        "attack_end_round": -1,
        "defense": "rtc_v3",
    }
    rounds = pd.DataFrame(
        {
            "round": [1, 2],
            "planned_attack_active": [0, 0],
            "server_accuracy": [0.5, 0.6],
            "server_asr": [0.1, 0.2],
            "fit_selected_active_attackers": [0, 0],
            "fit_rtc_v3_semantic_risk_mean": [0.0, 0.2],
        }
    )
    clients = pd.DataFrame(
        {
            "round": [1, 2],
            "is_malicious": [False, False],
            "attack_active": [False, False],
            "clipped": [False, False],
            "semantic_risk": [0.0, 0.2],
            "semantic_q": [1.0, 0.8],
            "principal_id": ["a", "a"],
        }
    )

    result = analyze_run(spec, rounds, clients)

    assert result["false_persistence"] == 0.5


def test_clean_false_persistence_uses_observations_and_watch_threshold() -> None:
    clients = pd.DataFrame(
        {
            "round": [1, 4, 7, 2, 8],
            "principal_id": ["a", "a", "a", "b", "b"],
            "is_malicious": [False] * 5,
            "semantic_risk": [0.01, 0.2, 0.3, 0.09, 0.0],
            "semantic_q": [0.99, 0.8, 0.7, 0.91, 1.0],
        }
    )
    result = clean_false_persistence_metrics(
        clients=clients, watch_threshold=0.1
    )
    assert result["false_persistence"] == 0.4
    assert result["false_soft_downweight_rate"] == 0.8
    assert result["false_persistent_principal_rate"] == 0.5
    assert result["false_persistence_max_observation_streak"] == 2


def test_semantic_budget_replay_separates_global_group_and_principal() -> None:
    manifest = CalibrationManifest(
        payload={
            "semantic_temporal_exposure": {
                "num_classes": 2,
                "top_k_pairs": 1,
                "hard_exposure_risk_floor": 0.0,
                "global_windows": [1],
                "global_budgets": {"1": 0.4},
                "pair_windows": [1],
                "pair_budgets": {"1": 0.4},
                "principal_windows": [1],
                "principal_budgets": {"1": 0.4},
                "phases": {"warmup": {"head_exposure_scale": 1.0}},
            }
        }
    )
    rounds = pd.DataFrame(
        {"round": [1], "fit_rtc_v3_cone_profile": ["warmup"]}
    )
    clients = pd.DataFrame(
        {
            "round": [1, 1],
            "principal_id": ["a", "b"],
            "nominal_mass": [0.5, 0.5],
            "aggregation_weight": [0.5, 0.5],
            "rtc_v3_semantic_head_norm": [1.0, 1.0],
            "semantic_risk": [1.0, 1.0],
            "semantic_top_source": [0, 1],
            "semantic_top_target": [1, 0],
        }
    )

    replay = replay_semantic_budget_triggers(
        rounds=rounds, clients=clients, manifest=manifest
    )

    assert replay == {"global": 1, "group": 2, "principal": 2}


def test_recovery_requires_actual_low_risk_observations_for_every_attacker() -> None:
    rounds = pd.DataFrame(
        {"round": list(range(1, 9)), "planned_attack_active": [1, 1, 1, 1, 0, 0, 0, 0]}
    )
    clients = pd.DataFrame(
        {
            "round": [4, 4, 5, 6, 7, 5, 8],
            "principal_id": ["a", "b", "a", "a", "a", "b", "b"],
            "is_malicious": [True] * 7,
            "attack_active": [True, True, False, False, False, False, False],
            "semantic_risk": [0.8, 0.6, 0.04, 0.03, 0.02, 0.01, 0.01],
        }
    )
    result = principal_recovery_metrics(
        rounds=rounds, clients=clients, recovery_observations=3
    )
    assert result["recovered_principal_count"] == 1
    assert pd.isna(result["time_to_recovery"])
    # Principal b's absence in rounds 6/7 does not count as recovery evidence.


def test_recovery_delay_is_slowest_principal_and_half_life_uses_pre_stop_risk() -> None:
    rounds = pd.DataFrame(
        {"round": list(range(1, 9)), "planned_attack_active": [1, 1, 1, 1, 0, 0, 0, 0]}
    )
    clients = pd.DataFrame(
        {
            "round": [4, 4, 5, 6, 7, 5, 6, 8],
            "principal_id": ["a", "b", "a", "a", "a", "b", "b", "b"],
            "is_malicious": [True] * 8,
            "attack_active": [True, True, False, False, False, False, False, False],
            "semantic_risk": [0.8, 0.6, 0.3, 0.04, 0.03, 0.2, 0.04, 0.03],
        }
    )
    result = principal_recovery_metrics(
        rounds=rounds, clients=clients, recovery_observations=2
    )
    assert result["time_to_recovery"] == 4
    assert result["risk_release_half_life"] == 1
    assert result["recovered_principal_count"] == 2
