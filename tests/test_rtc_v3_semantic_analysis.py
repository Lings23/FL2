from __future__ import annotations

import pandas as pd

from experiments.rtc_v3_semantic_analysis import analyze_run


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
