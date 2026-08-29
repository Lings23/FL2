import pandas as pd
import pytest

from experiments.rtc_v3_semantic_provisional import build_provisional_snapshot


def test_provisional_snapshot_is_never_formal_and_requires_matching_keys(
    monkeypatch, tmp_path
) -> None:
    candidate = pd.DataFrame(
        [
            {
                "trial_plan_hash": "plan",
                "seed": 42,
                "attack": "label_flip_targeted",
                "defense": "rtc_v3",
                "active_mean_asr": 0.1,
                "peak_asr": 0.2,
                "final_accuracy": 0.9,
                "asr_k3_worst": 0.2,
                "asr_k4_worst": 0.15,
                "malicious_effective_weight": 0.01,
            },
            {
                "trial_plan_hash": "plan",
                "seed": 42,
                "attack": "label_flip_targeted",
                "defense": "clip_only",
                "active_mean_asr": 0.4,
                "peak_asr": 0.7,
                "final_accuracy": 0.85,
                "asr_k3_worst": 0.6,
                "asr_k4_worst": 0.7,
                "malicious_effective_weight": 0.2,
            },
        ]
    )
    baseline = candidate.iloc[[1]].assign(defense="rtc_v3")
    monkeypatch.setattr(
        "experiments.rtc_v3_semantic_provisional._completed_rows",
        lambda root, attack: (candidate, []),
    )
    monkeypatch.setattr(
        "experiments.rtc_v3_semantic_provisional.analyze_root", lambda root: baseline
    )
    monkeypatch.setattr(
        "experiments.rtc_v3_semantic_provisional.strict_root_is_valid",
        lambda root: True,
    )
    payload = build_provisional_snapshot(
        attack_root=tmp_path / "attack",
        baseline_root=tmp_path / "baseline",
        attack="label_flip_targeted",
    )
    assert payload["formal_statistics_eligible"] is False
    assert payload["attack_root_strict_validation_present_and_passed"] is True
    assert payload["candidate_delta_vs_promoted_v2"]["active_mean_asr"] == pytest.approx(-0.3)

