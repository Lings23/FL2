import pandas as pd
import pytest

from experiments.rtc_v3_semantic_performance import audit_performance_frames


def _frames(candidate_times=(1.02, 1.02, 1.02), semantic=(0.01, 0.01, 0.01)):
    baseline = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "fit_trial_plan_hash": ["plan"] * 3,
            "fit_total_defense_seconds": [1.0, 1.0, 1.0],
            "fit_solver_seconds": [0.01, 0.01, 0.01],
        }
    )
    candidate = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "fit_trial_plan_hash": ["plan"] * 3,
            "fit_total_defense_seconds": candidate_times,
            "fit_semantic_total_seconds": semantic,
            "fit_solver_seconds": [0.01, 0.01, 0.01],
            "fit_rtc_v3_cone_update_applied_count": [0.0, 0.0, 0.0],
        }
    )
    return baseline, candidate


def test_paired_performance_gates_pass_and_report_exact_pairing():
    gates, paired = audit_performance_frames(*_frames())
    assert gates["passed"] is True
    assert gates["paired_rounds"] == 3
    assert len(paired) == 3


def test_performance_gate_rejects_overhead_and_online_cone_update():
    baseline, candidate = _frames(candidate_times=(1.2, 1.2, 1.2))
    candidate.loc[1, "fit_rtc_v3_cone_update_applied_count"] = 1.0
    gates, _ = audit_performance_frames(baseline, candidate)
    assert gates["passed"] is False
    assert gates["defense_mean_increase_gate"] is False
    assert gates["online_cones_frozen_gate"] is False


def test_performance_audit_rejects_unpaired_rounds():
    baseline, candidate = _frames()
    candidate.loc[2, "fit_trial_plan_hash"] = "other"
    with pytest.raises(ValueError, match="not exactly paired"):
        audit_performance_frames(baseline, candidate)
