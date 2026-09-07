import pandas as pd
import pytest

from analysis.rtc_v1_multiseed.analyze_results import (
    _client_metrics,
    _first_client_divergence,
    _paired_stats,
)


def test_v1_paired_stats_reports_three_seed_mean_and_interval():
    result = _paired_stats(pd.Series([0.01, 0.02, 0.00]))

    assert result["n"] == 3
    assert result["degrees_of_freedom"] == 2
    assert result["inference_scope"] == "exploratory_three_seed_paired"
    assert result["equivalence_test_performed"] is False
    assert result["mean_delta"] == pytest.approx(0.01)
    assert result["ci95_low"] < result["mean_delta"] < result["ci95_high"]
    assert 0.0 <= result["paired_p_two_sided"] <= 1.0


def test_v1_paired_stats_requires_all_three_seeds():
    with pytest.raises(RuntimeError, match="expected 3 paired seeds"):
        _paired_stats(pd.Series([0.01, 0.02]))


def test_v1_client_metrics_count_clean_behavior_as_nonattackers():
    frame = pd.DataFrame(
        [
            {
                "round": 1,
                "cid": str(cid),
                "is_malicious": cid == 0,
                "attack_active": False,
                "clipped": cid == 0,
                "state": "watch" if cid == 0 else "trusted",
            }
            for cid in range(10)
        ]
    )

    metrics = _client_metrics(frame, {1})

    assert metrics["benign_clipping_rate"] == 0.0
    assert metrics["nonattacker_clipping_rate"] == pytest.approx(0.1)
    assert metrics["nonattacker_watch_rate"] == pytest.approx(0.1)
    assert pd.isna(metrics["active_attacker_clipping_rate"])


def test_v1_first_client_divergence_reports_any_and_metric_window():
    rows = []
    for round_number in range(1, 12):
        for cid in range(10):
            rows.append(
                {
                    "round": round_number,
                    "cid": str(cid),
                    "is_malicious": cid < 3,
                    "attack_active": round_number >= 11 and cid < 3,
                    "aggregation_weight": 0.1,
                    "effective_weight": 0.1,
                    "clipped_delta_norm": 1.0,
                    "clipped": False,
                    "quarantined": False,
                    "state": "trusted",
                }
            )
    multi_krum = pd.DataFrame(rows)
    rtc = multi_krum.copy()
    rtc.loc[(rtc["round"] == 1) & (rtc["cid"] == "0"), "aggregation_weight"] = 0.09
    rtc.loc[(rtc["round"] == 11) & (rtc["cid"] == "1"), "state"] = "restricted"

    result = _first_client_divergence(rtc, multi_krum, condition="lie_z05", seed=42)

    assert result["first_any_round"] == 1
    assert result["first_any_cid"] == "0"
    assert result["first_metric_window_round"] == 11
    assert result["first_metric_window_cid"] == "1"
