from __future__ import annotations

import pandas as pd
import pytest

from experiments.plot_rtc_v3_semantic_results import active_summary


def test_active_summary_excludes_round_zero_and_inactive_rounds() -> None:
    frame = pd.DataFrame(
        {
            "round": [0, 1, 2, 3],
            "planned_attack_active": [0, 0, 1, 1],
            "server_asr": [0.99, 0.80, 0.20, 0.40],
            "server_accuracy": [0.10, 0.50, 0.60, 0.70],
        }
    )
    result = active_summary(frame)
    assert result["active_mean_asr"] == pytest.approx(0.30)
    assert result["active_peak_asr"] == pytest.approx(0.40)
    assert result["final_accuracy"] == pytest.approx(0.70)
    assert result["last10_accuracy"] == pytest.approx(0.60)


def test_active_summary_rejects_missing_active_window() -> None:
    frame = pd.DataFrame(
        {
            "round": [0, 1],
            "planned_attack_active": [0, 0],
            "server_asr": [0.1, 0.2],
            "server_accuracy": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="no active attack"):
        active_summary(frame)
