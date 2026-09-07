import pandas as pd
import pytest

from analysis.rtc_clip_mad_b5_225.analyze_results import _training_rounds


def test_b5_runtime_validation_excludes_evaluation_only_round_zero():
    frame = pd.DataFrame(
        {
            "round": list(range(61)),
            "fit_rtc_v3_norm_clip_mad_k": [None] + [2.25] * 60,
        }
    )

    training = _training_rounds(frame, "lie")

    assert len(training) == 60
    assert training["round"].min() == 1
    assert training["fit_rtc_v3_norm_clip_mad_k"].isna().sum() == 0


def test_b5_runtime_validation_rejects_missing_training_round():
    frame = pd.DataFrame({"round": list(range(60))})

    with pytest.raises(RuntimeError, match="expected 60 training rounds, got 59"):
        _training_rounds(frame, "dba")
