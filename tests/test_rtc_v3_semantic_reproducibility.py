import pandas as pd

from experiments.rtc_v3_semantic_reproducibility import _compare_frame


def test_reproducibility_frame_requires_exact_seeds_and_bounded_floats() -> None:
    left = pd.DataFrame({"round": [2, 1], "fit_seed": [22, 11], "value": [0.2, 0.1]})
    right = pd.DataFrame({"round": [1, 2], "fit_seed": [11, 22], "value": [0.1 + 1e-9, 0.2]})
    result = _compare_frame(
        left, right, sort_keys=["round"], exact_columns={"round", "fit_seed"},
        float_columns={"value"}, atol=1e-7, rtol=1e-6,
    )
    assert result["passed"] is True
    right.loc[right["round"] == 2, "fit_seed"] = 99
    result = _compare_frame(
        left, right, sort_keys=["round"], exact_columns={"round", "fit_seed"},
        float_columns={"value"}, atol=1e-7, rtol=1e-6,
    )
    assert result["passed"] is False
    assert result["exact_mismatches"] == ["fit_seed"]

