from experiments.rtc_v3_semantic_audit import (
    CLEAN_BENIGN_CLIPPING_MAX,
    REQUIRED_EXTRA_HELDOUT_SEEDS,
    _as_paths,
)


def test_promotion_audit_uses_preregistered_clean_and_seed_gates() -> None:
    assert CLEAN_BENIGN_CLIPPING_MAX == 0.03
    assert REQUIRED_EXTRA_HELDOUT_SEEDS == 4


def test_multiple_heldout_roots_must_be_unique(tmp_path) -> None:
    assert _as_paths([tmp_path / "a", tmp_path / "b"])[0].name == "a"
    try:
        _as_paths([tmp_path / "a", tmp_path / "a"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate held-out roots must fail")
