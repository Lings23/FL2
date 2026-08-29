from experiments.rtc_v3_semantic_heldout_contract import (
    EXPECTED_ATTACKS,
    EXPECTED_SEEDS,
)


def test_extra_heldout_seed_and_attack_sets_are_preregistered() -> None:
    assert EXPECTED_SEEDS == {46, 47, 48, 51}
    assert EXPECTED_ATTACKS == {
        "label_flip_targeted",
        "label_flip_all_reverse",
    }

