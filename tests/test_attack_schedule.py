from experiments.attack_schedule import (
    SPARSE_RANDOM_SCHEDULE,
    attack_schedule_active,
    build_attack_schedule,
)
from experiments.periodic_attack import attack_active


def test_existing_periodic_schedules_are_bitwise_unchanged() -> None:
    for schedule, on_rounds, off_rounds in (
        ("continuous_1_0", 1, 0),
        ("short_1_1", 1, 1),
        ("long_3_3", 3, 3),
    ):
        observed = build_attack_schedule(
            schedule=schedule,
            rounds=70,
            start_round=11,
            end_round=40,
            base_seed=42,
            attack="label_flip_targeted",
            malicious_fraction=0.2,
        )
        expected = tuple(
            attack_active(round_number, 11, on_rounds, off_rounds, 40)
            for round_number in range(1, 71)
        )
        assert observed == expected


def test_sparse_schedule_is_deterministic_defense_independent_and_bounded() -> None:
    kwargs = {
        "schedule": SPARSE_RANDOM_SCHEDULE,
        "rounds": 60,
        "start_round": 11,
        "end_round": -1,
        "base_seed": 42,
        "attack": "label_flip_targeted",
        "malicious_fraction": 0.2,
    }
    first = build_attack_schedule(**kwargs)
    second = build_attack_schedule(**kwargs)
    assert first == second
    assert not any(first[:10])
    active = sum(first[10:])
    assert 1 <= active < 50
    # No defense parameter exists in the schedule contract.
    assert attack_schedule_active(
        schedule=SPARSE_RANDOM_SCHEDULE,
        round_number=17,
        start_round=11,
        end_round=-1,
        base_seed=42,
        attack="label_flip_targeted",
        malicious_fraction=0.2,
    ) == first[16]


def test_sparse_schedule_changes_with_seed_and_attack_condition() -> None:
    common = {
        "schedule": SPARSE_RANDOM_SCHEDULE,
        "rounds": 60,
        "start_round": 11,
        "end_round": -1,
        "malicious_fraction": 0.2,
    }
    seed42 = build_attack_schedule(
        **common, base_seed=42, attack="label_flip_targeted"
    )
    seed43 = build_attack_schedule(
        **common, base_seed=43, attack="label_flip_targeted"
    )
    reverse = build_attack_schedule(
        **common, base_seed=42, attack="label_flip_all_reverse"
    )
    assert seed42 != seed43
    assert seed42 != reverse

