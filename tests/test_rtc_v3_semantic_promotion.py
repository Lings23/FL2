import copy

import pytest

from experiments.rtc_v3_semantic_promotion import (
    ENGINEERING_SCOPE,
    SEED42_SCOPE,
    _validate_audit_for_scope,
    _validate_promotion_audit,
)


def _audit() -> dict:
    rows = []
    for seed in (42, 46, 47, 48, 51):
        for attack in ("label_flip_targeted", "label_flip_all_reverse"):
            rows.append({"seed": seed, "attack": attack})
    return {
        "audit_stage": "promotion",
        "passed": True,
        "seed42_passed": True,
        "promotion_evidence_passed": True,
        "promotion_attestation_generated": False,
        "candidate_manifest_hash": "candidate",
        "gates": [{"gate": "all", "passed": True}],
        "attack_comparison": rows,
    }


def test_promotion_validation_requires_seed42_and_four_extra_complete_seeds() -> None:
    seeds, attacks = _validate_promotion_audit(_audit(), "candidate")
    assert seeds == [42, 46, 47, 48, 51]
    assert attacks == ["label_flip_all_reverse", "label_flip_targeted"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"audit_stage": "seed42"}),
        lambda value: value.update({"promotion_evidence_passed": False}),
        lambda value: value["gates"].append({"gate": "failed", "passed": False}),
        lambda value: value["attack_comparison"].pop(),
    ],
)
def test_promotion_validation_fails_closed(mutation) -> None:
    audit = copy.deepcopy(_audit())
    mutation(audit)
    with pytest.raises(ValueError):
        _validate_promotion_audit(audit, "candidate")


def test_seed42_scope_ignores_only_explicit_promotion_only_gates() -> None:
    audit = _audit()
    audit.update(
        {
            "audit_stage": "seed42",
            "promotion_evidence_passed": False,
            "attack_comparison": [
                row for row in audit["attack_comparison"] if row["seed"] == 42
            ],
            "gates": [
                {"gate": "seed42-safety", "passed": True, "promotion_only": False},
                {"gate": "extra-seeds", "passed": False, "promotion_only": True},
            ],
        }
    )
    seeds, attacks = _validate_audit_for_scope(audit, "candidate", SEED42_SCOPE)
    assert seeds == [42]
    assert attacks == ["label_flip_all_reverse", "label_flip_targeted"]


def test_seed42_scope_fails_on_any_seed42_safety_gate() -> None:
    audit = _audit()
    audit.update(
        {
            "audit_stage": "seed42",
            "promotion_evidence_passed": False,
            "attack_comparison": [
                row for row in audit["attack_comparison"] if row["seed"] == 42
            ],
            "gates": [
                {"gate": "seed42-safety", "passed": False, "promotion_only": False},
                {"gate": "extra-seeds", "passed": False, "promotion_only": True},
            ],
        }
    )
    with pytest.raises(ValueError):
        _validate_audit_for_scope(audit, "candidate", SEED42_SCOPE)


def test_engineering_scope_accepts_only_authorized_advisories() -> None:
    audit = _audit()
    audit.update(
        {
            "audit_stage": "seed42",
            "passed": False,
            "seed42_passed": False,
            "promotion_evidence_passed": False,
            "attack_comparison": [
                row for row in audit["attack_comparison"] if row["seed"] == 42
            ],
        }
    )
    required = {
        "clean_final_accuracy_vs_v2",
        "clean_final_accuracy_vs_fedavg",
        "clean_overflow",
        "clean_false_persistence_observation_rate",
        "clean_false_persistence_max_streak",
        "clean_budget_violation",
        "clean_semantic_budget_activation",
        "runtime_cone_set_frozen",
        "controlled_recovery_risk_declines",
        "risk_release_half_life_observed",
        "all_attacking_principals_reached_half_life",
        "attack_budget_violation",
    }
    audit["gates"] = [
        {"gate": name, "passed": True, "promotion_only": False}
        for name in required
    ] + [
        {
            "gate": "clean_benign_clipping",
            "passed": False,
            "promotion_only": False,
            "observed": 0.06166666666666667,
        },
        {
            "gate": "time_to_recovery_observed",
            "passed": False,
            "promotion_only": False,
            "observed": float("nan"),
        },
        {
            "gate": "all_attacking_principals_recovered",
            "passed": False,
            "promotion_only": False,
            "observed": {"attacking": 4, "recovered": 2},
        },
    ]
    seeds, attacks = _validate_audit_for_scope(audit, "candidate", ENGINEERING_SCOPE)
    assert seeds == [42]
    assert attacks == ["label_flip_all_reverse", "label_flip_targeted"]
