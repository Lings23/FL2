import json

from experiments.rtc_v3_semantic_smoke_contract import validate_smoke_contract


def test_smoke_contract_requires_six_defenses_one_plan_and_candidate(tmp_path) -> None:
    root = tmp_path / "smoke"
    plans = root / "trial_plans"
    plans.mkdir(parents=True)
    plan = plans / "plan.json"
    plan.write_text(json.dumps({"rounds": [{}, {}, {}]}), encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    defenses = [
        "fedavg", "rtc_v3", "semantic_observe", "semantic_soft",
        "semantic_temporal", "semantic_exposure",
    ]
    specs = []
    for defense in defenses:
        specs.append(
            {
                "trial_plan_hash": "hash",
                "trial_plan_path": str(plan),
                "defense": defense,
                "seed": 52,
                "attack": "label_flip_targeted",
                "pairing_mode": "strict",
                "sampling_protocol": "principal_uniform",
                "custom_params": (
                    {} if defense == "fedavg" else {"calibration_path": str(candidate.resolve())}
                ),
            }
        )
    (root / "experiment_manifest.json").write_text(
        json.dumps({"specs": specs}), encoding="utf-8"
    )
    evidence = validate_smoke_contract(root, candidate_manifest=candidate)
    assert evidence["passed"] is True
    assert evidence["execution_complete"] is False

