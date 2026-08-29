from experiments.rtc_v3_semantic_expansion_registry import (
    build_registry,
    validate_registry,
)


def test_expansion_registry_covers_preregistered_dimensions_without_cartesian_grid() -> None:
    payload = build_registry()
    assert validate_registry(payload)["passed"] is True
    coverage = payload["coverage"]
    assert coverage["poison_fractions"] == [0.25, 0.5, 1.0]
    assert coverage["malicious_fractions"] == [0.1, 0.2, 0.3]
    assert "dirichlet:0.1" in coverage["partitions"]
    assert "sparse_random_0.25" in coverage["periods"]
    assert len(payload["conditions"]) == 13


def test_adaptive_condition_is_explicitly_pending_until_implemented() -> None:
    payload = build_registry()
    adaptive = [item for item in payload["conditions"] if item["family"] == "adaptive"]
    assert len(adaptive) == 1
    assert adaptive[0]["status"] == "implementation_pending"
    recovery = [
        item for item in payload["conditions"]
        if item["id"] == "temporal_attack_stop_recovery"
    ][0]
    assert recovery["candidate_defenses"] == ["rtc_v3"]
    assert recovery["baseline_defenses"] == []
