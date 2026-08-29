"""Preregister the staged RTC-V3 semantic expansion experiment conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


REGISTRY_VERSION = "rtc_v3.semantic_expansions.v1"


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def build_registry() -> dict[str, Any]:
    common = {
        "seed": 42,
        "attack": "label_flip_targeted",
        "source_label": 5,
        "target_label": 3,
        "poison_fraction": 1.0,
        "malicious_fraction": 0.2,
        "period": "continuous_1_0",
        "attack_start_round": 11,
        "attack_end_round": -1,
        "partition": "iid",
        "dirichlet_alpha": None,
        "rounds": 60,
    }
    conditions: list[dict[str, Any]] = []

    def add(identifier: str, family: str, **changes: Any) -> None:
        condition = {
            **common,
            "id": identifier,
            "family": family,
            "candidate_defenses": ["clip_only", "rtc_v3"],
            "baseline_defenses": ["rtc_v3"],
            "candidate_manifest": "config/rtc_v3_manifest_formal_iid_semantic_candidate.json",
            "baseline_manifest": "config/rtc_v3_manifest_formal_iid_cone_v2_candidate.json",
            "pairing_mode": "strict",
            "sampling_protocol": "principal_uniform",
            "status": "preregistered_not_executed",
            **changes,
        }
        conditions.append(condition)

    # Temporal schedules change only the attack on/off process.
    add("temporal_short_1_1", "temporal", period="short_1_1")
    add("temporal_long_3_3", "temporal", period="long_3_3")
    add("temporal_sparse_random_025", "temporal", period="sparse_random_0.25")
    # Recovery is already separately preregistered as attack rounds 11..40,
    # train through round 70; list it here so expansion completeness can bind it.
    add(
        "temporal_attack_stop_recovery",
        "temporal",
        attack_end_round=40,
        rounds=70,
        candidate_defenses=["rtc_v3"],
        baseline_defenses=[],
    )
    # Multiple unknown source/target pairs.  Core 5->3 remains the reference.
    add("pair_0_to_1", "source_target", source_label=0, target_label=1)
    add("pair_8_to_2", "source_target", source_label=8, target_label=2)
    # Poison intensity; 1.0 is provided by the core run.
    add("poison_025", "poison_fraction", poison_fraction=0.25)
    add("poison_050", "poison_fraction", poison_fraction=0.5)
    # Malicious population; 0.2 is provided by the core run.
    add("malicious_010", "malicious_fraction", malicious_fraction=0.1)
    add("malicious_030", "malicious_fraction", malicious_fraction=0.3)
    # IID is provided by the core run.
    add(
        "dirichlet_050",
        "partition",
        partition="dirichlet",
        dirichlet_alpha=0.5,
    )
    add(
        "dirichlet_010",
        "partition",
        partition="dirichlet",
        dirichlet_alpha=0.1,
    )
    # Cross-round-consistent adaptive pressure requires a dedicated attack
    # implementation and therefore cannot be silently approximated by a
    # periodic label flip.
    add(
        "adaptive_consistent_pressure",
        "adaptive",
        attack="label_flip_targeted_adaptive_consistent",
        status="implementation_pending",
    )
    payload: dict[str, Any] = {
        "schema_version": REGISTRY_VERSION,
        "selection_method": "one-factor-at-a-time around seed42 IID continuous 5->3, plus a separately preregistered adaptive pressure test",
        "prerequisites": [
            "seed42_gate_audit passed=true",
            "candidate parameters remain frozen",
            "no held-out feedback is used for tuning",
        ],
        "core_reference": common,
        "conditions": conditions,
        "coverage": {
            "periods": ["continuous_1_0", "short_1_1", "long_3_3", "sparse_random_0.25"],
            "source_target_pairs": [[5, 3], [0, 1], [8, 2]],
            "poison_fractions": [0.25, 0.5, 1.0],
            "malicious_fractions": [0.1, 0.2, 0.3],
            "partitions": ["iid", "dirichlet:0.5", "dirichlet:0.1"],
            "adaptive_tests": ["adaptive_consistent_pressure"],
        },
    }
    payload["registry_hash"] = _digest(payload)
    return payload


def validate_registry(payload: dict[str, Any]) -> dict[str, Any]:
    observed_hash = str(payload.get("registry_hash", ""))
    hash_payload = dict(payload)
    hash_payload.pop("registry_hash", None)
    conditions = payload.get("conditions", [])
    identifiers = [str(item.get("id")) for item in conditions]
    allowed_statuses = {"preregistered_not_executed", "implementation_pending"}
    gates = {
        "content_hash": observed_hash == _digest(hash_payload),
        "unique_condition_ids": len(identifiers) == len(set(identifiers)),
        "strict_pairing": all(
            item.get("pairing_mode") == "strict"
            and item.get("sampling_protocol") == "principal_uniform"
            for item in conditions
        ),
        "status_explicit": all(item.get("status") in allowed_statuses for item in conditions),
        "candidate_and_baseline_registered": all(
            item.get("candidate_manifest") and item.get("baseline_manifest")
            for item in conditions
            if item.get("id") != "temporal_attack_stop_recovery"
        ),
        "adaptive_not_misrepresented_as_executed": any(
            item.get("family") == "adaptive"
            and item.get("status") == "implementation_pending"
            for item in conditions
        ),
    }
    return {"passed": all(gates.values()), "gates": gates}


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="logs/rtc_v3_semantic_expansion_registry.json"
    )
    args = parser.parse_args(argv)
    payload = build_registry()
    validation = validate_registry(payload)
    payload["validation"] = validation
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError("expansion registry validation failed")
    return destination


if __name__ == "__main__":
    main()
