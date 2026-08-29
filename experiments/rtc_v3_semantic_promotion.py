"""Generate RTC-V3 semantic promotion evidence only after a full audit.

The utility has no path that mutates a calibration manifest, changes rtc_full,
or overwrites an existing evidence file.  Readiness and the formal attestation
are separate artifacts; the latter requires a hash-bound readiness artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.rtc.calibration import CalibrationManifest, content_hash


READINESS_SCHEMA = "rtc_v3.semantic.promotion_readiness.v1"
ATTESTATION_SCHEMA = "rtc_v3.semantic.promotion_attestation.v1"
SEED42_SCOPE = "seed42_descriptive_iid_continuous_label_flip"
MULTI_SEED_SCOPE = "multi_seed_preregistered"
ENGINEERING_SCOPE = "seed42_engineering_iid_continuous_label_flip"
ENGINEERING_ADVISORY_GATES = {
    "clean_benign_clipping": (
        "Observed clean clipping is advisory because clean utility, overflow, "
        "false persistence, false soft-downweighting, budgets, and strict pairing passed."
    ),
    "time_to_recovery_observed": (
        "The 70-round horizon right-censored full re-entry for two principals; "
        "aggregate risk declined to zero and every attacking principal reached half-life."
    ),
    "all_attacking_principals_recovered": (
        "Two principals had only one or two post-release low-risk observations by round 70; "
        "absence is not counted as recovery and the exception remains explicit."
    ),
}


def _json_safe(value: Any) -> Any:
    """Normalize audit evidence for canonical, standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {resolved}")
    return resolved, payload


def _validate_promotion_audit(
    audit: Mapping[str, Any], candidate_hash: str
) -> tuple[list[int], list[str]]:
    if audit.get("audit_stage") != "promotion":
        raise ValueError("formal promotion evidence requires audit_stage=promotion")
    if not bool(audit.get("passed")):
        raise ValueError("promotion audit passed=false")
    if not bool(audit.get("seed42_passed")):
        raise ValueError("seed42 promotion prerequisite failed")
    if not bool(audit.get("promotion_evidence_passed")):
        raise ValueError("multi-seed promotion evidence failed")
    if bool(audit.get("promotion_attestation_generated")):
        raise ValueError("input audit must be pre-attestation evidence")
    if str(audit.get("candidate_manifest_hash")) != str(candidate_hash):
        raise ValueError("promotion audit candidate hash mismatch")
    gates = audit.get("gates", [])
    if not gates or any(not bool(item.get("passed")) for item in gates):
        raise ValueError("every seed42 and promotion-only gate must pass")
    attacks_by_seed: dict[int, set[str]] = {}
    for row in audit.get("attack_comparison", []):
        attacks_by_seed.setdefault(int(row["seed"]), set()).add(str(row["attack"]))
    expected_attacks = {"label_flip_targeted", "label_flip_all_reverse"}
    if attacks_by_seed.get(42) != expected_attacks:
        raise ValueError("seed42 must contain both preregistered attacks")
    extra = sorted(
        seed
        for seed, attacks in attacks_by_seed.items()
        if seed != 42 and attacks == expected_attacks
    )
    if len(extra) < 4:
        raise ValueError("at least four complete extra held-out seeds are required")
    return [42, *extra], sorted(expected_attacks)


def _validate_seed42_audit(
    audit: Mapping[str, Any], candidate_hash: str
) -> tuple[list[int], list[str]]:
    """Validate the explicitly limited seed-42 safety-closure contract.

    Promotion-only gates (first-layer ablations and extra held-out seeds) are
    deliberately outside this scope.  They remain mandatory for the original
    multi-seed promotion path above.
    """
    if audit.get("audit_stage") != "seed42":
        raise ValueError("seed42-scoped evidence requires audit_stage=seed42")
    if not bool(audit.get("passed")) or not bool(audit.get("seed42_passed")):
        raise ValueError("seed42 safety audit failed")
    if bool(audit.get("promotion_attestation_generated")):
        raise ValueError("input audit must be pre-attestation evidence")
    if str(audit.get("candidate_manifest_hash")) != str(candidate_hash):
        raise ValueError("seed42 audit candidate hash mismatch")
    gates = audit.get("gates", [])
    required_gates = [item for item in gates if not bool(item.get("promotion_only"))]
    if not required_gates or any(not bool(item.get("passed")) for item in required_gates):
        raise ValueError("every seed42 safety gate must pass")
    attacks = {
        str(row["attack"])
        for row in audit.get("attack_comparison", [])
        if int(row["seed"]) == 42
    }
    expected_attacks = {"label_flip_targeted", "label_flip_all_reverse"}
    if attacks != expected_attacks:
        raise ValueError("seed42 must contain both preregistered attacks")
    return [42], sorted(expected_attacks)


def _validate_engineering_audit(
    audit: Mapping[str, Any], candidate_hash: str
) -> tuple[list[int], list[str]]:
    """Validate the user-authorized, explicitly scoped engineering promotion.

    This path never converts the original preregistered audit to a pass.  It
    accepts only the named advisory exceptions and requires every other
    seed-42 safety gate to pass.
    """
    if audit.get("audit_stage") != "seed42":
        raise ValueError("engineering evidence requires audit_stage=seed42")
    if bool(audit.get("promotion_attestation_generated")):
        raise ValueError("input audit must be pre-attestation evidence")
    if str(audit.get("candidate_manifest_hash")) != str(candidate_hash):
        raise ValueError("engineering audit candidate hash mismatch")
    gates = [
        item for item in audit.get("gates", []) if not bool(item.get("promotion_only"))
    ]
    if not gates:
        raise ValueError("engineering audit has no seed42 safety gates")
    failed = {str(item.get("gate")): item for item in gates if not bool(item.get("passed"))}
    if set(failed) != set(ENGINEERING_ADVISORY_GATES):
        raise ValueError(
            "engineering audit failures differ from the authorized advisory set: "
            f"{sorted(failed)}"
        )
    clipping = float(failed["clean_benign_clipping"].get("observed"))
    if not 0.03 < clipping <= 0.07:
        raise ValueError("engineering clean clipping exception is outside (0.03, 0.07]")
    recovery = failed["all_attacking_principals_recovered"].get("observed", {})
    if not isinstance(recovery, Mapping):
        raise ValueError("engineering recovery exception has invalid evidence")
    attacking = int(recovery.get("attacking", 0))
    recovered = int(recovery.get("recovered", 0))
    if attacking <= 0 or recovered < 2 or recovered >= attacking:
        raise ValueError("engineering recovery exception is outside its authorized evidence")
    required_passes = {
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
    passed_names = {str(item.get("gate")) for item in gates if bool(item.get("passed"))}
    missing = sorted(required_passes - passed_names)
    if missing:
        raise ValueError(f"engineering promotion is missing required passing gates: {missing}")
    attacks = {
        str(row["attack"])
        for row in audit.get("attack_comparison", [])
        if int(row["seed"]) == 42
    }
    expected_attacks = {"label_flip_targeted", "label_flip_all_reverse"}
    if attacks != expected_attacks:
        raise ValueError("engineering seed42 evidence must contain both attacks")
    return [42], sorted(expected_attacks)


def _validate_audit_for_scope(
    audit: Mapping[str, Any], candidate_hash: str, scope: str
) -> tuple[list[int], list[str]]:
    if scope == MULTI_SEED_SCOPE:
        return _validate_promotion_audit(audit, candidate_hash)
    if scope == SEED42_SCOPE:
        return _validate_seed42_audit(audit, candidate_hash)
    if scope == ENGINEERING_SCOPE:
        return _validate_engineering_audit(audit, candidate_hash)
    raise ValueError(f"unsupported promotion scope: {scope!r}")


def build_readiness(
    *,
    audit_path: str | Path,
    candidate_manifest: str | Path,
    candidate_validation: str | Path,
    candidate_provenance: str | Path,
    scope: str = MULTI_SEED_SCOPE,
) -> dict[str, Any]:
    audit_source, audit = _load_json(audit_path)
    manifest = CalibrationManifest.load(candidate_manifest)
    validation_source, validation = _load_json(candidate_validation)
    provenance_source, provenance = _load_json(candidate_provenance)
    metadata = manifest.payload.get("metadata", {})
    if bool(metadata.get("promotion_ready", False)):
        raise ValueError("candidate must remain promotion_ready=false before evidence")
    if not bool(validation.get("passed")):
        raise ValueError("candidate validation passed=false")
    for label, payload in (("validation", validation), ("provenance", provenance)):
        if str(payload.get("content_hash")) != manifest.hash:
            raise ValueError(f"candidate {label} hash mismatch")
    seeds, attacks = _validate_audit_for_scope(audit, manifest.hash, scope)
    applicable_gates = [
        item
        for item in audit["gates"]
        if scope == MULTI_SEED_SCOPE or not bool(item.get("promotion_only"))
    ]
    gate_names = [str(item["gate"]) for item in applicable_gates]
    return {
        "schema": READINESS_SCHEMA,
        "decision": (
            "promotion_ready"
            if scope == MULTI_SEED_SCOPE
            else (
                "engineering_promotion_ready"
                if scope == ENGINEERING_SCOPE
                else "seed42_scoped_promotion_ready"
            )
        ),
        "scope": scope,
        "advisory_exceptions": (
            [
                {
                    "gate": name,
                    "reason": reason,
                    "observed": _json_safe(next(
                        item.get("observed")
                        for item in audit["gates"]
                        if item.get("gate") == name
                    )),
                    "original_expected": _json_safe(next(
                        item.get("expected")
                        for item in audit["gates"]
                        if item.get("gate") == name
                    )),
                }
                for name, reason in ENGINEERING_ADVISORY_GATES.items()
            ]
            if scope == ENGINEERING_SCOPE
            else []
        ),
        "claim_limitations": (
            []
            if scope == MULTI_SEED_SCOPE
            else [
                "single held-out seed only",
                "IID only",
                "continuous targeted and all-reverse label-flip attacks only",
                "no first-layer causal ablation",
                "no cross-seed statistical significance or generalization claim",
            ]
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_hash": manifest.hash,
        "heldout_seeds": seeds,
        "attacks": attacks,
        "all_gate_count": len(gate_names),
        "all_required_gates_passed": True,
        "all_original_gates_passed": scope != ENGINEERING_SCOPE,
        "promotion_attestation_generated": False,
        "manifest_mutated": False,
        "rtc_full_pointer_mutated": False,
        "sources": {
            "promotion_audit": _source(audit_source),
            "candidate_manifest": _source(candidate_manifest),
            "candidate_validation": _source(validation_source),
            "candidate_provenance": _source(provenance_source),
        },
    }


def build_attestation(
    *,
    readiness_path: str | Path,
    audit_path: str | Path,
    candidate_manifest: str | Path,
) -> dict[str, Any]:
    readiness_source, readiness = _load_json(readiness_path)
    audit_source, audit = _load_json(audit_path)
    manifest = CalibrationManifest.load(candidate_manifest)
    if readiness.get("schema") != READINESS_SCHEMA:
        raise ValueError("invalid promotion readiness schema")
    scope = str(readiness.get("scope", MULTI_SEED_SCOPE))
    expected_decision = {
        MULTI_SEED_SCOPE: "promotion_ready",
        SEED42_SCOPE: "seed42_scoped_promotion_ready",
        ENGINEERING_SCOPE: "engineering_promotion_ready",
    }.get(scope)
    if expected_decision is None:
        raise ValueError(f"unsupported readiness scope: {scope!r}")
    if readiness.get("decision") != expected_decision:
        raise ValueError("readiness decision does not match its promotion scope")
    if bool(readiness.get("promotion_attestation_generated")):
        raise ValueError("readiness artifact already claims attestation generation")
    if str(readiness.get("candidate_hash")) != manifest.hash:
        raise ValueError("readiness candidate hash mismatch")
    seeds, attacks = _validate_audit_for_scope(audit, manifest.hash, scope)
    if list(readiness.get("heldout_seeds", [])) != seeds:
        raise ValueError("readiness held-out seed set differs from final audit")
    return {
        "schema": ATTESTATION_SCHEMA,
        "decision": (
            "attested_for_manual_promotion"
            if scope == MULTI_SEED_SCOPE
            else (
                "attested_for_engineering_promotion"
                if scope == ENGINEERING_SCOPE
                else "attested_for_seed42_scoped_manual_promotion"
            )
        ),
        "scope": scope,
        "advisory_exceptions": list(readiness.get("advisory_exceptions", [])),
        "claim_limitations": list(readiness.get("claim_limitations", [])),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_hash": manifest.hash,
        "heldout_seeds": seeds,
        "attacks": attacks,
        "manifest_mutated": False,
        "rtc_full_pointer_mutated": False,
        "sources": {
            "readiness": _source(readiness_source),
            "promotion_audit": _source(audit_source),
            "candidate_manifest": _source(candidate_manifest),
        },
    }


def build_promoted_manifest(
    *,
    readiness_path: str | Path,
    attestation_path: str | Path,
    candidate_manifest: str | Path,
) -> dict[str, Any]:
    readiness_source, readiness = _load_json(readiness_path)
    attestation_source, attestation = _load_json(attestation_path)
    candidate = CalibrationManifest.load(candidate_manifest)
    scope = str(readiness.get("scope", ""))
    if scope != ENGINEERING_SCOPE:
        raise ValueError("this promotion command only emits the engineering-scoped manifest")
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("invalid engineering attestation schema")
    if attestation.get("decision") != "attested_for_engineering_promotion":
        raise ValueError("engineering attestation decision mismatch")
    if str(readiness.get("candidate_hash")) != candidate.hash:
        raise ValueError("engineering readiness candidate hash mismatch")
    if str(attestation.get("candidate_hash")) != candidate.hash:
        raise ValueError("engineering attestation candidate hash mismatch")
    if attestation.get("advisory_exceptions") != readiness.get("advisory_exceptions"):
        raise ValueError("engineering advisory exceptions changed after readiness")
    promoted = json.loads(json.dumps(candidate.payload))
    metadata = promoted.setdefault("metadata", {})
    metadata.update(
        {
            "artifact_kind": "rtc_v3_semantic_temporal_exposure_engineering_promoted",
            "promotion_ready": True,
            "promotion_scope": ENGINEERING_SCOPE,
            "statistical_significance_claimed": False,
            "generalized_byzantine_claimed": False,
            "rtc_v2_status": "legacy_not_default",
            "advisory_exceptions": list(readiness.get("advisory_exceptions", [])),
            "engineering_promotion_sources": {
                "readiness": _source(readiness_source),
                "attestation": _source(attestation_source),
            },
        }
    )
    promoted["content_hash"] = content_hash(promoted)
    CalibrationManifest.load(promoted)
    return promoted


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite promotion evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--audit", required=True)
    readiness.add_argument("--candidate-manifest", required=True)
    readiness.add_argument("--candidate-validation", required=True)
    readiness.add_argument("--candidate-provenance", required=True)
    readiness.add_argument(
        "--scope",
        choices=(MULTI_SEED_SCOPE, SEED42_SCOPE, ENGINEERING_SCOPE),
        default=MULTI_SEED_SCOPE,
    )
    readiness.add_argument("--output", required=True)
    attestation = subparsers.add_parser("attestation")
    attestation.add_argument("--readiness", required=True)
    attestation.add_argument("--audit", required=True)
    attestation.add_argument("--candidate-manifest", required=True)
    attestation.add_argument("--output", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--readiness", required=True)
    promote.add_argument("--attestation", required=True)
    promote.add_argument("--candidate-manifest", required=True)
    promote.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.mode == "readiness":
        payload = build_readiness(
            audit_path=args.audit,
            candidate_manifest=args.candidate_manifest,
            candidate_validation=args.candidate_validation,
            candidate_provenance=args.candidate_provenance,
            scope=args.scope,
        )
    elif args.mode == "attestation":
        payload = build_attestation(
            readiness_path=args.readiness,
            audit_path=args.audit,
            candidate_manifest=args.candidate_manifest,
        )
    else:
        payload = build_promoted_manifest(
            readiness_path=args.readiness,
            attestation_path=args.attestation,
            candidate_manifest=args.candidate_manifest,
        )
    return _write_new(args.output, payload)


if __name__ == "__main__":
    main()
