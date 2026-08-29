"""Requirement-by-requirement completion audit for the semantic RTC-V3 goal.

This is deliberately separate from the promotion auditor: incomplete work is
reported as pending instead of being mistaken for a failed defense gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
    }


def _root_complete(root: Path) -> tuple[bool, dict[str, Any]]:
    validation = root / "execution_validation.csv"
    summary = root / "comparison_runs.csv"
    evidence = {
        "root": str(root.resolve()),
        "execution_validation": validation.is_file(),
        "comparison_summary": summary.is_file(),
    }
    if not validation.is_file():
        return False, evidence
    import pandas as pd

    frame = pd.read_csv(validation)
    passed = bool(
        {"gate", "passed"}.issubset(frame)
        and len(frame)
        and frame["passed"].astype(str).str.lower().isin({"true", "1", "1.0"}).all()
    )
    evidence["all_gates_passed"] = passed
    evidence["gate_count"] = int(len(frame))
    return bool(passed and summary.is_file()), evidence


def _item(
    requirement: str,
    status: str,
    evidence: Any,
    expected: str,
) -> dict[str, Any]:
    if status not in {"complete", "pending", "failed"}:
        raise ValueError(f"invalid completion status: {status}")
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "expected": expected,
    }


def build_completion_audit(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    required_artifacts = {
        "design": root / "docs/rtc_v3_semantic_temporal_exposure_design.md",
        "implementation": root / "docs/rtc_v3_semantic_temporal_exposure_implementation.md",
        "report": root / "docs/rtc_v3_semantic_temporal_exposure_experiment_report.md",
        "reproduction": root / "docs/rtc_v3_semantic_temporal_exposure_reproduction.md",
        "candidate": root / "config/rtc_v3_manifest_formal_iid_semantic_candidate.json",
        "validation": root / "config/rtc_v3_manifest_formal_iid_semantic_candidate.validation.json",
        "provenance": root / "config/rtc_v3_manifest_formal_iid_semantic_candidate.provenance.json",
    }
    artifact_evidence = {name: _artifact(path) for name, path in required_artifacts.items()}
    foundational = all(
        artifact_evidence[name]["exists"]
        for name in ("design", "implementation", "reproduction", "candidate", "validation", "provenance")
    )
    experiment_roots = {
        "seed42_attacks": root / "logs/rtc_v3_semantic_heldout_s42_attacks",
        "seed42_clean": root / "logs/rtc_v3_semantic_heldout_s42_clean",
        "seed42_recovery": root / "logs/rtc_v3_semantic_heldout_s42_recovery",
        "seed42_ablations": root / "logs/rtc_v3_semantic_heldout_s42_ablations",
        "cpu_smoke": root / "logs/rtc_v3_semantic_cpu_smoke_final",
    }
    root_evidence: dict[str, Any] = {}
    root_status: dict[str, bool] = {}
    for name, path in experiment_roots.items():
        complete, evidence = _root_complete(path)
        root_status[name] = complete
        root_evidence[name] = evidence
    performance = root / "logs/rtc_v3_semantic_performance_formal/performance_gates.json"
    performance_payload = (
        json.loads(performance.read_text(encoding="utf-8"))
        if performance.is_file()
        else {}
    )
    seed42_audit = root / "logs/rtc_v3_semantic_seed42_gate_audit.json"
    promotion_audit = root / "logs/rtc_v3_semantic_promotion_gate_audit.json"
    expansion_registry = root / "logs/rtc_v3_semantic_expansion_registry.json"
    expansion_payload = (
        json.loads(expansion_registry.read_text(encoding="utf-8"))
        if expansion_registry.is_file()
        else {}
    )
    expansion_conditions = expansion_payload.get("conditions", [])
    expansion_executed = bool(
        expansion_conditions
        and all(item.get("status") == "completed_strict" for item in expansion_conditions)
    )
    pytest_evidence = root / "logs/rtc_v3_semantic_pytest_evidence.json"
    pytest_payload = (
        json.loads(pytest_evidence.read_text(encoding="utf-8"))
        if pytest_evidence.is_file()
        else {}
    )
    items = [
        _item(
            "design_implementation_calibration_candidate",
            "complete" if foundational else "pending",
            artifact_evidence,
            "design, implementation, reproduction, candidate, validation and provenance exist",
        ),
        _item(
            "performance_gates",
            "complete" if bool(performance_payload.get("passed")) else "pending",
            _artifact(performance),
            "performance_gates.json passed=true",
        ),
        _item(
            "full_test_suite",
            "complete"
            if bool(pytest_payload.get("passed"))
            and int(pytest_payload.get("tests", 0)) > 0
            and int(pytest_payload.get("failures", -1)) == 0
            and int(pytest_payload.get("errors", -1)) == 0
            else "pending",
            {
                "evidence": _artifact(pytest_evidence),
                "tests": pytest_payload.get("tests"),
                "failures": pytest_payload.get("failures"),
                "errors": pytest_payload.get("errors"),
                "junit_sha256": pytest_payload.get("junit_sha256"),
                "source_snapshot_sha256": pytest_payload.get("source_snapshot_sha256"),
            },
            "non-empty hash-bound JUnit evidence with zero failures/errors",
        ),
        *(
            _item(
                name,
                "complete" if root_status[name] else "pending",
                root_evidence[name],
                "execution_validation all true and comparison_runs.csv exists",
            )
            for name in experiment_roots
        ),
        _item(
            "seed42_gate_audit",
            "complete"
            if seed42_audit.is_file()
            and bool(json.loads(seed42_audit.read_text(encoding="utf-8")).get("passed"))
            else "pending",
            _artifact(seed42_audit),
            "seed42 audit exists and passed=true",
        ),
        _item(
            "four_extra_heldout_seeds",
            "complete"
            if promotion_audit.is_file()
            and bool(json.loads(promotion_audit.read_text(encoding="utf-8")).get("promotion_evidence_passed"))
            else "pending",
            _artifact(promotion_audit),
            "promotion audit proves at least four complete extra held-out seeds",
        ),
        _item(
            "periodic_non_iid_adaptive_extensions",
            "complete" if expansion_executed else "pending",
            {
                "registry": _artifact(expansion_registry),
                "registry_validation_passed": bool(
                    expansion_payload.get("validation", {}).get("passed")
                ),
                "condition_statuses": {
                    str(item.get("id")): str(item.get("status"))
                    for item in expansion_conditions
                },
            },
            "short/long/sparse, Dirichlet 0.5/0.1 and adaptive pressure results",
        ),
        _item(
            "final_experiment_report",
            "complete" if artifact_evidence["report"]["exists"] else "pending",
            artifact_evidence["report"],
            "final evidence-backed experiment report exists",
        ),
    ]
    return {
        "artifact_kind": "rtc_v3_semantic_goal_completion_audit_v1",
        "workspace": str(root),
        "complete": bool(items and all(item["status"] == "complete" for item in items)),
        "promotion_attestation_generated": False,
        "items": items,
    }


def main(argv: Iterable[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--output", default="logs/rtc_v3_semantic_completion_audit.json"
    )
    args = parser.parse_args(argv)
    evidence = build_completion_audit(args.workspace)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    main()
