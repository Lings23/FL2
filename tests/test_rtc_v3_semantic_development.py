from __future__ import annotations

import json
from pathlib import Path

from defenses.rtc.calibration import CalibrationManifest
from experiments.rtc_v3_semantic_development import _write_freeze_sidecars, main
import pytest


def test_freeze_sidecars_are_hash_bound_and_not_an_attestation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "candidate.json"
    evidence_path = tmp_path / "evidence.json"
    source.write_text("source", encoding="utf-8")
    output.write_text(
        json.dumps(
            {
                "schema_version": "rtc_v3.calibration.v3",
                "content_hash": "placeholder",
                "metadata": {
                    "formal_evaluation_ready": True,
                    "promotion_ready": False,
                    "development_provenance": {
                        "predevelopment_candidate_hash": "predev"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = {
        "candidate_hash": "predev",
        "decision": "freeze_unchanged",
        "seeds": [43, 44, 45],
        "attacks": ["label_flip_targeted", "label_flip_all_reverse"],
        "all_strict_gates_passed": True,
        "all_development_gates_passed": True,
        "gates": {
            "zero_update_rounds": 0,
            "prototype_update_max": 0.0,
            "max_constraint_violation": 1e-13,
        },
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    class StubManifest:
        hash = "frozen"
        payload = json.loads(output.read_text(encoding="utf-8"))

    _write_freeze_sidecars(
        output=output,
        manifest=StubManifest(),  # type: ignore[arg-type]
        evidence_path=evidence_path,
        evidence=evidence,
        source_manifest=source,
    )

    validation = json.loads(
        output.with_suffix(".validation.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    assert validation["passed"] is True
    assert validation["content_hash"] == "frozen"
    assert validation["promotion_ready"] is False
    assert provenance["promotion_attestation_generated"] is False


def test_development_evidence_input_and_rebuild_roots_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either --development-root"):
        main(
            [
                "--source-manifest",
                "source.json",
                "--development-root",
                "run",
                "--development-evidence",
                "evidence.json",
            ]
        )
