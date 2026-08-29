import json

import pandas as pd

from experiments.rtc_v3_semantic_completion_audit import (
    _root_complete,
    build_completion_audit,
)


def test_incomplete_root_is_pending_not_complete(tmp_path) -> None:
    complete, evidence = _root_complete(tmp_path / "missing")
    assert complete is False
    assert evidence["execution_validation"] is False


def test_root_requires_all_strict_gates_and_summary(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    pd.DataFrame([{"gate": "strict", "passed": True}]).to_csv(
        root / "execution_validation.csv", index=False
    )
    assert _root_complete(root)[0] is False
    pd.DataFrame([{"run": 1}]).to_csv(root / "comparison_runs.csv", index=False)
    assert _root_complete(root)[0] is True


def test_goal_audit_cannot_claim_completion_from_partial_artifacts(tmp_path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/rtc_v3_semantic_temporal_exposure_design.md").write_text(
        "design", encoding="utf-8"
    )
    evidence = build_completion_audit(tmp_path)
    assert evidence["complete"] is False
    assert evidence["promotion_attestation_generated"] is False
    statuses = {item["requirement"]: item["status"] for item in evidence["items"]}
    assert statuses["seed42_attacks"] == "pending"
    assert statuses["final_experiment_report"] == "pending"
    assert statuses["periodic_non_iid_adaptive_extensions"] == "pending"
    assert statuses["full_test_suite"] == "pending"
