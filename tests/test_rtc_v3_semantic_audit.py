from experiments.rtc_v3_semantic_audit import (
    CLEAN_BENIGN_CLIPPING_MAX,
    REQUIRED_EXTRA_HELDOUT_SEEDS,
    REQUIRED_ABLATION_DEFENSES,
    _as_paths,
    _assemble_ablation_summary,
    _clean_summary,
    _gate,
    _frozen_cone_runtime_matches,
    write_report,
)
from defenses.rtc.calibration import CalibrationManifest
import pandas as pd


def test_promotion_audit_uses_preregistered_clean_and_seed_gates() -> None:
    assert CLEAN_BENIGN_CLIPPING_MAX == 0.03
    assert REQUIRED_EXTRA_HELDOUT_SEEDS == 4
    assert REQUIRED_ABLATION_DEFENSES == {
        "fedavg",
        "clip_only",
        "rtc_v3_v2",
        "semantic_observe",
        "semantic_soft",
        "semantic_temporal",
        "semantic_exposure",
    }


def test_multiple_heldout_roots_must_be_unique(tmp_path) -> None:
    assert _as_paths([tmp_path / "a", tmp_path / "b"])[0].name == "a"
    try:
        _as_paths([tmp_path / "a", tmp_path / "a"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate held-out roots must fail")


def test_extra_seed_gate_can_be_promotion_only() -> None:
    ordinary = _gate("seed42", True, True, "true")
    extra = _gate(
        "four_extra_heldout_seeds_present",
        False,
        [42],
        "seed 42 plus four extras",
        promotion_only=True,
    )
    gates = [ordinary, extra]
    assert all(item["passed"] for item in gates if not item["promotion_only"])
    assert not all(item["passed"] for item in gates)


def test_frozen_cone_gate_compares_runtime_ids_to_phase_manifest(
    tmp_path,
) -> None:
    root = tmp_path / "run"
    (root / "rounds").mkdir(parents=True)
    pd.DataFrame(
        {
            "round": [1],
            "fit_rtc_v3_calibration_hash": ["candidate"],
            "fit_rtc_v3_cone_profile": ["warmup"],
            "fit_rtc_v3_cone_prototype_count": [2],
            "fit_rtc_v3_cone_prototype_ids_full": ["cone:1,cone:0"],
            "fit_rtc_v3_cone_online_updates_enabled": [0],
            "fit_rtc_v3_cone_update_applied_count": [0],
        }
    ).to_csv(root / "rounds" / "run.csv", index=False)
    manifest = CalibrationManifest(
        payload={
            "stable_cones": {
                "phases": {
                    "warmup": {
                        "prototypes": [
                            {"cone_id": "cone:0"},
                            {"cone_id": "cone:1"},
                        ]
                    }
                }
            }
        }
    )

    passed, mismatches = _frozen_cone_runtime_matches([root], manifest)

    assert passed is True
    assert mismatches == []


def test_report_preserves_metrics_instead_of_only_rendering_gates(tmp_path) -> None:
    destination = tmp_path / "report.md"
    write_report(
        {
            "candidate_manifest_hash": "candidate",
            "audit_stage": "seed42",
            "passed": True,
            "seed42_passed": True,
            "promotion_evidence_passed": False,
            "performance": {
                "semantic_mean_seconds": 0.004,
                "semantic_p95_seconds": 0.005,
                "defense_mean_increase_fraction": 0.01,
                "defense_p95_increase_fraction": 0.02,
            },
            "clean_comparison": {
                "seed": 42,
                "trial_plan_hash": "clean-plan",
                "candidate_final_accuracy": 0.89,
                "candidate_best_accuracy": 0.90,
                "candidate_last10_accuracy": 0.895,
                "baseline_final_accuracy": 0.891,
                "fedavg_final_accuracy": 0.892,
                "candidate_delta_vs_v2": -0.001,
                "candidate_delta_vs_fedavg": -0.002,
                "overflow_rate": 0.01,
                "benign_clipping_rate": 0.02,
                "semantic_budget_activation_events": 0,
                "false_persistence": 0.0,
                "false_soft_downweight_rate": 0.02,
                "false_persistent_principal_rate": 0.0,
                "false_persistence_max_observation_streak": 0,
            },
            "attack_comparison": [
                {
                    "attack": "label_flip_targeted",
                    "seed": 42,
                    "candidate_active_asr": 0.1,
                    "baseline_active_asr": 0.4,
                    "clip_only_active_asr": 0.39,
                    "candidate_peak_asr": 0.2,
                    "baseline_peak_asr": 0.75,
                    "clip_only_peak_asr": 0.72,
                    "candidate_final_accuracy": 0.88,
                    "baseline_final_accuracy": 0.86,
                    "clip_only_final_accuracy": 0.85,
                }
            ],
            "attack_runtime": {
                "semantic_budget_activation_events": 7,
                "prototype_update_max": 0,
                "max_constraint_violation": 0,
            },
            "recovery_metrics": {
                "time_to_recovery": 3,
                "risk_release_half_life": 5,
                "false_persistence": 0,
                "recovery_principal_count": 4,
                "recovered_principal_count": 4,
                "half_life_principal_count": 4,
            },
            "gates": [_gate("example", True, True, "true")],
        },
        destination,
    )
    report = destination.read_text(encoding="utf-8")
    assert "Held-out attack comparison" in report
    assert "0.100000" in report
    assert "Time-to-recovery: `3`" in report
    assert "semantic total mean | 0.004 s" in report
    assert "Promotion attestation generated: `false`" in report


def test_clean_summary_uses_audit_grade_analysis(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "defense": ["rtc_v3", "fedavg"],
            "best_accuracy": [0.90, 0.91],
            "last10_accuracy": [0.89, 0.90],
        }
    )
    monkeypatch.setattr(
        "experiments.rtc_v3_semantic_audit.analyze_root", lambda root: frame
    )
    selected = _clean_summary(tmp_path, "rtc_v3")
    assert selected.iloc[0]["best_accuracy"] == 0.90
    assert selected.iloc[0]["last10_accuracy"] == 0.89


def test_assembled_ablation_normalizes_reused_active_asr() -> None:
    common = {
        "trial_plan_hash": ["plan"],
        "seed": [42],
        "attack": ["label_flip_targeted"],
        "active_asr": [0.1],
    }
    dedicated = pd.DataFrame(
        {
            "trial_plan_hash": ["plan"],
            "seed": [42],
            "attack": ["label_flip_targeted"],
            "defense": ["fedavg"],
            "active_mean_asr": [0.4],
        }
    )
    result = _assemble_ablation_summary(
        dedicated=dedicated,
        candidate=pd.DataFrame(common),
        clip_only=pd.DataFrame(common),
        baseline=pd.DataFrame(common),
    )
    assert set(result["defense"]) == {
        "fedavg",
        "semantic_exposure",
        "clip_only",
        "rtc_v3_v2",
    }
    assert result["active_mean_asr"].notna().all()
