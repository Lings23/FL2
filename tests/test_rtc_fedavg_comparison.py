"""Matrix and report tests for the focused RTC/FedAvg experiment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from defenses.rtc.calibration import build_manifest
from experiments.rtc_fedavg_comparison import (
    DEFAULT_ATTACKS,
    RTC_V3_DEFENSE_TYPE,
    add_clean_comparators,
    build_category_report,
    build_comparison_matrix,
    build_pairwise_report,
    build_rtc_v3_custom_params,
    parse_args,
    run_comparison,
    validate_execution,
)
from experiments.periodic_attack import run_id


def test_execution_validation_recovers_legacy_all_reverse_asr(tmp_path):
    rounds_dir = tmp_path / "rounds"
    raw_dir = tmp_path / "raw"
    rounds_dir.mkdir()
    raw_dir.mkdir()
    spec = {
        "attack": "label_flip_all_reverse",
        "attack_group": "untargeted",
        "period": "continuous_1_0",
        "on_rounds": 1,
        "off_rounds": 0,
        "attack_start_round": 1,
        "attack_end_round": -1,
        "malicious_fraction": 0.2,
        "seed": 42,
        "defense": "fedavg",
        "partition": "iid",
        "participation_rate": 0.5,
        "label_flip_poison_fraction": 1.0,
    }
    frame = pd.DataFrame({
        "round": [0, 1],
        "server_accuracy": [0.1, 0.2],
        "server_confusion_matrix_json": [
            json.dumps([[0, 2], [3, 0]]),
            json.dumps([[0, 4], [5, 0]]),
        ],
        "planned_attack_active": [0, 1],
    })
    frame.to_csv(rounds_dir / f"{run_id(spec)}.csv", index=False)

    gates = validate_execution([spec], rounds_dir, raw_dir, expected_rounds=1)

    finite = next(row for row in gates if row["gate"] == "critical_metrics_finite")
    assert finite["passed"] is True


def test_semantic_ablation_is_validated_as_rtc_v3(tmp_path, monkeypatch):
    rounds_dir = tmp_path / "rounds"
    raw_dir = tmp_path / "raw"
    rounds_dir.mkdir()
    raw_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    payload = build_manifest(
        params=[np.zeros(1, dtype=np.float32)],
        schema_version="rtc_v3.calibration.v1",
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    spec = {
        "attack": "label_flip_targeted",
        "attack_group": "targeted",
        "period": "continuous_1_0",
        "on_rounds": 1,
        "off_rounds": 0,
        "attack_start_round": 1,
        "attack_end_round": -1,
        "malicious_fraction": 0.2,
        "seed": 42,
        "defense": "semantic_observe",
        "partition": "iid",
        "participation_rate": 0.5,
        "label_flip_poison_fraction": 1.0,
        "label_flip_source_label": 5,
        "label_flip_target_label": 3,
        "custom_params": {"calibration_path": str(manifest_path)},
    }
    pd.DataFrame(
        {
            "round": [0, 1],
            "server_accuracy": [0.1, 0.2],
            "server_asr": [0.0, 0.5],
            "planned_attack_active": [0, 1],
            "fit_rtc_v3_calibration_hash": [np.nan, "wrong-hash"],
            "fit_rtc_v3_max_constraint_violation": [np.nan, 0.0],
            "fit_rtc_v3_weight_sum": [np.nan, 1.0],
        }
    ).to_csv(rounds_dir / f"{run_id(spec)}.csv", index=False)
    monkeypatch.setattr(
        "experiments.rtc_fedavg_comparison.validate_sampling_manifests",
        lambda *args, **kwargs: [],
    )

    gates = validate_execution([spec], rounds_dir, raw_dir, expected_rounds=1)

    manifest_gate = next(
        row for row in gates if row["gate"] == "rtc_v3_manifest_hash_match"
    )
    assert manifest_gate["passed"] is False


def test_default_matrix_covers_both_attack_groups_and_periods():
    matrix = build_comparison_matrix()

    assert len(matrix) == 14
    assert {spec["seed"] for spec in matrix} == {42}
    assert {spec["defense"] for spec in matrix} == {"fedavg", "rtc_full"}
    assert {spec["period"] for spec in matrix} == {"continuous_1_0"}
    assert {spec["attack"] for spec in matrix} == set(DEFAULT_ATTACKS)
    assert {spec["attack_group"] for spec in matrix} == {"targeted", "untargeted"}
    assert all(spec["attack_start_round"] == 11 for spec in matrix)


def test_clean_comparators_add_one_trajectory_per_defense():
    matrix = build_comparison_matrix(attacks=("model_replacement",))
    complete = add_clean_comparators(matrix)
    clean = [spec for spec in complete if spec["attack"] == "none"]

    assert len(matrix) == 2
    assert len(clean) == 2
    assert {spec["defense"] for spec in clean} == {"fedavg", "rtc_full"}
    assert {spec["attack_group"] for spec in clean} == {"clean"}


def test_matrix_can_add_clip_only_without_changing_public_pair():
    matrix = build_comparison_matrix(
        attacks=("model_replacement",), periods=("long_3_3",),
        defenses=("fedavg", "clip_only", "rtc_full"),
    )

    assert len(matrix) == 3
    assert {spec["defense"] for spec in matrix} == {"fedavg", "clip_only", "rtc_full"}


def test_label_flip_uses_explicit_semantics_and_ignores_boost_factor():
    matrix = build_comparison_matrix(
        attacks=("label_flip_targeted",), periods=("continuous_1_0",),
        defenses=("fedavg",),
        boost_factor=50.0,
    )

    assert len(matrix) == 1
    assert matrix[0]["period"] == "continuous_1_0"
    assert matrix[0]["on_rounds"] == 1
    assert matrix[0]["off_rounds"] == 0
    assert matrix[0]["label_flip_source_label"] == 5
    assert matrix[0]["label_flip_target_label"] == 3
    assert matrix[0]["label_flip_poison_fraction"] == pytest.approx(1.0)
    assert "boost_factor" not in matrix[0]


def test_legacy_label_flip_name_is_rejected():
    with pytest.raises(ValueError, match="was removed"):
        build_comparison_matrix(attacks=("label_flip",))


def test_label_flip_run_id_tracks_protocol_not_boost_factor():
    spec = build_comparison_matrix(
        attacks=("label_flip_targeted",),
        defenses=("fedavg",),
    )[0]
    with_irrelevant_boost = {**spec, "boost_factor": 50.0}

    assert "lf5to3-p1" in run_id(spec)
    assert run_id(spec) == run_id(with_irrelevant_boost)

    reverse = build_comparison_matrix(
        attacks=("label_flip_all_reverse",),
        defenses=("fedavg",),
        label_flip_source_label=1,
        label_flip_target_label=9,
    )[0]
    assert "label_flip_source_label" not in reverse
    assert "label_flip_target_label" not in reverse
    assert "lfrev-p1" in run_id(reverse)


def test_cli_defaults_lock_sixty_rounds_and_one_seed():
    args = parse_args([])

    assert args.rounds == 60
    assert args.seed == 42
    assert args.attacks.split(",") == list(DEFAULT_ATTACKS)
    assert args.periods == "continuous_1_0"


def test_rtc_v3_matrix_uses_frozen_manifest_and_identity_principals(tmp_path):
    manifest_path = tmp_path / "rtc_v3_manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(params=[np.zeros(2)])),
        encoding="utf-8",
    )
    custom_params = build_rtc_v3_custom_params(
        manifest_path=manifest_path,
        implementation_phase=6,
        num_clients=3,
    )

    matrix = build_comparison_matrix(
        attacks=("label_flip_targeted",),
        defenses=("fedavg", "rtc_v3"),
        rtc_v3_custom_params=custom_params,
    )
    rtc = next(spec for spec in matrix if spec["defense"] == "rtc_v3")

    assert rtc["defense_type"] == RTC_V3_DEFENSE_TYPE
    assert rtc["benchmark_version"] == 3
    assert rtc["custom_params"]["implementation_phase"] == 6
    assert rtc["custom_params"]["principal_map"] == {"0": "0", "1": "1", "2": "2"}
    # The raw defense config cannot attest to sampling. TrialPlanV1 flips this
    # only after the concrete principal-uniform schedule has been validated.
    assert rtc["custom_params"]["principal_first_sampling_verified"] is False


def test_rtc_v3_defaults_to_promoted_manifest_and_rejects_incomplete_principal_map(tmp_path):
    defaults = build_rtc_v3_custom_params(
        manifest_path="",
        implementation_phase=6,
        num_clients=2,
    )
    assert defaults["calibration_path"].endswith(
        "rtc_v3_manifest_formal_iid_semantic.json"
    )

    manifest_path = tmp_path / "rtc_v3_manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(params=[np.zeros(2)])),
        encoding="utf-8",
    )
    principal_path = tmp_path / "principals.json"
    principal_path.write_text(json.dumps({"0": "person-a"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing simulated client IDs"):
        build_rtc_v3_custom_params(
            manifest_path=manifest_path,
            implementation_phase=6,
            num_clients=2,
            principal_map_path=principal_path,
        )

    matrix = build_comparison_matrix(
        attacks=("label_flip_targeted",),
        defenses=("fedavg", "rtc_v3"),
        num_clients=2,
    )
    rtc = next(row for row in matrix if row["defense"] == "rtc_v3")
    assert rtc["custom_params"]["calibration_path"].endswith(
        "rtc_v3_manifest_formal_iid_semantic.json"
    )


def test_rtc_v3_dry_run_writes_executable_candidate_specs(tmp_path):
    manifest_path = tmp_path / "rtc_v3_manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(params=[np.zeros(2)])),
        encoding="utf-8",
    )
    output = tmp_path / "matrix"
    args = parse_args([
        "--attacks", "label_flip_targeted",
        "--defenses", "fedavg,rtc_v3",
        "--rtc-v3-manifest", str(manifest_path),
        "--num-clients", "3",
        "--output", str(output),
        "--dry-run",
    ])

    result = run_comparison(args)
    matrix = pd.read_csv(output / "experiment_matrix.csv")
    rtc_rows = matrix[matrix["defense"] == "rtc_v3"]
    custom_params = [
        json.loads(value) for value in rtc_rows["custom_params"].tolist()
    ]

    assert result.empty
    assert set(matrix[matrix["attack"] == "none"]["defense"]) == {"fedavg"}
    assert set(rtc_rows["defense_type"]) == {RTC_V3_DEFENSE_TYPE}
    assert all(
        params["calibration_path"] == str(manifest_path.resolve())
        for params in custom_params
    )
    assert all(
        params["principal_map"] == {"0": "0", "1": "1", "2": "2"}
        for params in custom_params
    )
    assert all(params["principal_first_sampling_verified"] for params in custom_params)
    assert matrix["trial_plan_hash"].nunique() == 1


def test_rtc_v3_dry_run_can_skip_duplicate_shared_clean(tmp_path):
    manifest_path = tmp_path / "rtc_v3_manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(params=[np.zeros(2)])),
        encoding="utf-8",
    )
    output = tmp_path / "matrix"
    args = parse_args([
        "--attacks", "label_flip_targeted,label_flip_all_reverse",
        "--defenses", "rtc_v3",
        "--rtc-v3-manifest", str(manifest_path),
        "--num-clients", "3",
        "--output", str(output),
        "--skip-clean",
        "--dry-run",
    ])

    result = run_comparison(args)
    matrix = pd.read_csv(output / "experiment_matrix.csv")

    assert result.empty
    assert set(matrix["attack"]) == {
        "label_flip_targeted", "label_flip_all_reverse",
    }
    assert set(matrix["defense"]) == {"rtc_v3"}
    assert len(matrix) == 2


def test_pairwise_report_uses_objective_specific_primary_metrics():
    summary = pd.DataFrame([
        {"attack_group": "targeted", "attack": "model_replacement", "period": "short_1_1",
         "seed": 42, "malicious_fraction": 0.2, "defense": "fedavg",
         "active_asr_auc": 10.0, "final_accuracy": 0.7},
        {"attack_group": "targeted", "attack": "model_replacement", "period": "short_1_1",
         "seed": 42, "malicious_fraction": 0.2, "defense": "rtc_full",
         "active_asr_auc": 2.0, "final_accuracy": 0.69},
        {"attack_group": "untargeted", "attack": "byzantine", "period": "long_3_3",
         "seed": 42, "malicious_fraction": 0.2, "defense": "fedavg",
         "active_accuracy_drop": 0.4, "final_accuracy": 0.3},
        {"attack_group": "untargeted", "attack": "byzantine", "period": "long_3_3",
         "seed": 42, "malicious_fraction": 0.2, "defense": "rtc_full",
         "active_accuracy_drop": 0.1, "final_accuracy": 0.6},
    ])

    report = build_pairwise_report(summary).set_index("attack_group")

    assert report.loc["targeted", "primary_metric"] == "active_asr_auc"
    assert report.loc["targeted", "rtc_defense"] == "rtc_full"
    assert report.loc["targeted", "rtc_over_fedavg"] == pytest.approx(0.2)
    assert report.loc["untargeted", "primary_metric"] == "active_accuracy_drop"
    assert report.loc["untargeted", "rtc_over_fedavg"] == pytest.approx(0.25)
    grouped = build_category_report(report.reset_index())
    assert set(grouped["period"]) == {"short_1_1", "long_3_3"}


def test_pairwise_report_does_not_mislabel_ablation_as_rtc():
    summary = pd.DataFrame([
        {"attack_group": "targeted", "attack": "model_replacement",
         "period": "continuous_1_0", "seed": 42, "malicious_fraction": 0.2,
         "defense": "fedavg", "active_asr_auc": 10.0},
        {"attack_group": "targeted", "attack": "model_replacement",
         "period": "continuous_1_0", "seed": 42, "malicious_fraction": 0.2,
         "defense": "clip_only", "active_asr_auc": 5.0},
        {"attack_group": "targeted", "attack": "model_replacement",
         "period": "continuous_1_0", "seed": 42, "malicious_fraction": 0.2,
         "defense": "rtc_v3", "active_asr_auc": 2.0},
    ])

    report = build_pairwise_report(summary)

    assert len(report) == 1
    assert report.iloc[0]["rtc_defense"] == "rtc_v3"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attacks": ("unknown",)},
        {"periods": ("medium",)},
        {"malicious_fraction": 1.0},
        {"participation_rate": 0.0},
    ],
)
def test_invalid_matrix_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_comparison_matrix(**kwargs)
