import json
import hashlib

import pandas as pd
import experiments.rtc_v3_cone_v2_promotion as promotion

from experiments.rtc_v3_cone_v2_promotion import (
    REQUIRED_GATES,
    _calibration_runtime_audit,
    _calibration_validation_audit,
    _clean_cone_audit,
    _execution_gates,
    _reverse_mapping_asr,
    _attack_client_diagnostics,
    _raw_pairing_audit,
    _raw_execution_audit,
)


def test_calibration_validation_audit_checks_each_envelope_and_phase():
    manifest = {
        "content_hash": "candidate-hash",
        "stable_cones": {
            "phases": {
                phase: {
                    "prototypes": [{"cone_id": "cone:0"}],
                    "max_angular_drift": 0.02,
                }
                for phase in ("warmup", "steady", "late")
            }
        },
    }
    validation = {
        "passed": True,
        "content_hash": "candidate-hash",
        "calibration_seeds": [40, 41],
        "evaluation_seeds": [42],
        "seed_sets_disjoint": True,
        "q_min_less_than_one": True,
        "q_value_less_than_one": True,
        "server_budget_envelopes": {
            "full/1/total": {"observed_max": 0.9, "budget_min": 1.0}
        },
        "principal_budget_envelopes": {
            "full/1": {"observed_max": 0.9, "beta_min": 1.0}
        },
        "cone_budget_envelopes": {
            "full/1/residual": {"observed_max": 0.9, "budget_min": 1.0}
        },
        "cone_clustering": {
            phase: {
                "coverage": 0.95,
                "prototype_count": 1,
                "max_angular_drift": 0.02,
            }
            for phase in ("warmup", "steady", "late")
        },
        "cone_sketch_dimension": 256,
    }

    evidence, failures = _calibration_validation_audit(validation, manifest)

    assert failures == []
    assert all(
        item["violation_count"] == 0
        for item in evidence["budget_envelopes"].values()
    )

    validation["cone_budget_envelopes"]["full/1/residual"]["observed_max"] = 1.01
    _, failures = _calibration_validation_audit(validation, manifest)
    assert any("cone budget envelope violations" in value for value in failures)


def test_execution_gate_requires_every_strict_pairing_gate(tmp_path):
    valid = pd.DataFrame(
        [{"gate": gate, "passed": True} for gate in sorted(REQUIRED_GATES)]
    )
    valid.to_csv(tmp_path / "execution_validation.csv", index=False)
    values, failures = _execution_gates(tmp_path)
    assert not failures
    assert all(values[gate] for gate in REQUIRED_GATES)

    frame = pd.read_csv(tmp_path / "execution_validation.csv")
    frame.loc[frame["gate"] == "client_random_stream_match", "passed"] = False
    frame.to_csv(tmp_path / "execution_validation.csv", index=False)
    _, failures = _execution_gates(tmp_path)
    assert failures == ["client_random_stream_match"]

    pd.concat([valid, valid.iloc[[0]]], ignore_index=True).to_csv(
        tmp_path / "execution_validation.csv", index=False
    )
    _, failures = _execution_gates(tmp_path)
    assert any("duplicate execution gate rows" in value for value in failures)

    invalid = valid.astype({"passed": "object"}).copy()
    invalid.loc[0, "passed"] = "maybe"
    invalid.to_csv(tmp_path / "execution_validation.csv", index=False)
    _, failures = _execution_gates(tmp_path)
    assert any("invalid execution gate boolean" in value for value in failures)


def test_raw_pairing_audit_recomputes_and_hash_binds_manifest(tmp_path, monkeypatch):
    payload = {"benchmark_version": 3, "specs": []}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        promotion,
        "validate_sampling_manifests",
        lambda *args, **kwargs: [
            {"gate": gate, "passed": True} for gate in promotion.RAW_PAIRING_GATES
        ],
    )
    saved = {gate: True for gate in promotion.RAW_PAIRING_GATES}

    recomputed, failures = _raw_pairing_audit(tmp_path, saved)

    assert failures == []
    assert recomputed == saved

    payload["sha256"] = "tampered"
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    _, failures = _raw_pairing_audit(tmp_path, saved)
    assert any("experiment manifest hash mismatch" in value for value in failures)


def test_raw_execution_audit_recomputes_round_and_finite_gates(tmp_path, monkeypatch):
    (tmp_path / "rounds").mkdir()
    (tmp_path / "status").mkdir()
    (tmp_path / "raw").mkdir()
    spec = {"defense": "fedavg", "attack": "none"}
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps({"specs": [spec]}), encoding="utf-8"
    )
    monkeypatch.setattr(promotion, "run_id", lambda value: "run")
    frame = pd.DataFrame({
        "round": range(1, 61),
        "server_accuracy": [0.9] * 60,
    })
    frame.to_csv(tmp_path / "rounds" / "run.csv", index=False)
    (tmp_path / "status" / "run.json").write_text(
        json.dumps({
            "state": "completed",
            "last_round": 60,
            "exit_code": 0,
        }),
        encoding="utf-8",
    )
    saved = {gate: True for gate in promotion.RAW_EXECUTION_GATES}

    recomputed, failures = _raw_execution_audit(tmp_path, saved)

    assert failures == []
    assert recomputed == saved

    (tmp_path / "status" / "run.json").write_text(
        json.dumps({
            "state": "completed_cached",
            "last_round": 60,
            "exit_code": 0,
        }),
        encoding="utf-8",
    )
    recomputed, failures = _raw_execution_audit(tmp_path, saved)
    assert failures == []
    assert recomputed == saved

    frame.loc[10, "server_accuracy"] = float("nan")
    frame.to_csv(tmp_path / "rounds" / "run.csv", index=False)
    recomputed, failures = _raw_execution_audit(tmp_path, saved)
    assert recomputed["critical_metrics_finite"] is False
    assert any("critical_metrics_finite" in value for value in failures)


def test_raw_execution_audit_turns_missing_run_root_into_gate_failure(tmp_path):
    saved = {gate: True for gate in promotion.RAW_EXECUTION_GATES}

    recomputed, failures = _raw_execution_audit(tmp_path, saved)

    assert not any(recomputed.values())
    assert len(failures) == 1
    assert "missing experiment manifest" in failures[0]


def test_reverse_mapping_asr_is_recovered_from_confusion_matrix():
    confusion = json.dumps([
        [0, 0, 4],
        [0, 5, 0],
        [3, 0, 0],
    ])
    assert _reverse_mapping_asr(confusion) == 1.0


def test_attack_audit_never_compares_mismatched_strict_join_keys(monkeypatch):
    def rows(seed):
        values = []
        for attack, group in (
            ("label_flip_targeted", "targeted"),
            ("label_flip_all_reverse", "untargeted"),
        ):
            values.append({
                "defense": "rtc_v3",
                "trial_plan_hash": f"plan-{attack}",
                "seed": seed,
                "attack_group": group,
                "attack": attack,
                "period": "continuous_1_0",
                "on_rounds": 1,
                "off_rounds": 0,
                "malicious_fraction": 0.2,
                "attack_start_round": 11,
                "attack_end_round": -1,
                "partition": "iid",
                "dirichlet_alpha": 0.5,
                "participation_rate": 0.5,
                "label_flip_poison_fraction": 1.0,
                "label_flip_source_label": 5,
                "label_flip_target_label": 3,
                "pairing_mode": "strict",
                "sampling_protocol": "principal_uniform",
            })
        return pd.DataFrame(values)

    monkeypatch.setattr(
        promotion,
        "_rtc_summary",
        lambda root, attacks: rows(42 if str(root) == "candidate" else 43),
    )
    monkeypatch.setattr(
        promotion,
        "_round_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched rows must not enter metric comparison")
        ),
    )

    compared, failures = promotion._attack_audit(
        promotion.Path("candidate"), promotion.Path("baseline")
    )

    assert compared == []
    assert len(failures) == 2
    assert all("strict attack join-key mismatch" in value for value in failures)


def test_attack_client_diagnostics_use_only_active_rounds(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"round": 10, "is_malicious": True, "clipped": False, "rtc_v3_cone_full": "overflow"},
        {"round": 10, "is_malicious": False, "clipped": False, "rtc_v3_cone_full": "cone:0"},
        {"round": 11, "is_malicious": True, "clipped": False, "rtc_v3_cone_full": "cone:1"},
        {"round": 11, "is_malicious": False, "clipped": True, "rtc_v3_cone_full": "cone:1"},
    ]).to_csv(raw / "run_clients.csv", index=False)
    (raw / "run_config.json").write_text(
        json.dumps({"security": {"attack": {"type": "label_flip_targeted"}}}),
        encoding="utf-8",
    )
    rounds = pd.DataFrame({
        "round": [10, 11],
        "planned_attack_active": [0.0, 1.0],
        "fit_malicious_aggregation_weight_share": [0.9, 0.2],
    })

    evidence, failures = _attack_client_diagnostics(
        tmp_path,
        attack="label_flip_targeted",
        rounds=rounds,
        label="candidate",
    )

    assert failures == []
    assert evidence["diagnostic_rounds"] == [11]
    assert evidence["malicious_weight_share"] == 0.2
    assert evidence["benign_clipping_rate"] == 1.0
    assert evidence["malicious_overflow_count"] == 0
    assert evidence["malicious_matched_count"] == 1


def test_clean_cone_audit_checks_phases_ids_drift_and_update_eligibility(tmp_path):
    rounds = tmp_path / "rounds"
    raw = tmp_path / "raw"
    rounds.mkdir()
    raw.mkdir()
    rows = []
    for number, phase in ((1, "warmup"), (11, "steady"), (31, "late")):
        rows.append({
            "round": number,
            "defense": "rtc_v3",
            "attack": "label_flip_targeted",
            "server_accuracy": 0.9,
            "fit_rtc_v3_cone_profile": phase,
            "fit_rtc_v3_cone_mode": "offline_controlled_v1",
            "fit_selected_malicious_clients": 2,
            "fit_selected_benign_clients": 8,
            "fit_rtc_v3_cone_overflow_assignments": 0,
            "fit_rtc_v3_cone_prototype_ids_full": "cone:0",
            "fit_rtc_v3_cone_prototype_count": 1,
            "fit_rtc_v3_cone_update_max_angular_drift": 0.01,
            "fit_rtc_v3_cone_update_candidate_count": 2,
            "fit_rtc_v3_cone_update_applied_count": 1,
            "fit_rtc_v3_cone_update_drift_clipped_count": 0,
            "fit_rtc_v3_cone_update_principal_support_json": '{"full":{"cone:0":2}}',
            "fit_rtc_v3_cone_update_applied_ids_json": '{"full":["cone:0"]}',
            "fit_rtc_v3_full_cone_cone_0_used_w1": 0.5,
            "fit_rtc_v3_full_cone_cone_0_budget_w1": 1.0,
        })
    pd.DataFrame(rows).to_csv(rounds / "clean_rtc_v3.csv", index=False)
    pd.DataFrame([{
        "round": 1,
        "rtc_v3_cone_full": "cone:0",
        "rtc_v3_cone_update_eligible_full": True,
        "rtc_v3_cumulative_q_full": 1.0,
        "rtc_v3_direction_q_full": 1.0,
        "clipped": False,
        "capped": False,
        "quarantined": False,
        "is_malicious": False,
    }]).to_csv(raw / "clean_clients.csv", index=False)
    (raw / "clean_config.json").write_text(
        json.dumps({"security": {"attack": {"type": "label_flip_targeted"}}}),
        encoding="utf-8",
    )
    manifest = {
        "stable_cones": {
            "phases": {
                phase: {
                    "max_angular_drift": 0.02,
                    "prototypes": [{"cone_id": "cone:0"}],
                }
                for phase in ("warmup", "steady", "late")
            }
        }
    }
    evidence, failures = _clean_cone_audit(tmp_path, manifest)
    assert failures == []
    assert evidence["overflow_rate"] == 0.0
    assert evidence["invalid_update_eligible_clients"] == 0
    assert evidence["cone_budget_activation"]["eligible_cells"] == 3
    assert evidence["cone_budget_activation"]["rate"] == 0.0

    frame = pd.read_csv(raw / "clean_clients.csv")
    frame["clipped"] = True
    frame.to_csv(raw / "clean_clients.csv", index=False)
    _, failures = _clean_cone_audit(tmp_path, manifest)
    assert any("update-eligible" in value for value in failures)

    frame["clipped"] = False
    frame["rtc_v3_cumulative_q_full"] = 0.5
    frame.to_csv(raw / "clean_clients.csv", index=False)
    _, failures = _clean_cone_audit(tmp_path, manifest)
    assert any("cumulative/direction q<1" in value for value in failures)

    frame["rtc_v3_cumulative_q_full"] = 1.0
    frame.to_csv(raw / "clean_clients.csv", index=False)
    round_frame = pd.read_csv(rounds / "clean_rtc_v3.csv")
    round_frame.loc[0, "fit_rtc_v3_cone_update_principal_support_json"] = (
        '{"full":{"cone:0":1}}'
    )
    round_frame.to_csv(rounds / "clean_rtc_v3.csv", index=False)
    _, failures = _clean_cone_audit(tmp_path, manifest)
    assert any("without two-principal support" in value for value in failures)


def test_calibration_runtime_audit_proves_full_deterministic_contract(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "rounds").mkdir()
    (tmp_path / "status").mkdir()
    (tmp_path / "trial_plans").mkdir()
    sources = []
    for seed in (40, 41):
        run_id = f"none__calibration_clean__s{seed}"
        from experiments.trial_plan import sha256_json

        plan = {
            "schema_version": "trial-plan-v1",
            "pairing_group_id": f"group-{seed}",
            "sampling_protocol": "principal_uniform",
            "malicious_partition_ids": [],
            "rounds": [],
        }
        trial_hash = sha256_json(plan)
        plan["trial_plan_hash"] = trial_hash
        plan_path = tmp_path / "trial_plans" / f"{seed}.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        config = {
            "project": {"seed": seed},
            "federation": {
                "num_rounds": 60,
                "num_clients": 20,
                "clients_per_round": 10,
                "max_client_samples": 0,
                "pairing_mode": "strict",
                "sampling_protocol": "principal_uniform",
                "deterministic_client_training": True,
                "trial_plan_hash": trial_hash,
                "trial_plan_path": str(plan_path),
            },
            "client": {"batch_size": 48},
            "dataset": {"partition": "iid", "name": "cifar10"},
            "model": {
                "architecture": "resnet18",
                "num_classes": 10,
                "pretrained": False,
            },
            "evaluation": {"max_test_samples": 100},
            "ray": {"client_num_gpus": 0.25},
            "security": {"attack": {"enabled": False, "type": "none"}},
        }
        config_path = tmp_path / "raw" / f"{run_id}_config.json"
        config_path.write_text(
            json.dumps(config), encoding="utf-8"
        )
        data = {"dataset": "cifar10", "seed": seed, "max_client_samples": 0}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        data["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        data_path = tmp_path / "raw" / f"{run_id}_data_manifest.json"
        data_path.write_text(
            json.dumps(data), encoding="utf-8"
        )
        round_path = tmp_path / "rounds" / f"{run_id}.csv"
        pd.DataFrame({"round": range(1, 61)}).to_csv(round_path, index=False)
        clients = pd.DataFrame({"round": [value for value in range(1, 61) for _ in range(10)]})
        client_path = tmp_path / "raw" / f"{run_id}_clients.csv"
        clients.to_csv(client_path, index=False)
        sidecar = pd.DataFrame({
            "round": clients["round"],
            **{
                f"rtc_v3_sketch_full_{index:03d}": 0.0
                for index in range(256)
            },
        })
        sidecar_path = tmp_path / "raw" / f"{run_id}_rtc_v3_sketches.parquet"
        sidecar.to_parquet(sidecar_path, index=False)
        (tmp_path / "status" / f"{run_id}.json").write_text(
            json.dumps({
                "run_id": run_id,
                "state": "completed",
                "last_round": 60,
                "exit_code": 0,
            }),
            encoding="utf-8",
        )
        from experiments.rtc_v3_cone_v2_promotion import _sha256

        sources.append({
            "run_id": run_id,
            "seed": seed,
            "trial_plan_hash": trial_hash,
            "round_sha256": _sha256(round_path),
            "client_sha256": _sha256(client_path),
            "config_sha256": _sha256(config_path),
            "data_manifest_sha256": data["sha256"],
            "sketch_sidecar_sha256": _sha256(sidecar_path),
            "sketch_dimension": 256,
            "sketch_algorithm_version": "splitmix64_v1",
        })
    from defenses.rtc.calibration import model_metadata_hash
    from models.model_factory import get_model, get_parameters

    runtime_model = get_model(
        architecture="resnet18",
        num_classes=10,
        pretrained=False,
        dataset_name="cifar10",
    )
    manifest = {
        "model_metadata_hash": model_metadata_hash(get_parameters(runtime_model)),
        "metadata": {
            "reserved_evaluation_seeds": [42],
            "source_runs": sources,
        }
    }
    evidence, failures = _calibration_runtime_audit(
        tmp_path, manifest, {"source_runs": sources}
    )
    assert failures == []
    assert evidence["observed_seeds"] == [40, 41]
    assert len(evidence["contracts"]) == 2

    round_path.write_text(round_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _, failures = _calibration_runtime_audit(
        tmp_path, manifest, {"source_runs": sources}
    )
    assert any("round provenance mismatch" in value for value in failures)
