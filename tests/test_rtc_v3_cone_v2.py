import math

import numpy as np
import pandas as pd
import pytest

from defenses.rtc.calibration import CalibrationManifest, SCHEMA_VERSION_V2, build_manifest
from defenses.rtc.cone_calibration import calibrate_offline_cones
from defenses.rtc.prototypes import Prototype, StablePrototypeBank


def _clients() -> pd.DataFrame:
    rows = []
    phases = ((1, "warmup"), (11, "steady"), (31, "late"))
    for seed in (40, 41):
        for start, profile in phases:
            for offset in range(2):
                for client in range(6):
                    sign = 1.0 if client < 3 else -1.0
                    vector = np.zeros(256, dtype=np.float32)
                    vector[0] = sign
                    vector[1] = (client % 3 - 1) * 0.01 + (seed - 40) * 0.001
                    vector /= np.linalg.norm(vector)
                    row = {
                        "calibration_seed": seed,
                        "round": start + offset,
                        "profile": profile,
                        "cid": str(client),
                        "principal_id": str(client),
                        "nominal_mass": 1 / 6,
                        "rtc_v3_residual_norm": 1.0,
                    }
                    row.update(
                        {
                            f"rtc_v3_sketch_full_{index:03d}": float(value)
                            for index, value in enumerate(vector)
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def test_offline_calibration_is_deterministic_and_covers_clean_samples():
    clients = _clients()
    first, assignments_a, diagnostics_a = calibrate_offline_cones(
        clients, profile_names=("warmup", "steady", "late")
    )
    second, assignments_b, diagnostics_b = calibrate_offline_cones(
        clients.sample(frac=1.0, random_state=9),
        profile_names=("warmup", "steady", "late"),
    )

    assert first == second
    assert diagnostics_a == diagnostics_b
    assert assignments_a["cone_id"].tolist() == assignments_b["cone_id"].tolist()
    assert all(value["coverage"] >= 0.95 for value in diagnostics_a.values())
    assert first["mode"] == "offline_controlled_v1"


def test_controlled_bank_never_creates_ids_and_caps_drift():
    prototypes = (
        Prototype("cone:0", np.asarray([1.0, 0.0]), 0),
        Prototype("cone:1", np.asarray([-1.0, 0.0]), 0),
    )
    bank = StablePrototypeBank(
        initial_prototypes=prototypes,
        controlled=True,
        match_threshold=0.5,
        update_threshold=0.8,
        momentum=0.0,
        max_angular_drift=0.05,
    )
    transition = bank.prepare(
        server_round=1,
        sketches=(np.asarray([0.9, 0.4]), np.asarray([0.9, 0.3]), np.asarray([0.0, 1.0])),
        nominal_masses=(0.3, 0.3, 0.4),
        client_ids=("a", "b", "c"),
        principal_ids=("pa", "pb", "pc"),
    )
    bank.commit(transition, (True, True, True))

    assert set(bank.snapshot()) == {"cone:0", "cone:1"}
    assert bank.last_update_metrics["applied_count"] == 1
    assert bank.last_update_metrics["principal_support_by_cone"]["cone:0"] == 2
    assert bank.last_update_metrics["applied_cone_ids"] == ("cone:0",)
    assert bank.last_update_metrics["max_angular_drift"] <= 0.05 + 1e-12
    assert math.acos(float(np.dot(bank.snapshot()["cone:0"], np.asarray([1.0, 0.0])))) <= 0.05 + 1e-12


def test_v1_legacy_baseline_preserves_exact_hash_creation_semantics():
    bank = StablePrototypeBank(match_threshold=0.85)
    transition = bank.prepare(
        server_round=1,
        sketches=(
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.999, 0.001]),
        ),
        nominal_masses=(0.2, 0.3, 0.5),
        client_ids=("a", "b", "c"),
    )
    bank.commit(transition)

    # Exact duplicates share one V1 candidate, while a merely near-identical
    # sketch remains a distinct candidate as it did in the original baseline.
    assert len(bank.prototypes) == 2


def test_v2_manifest_rejects_tampered_offline_prototype():
    phase = {
        "match_threshold": 0.5,
        "update_threshold": 0.8,
        "max_angular_drift": 0.1,
        "prototypes": [{"cone_id": "cone:0", "vector": [1.0, 0.0]}],
    }
    payload = build_manifest(
        params=[np.zeros(1, dtype=np.float32)],
        schema_version=SCHEMA_VERSION_V2,
        training_phases=[{"start_round": 1, "profile": "warmup"}],
        stable_cones={
            "enabled": True,
            "mode": "offline_controlled_v1",
            "dimension": 2,
            "max_prototypes": 16,
            "phases": {"warmup": phase},
        },
    )
    CalibrationManifest.load(payload)
    payload["stable_cones"]["phases"]["warmup"]["prototypes"][0]["vector"] = [2.0, 0.0]
    # Re-hashing a semantically invalid vector must still fail static validation.
    from defenses.rtc.calibration import content_hash

    payload["content_hash"] = content_hash(payload)
    with pytest.raises(ValueError, match="invalid offline prototype"):
        CalibrationManifest.load(payload)
