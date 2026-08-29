"""Phase 0 gates: version isolation, manifest fail-fast, and report semantics."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from config.config_loader import DefenseConfig
from defenses.defense_base import get_defense
from defenses.rtc.calibration import CalibrationManifest, build_manifest
from defenses.rtc.v3 import RTCv3Defense
from defenses.time_consistency_defense import TimeConsistencyDefense
from experiments.periodic_attack import _auc, _normalized_auc, summarize_run


def _params() -> list[np.ndarray]:
    return [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.asarray([7], dtype=np.int64),
    ]


def _v3_config(manifest: dict, **params) -> DefenseConfig:
    return DefenseConfig(
        enabled=True,
        type="rtc_v3_candidate",
        custom_params={
            "calibration_manifest": manifest,
            "implementation_phase": 0,
            **params,
        },
    )


def test_only_explicit_legacy_aliases_remain_rtc_v2():
    for name in ("time_consistency", "rtc_v2_legacy"):
        defense = get_defense(DefenseConfig(enabled=True, type=name))
        assert isinstance(defense, TimeConsistencyDefense)


def test_rtc_full_is_promoted_v3_alias():
    manifest = build_manifest(params=_params())
    cfg = _v3_config(manifest)
    cfg.type = "rtc_full"
    defense = get_defense(cfg)
    assert isinstance(defense, RTCv3Defense)


def test_v3_has_separate_entry_and_requires_manifest():
    with pytest.raises(ValueError, match="requires calibration"):
        get_defense(DefenseConfig(enabled=True, type="rtc_v3_candidate"))

    manifest = build_manifest(params=_params())
    defense = get_defense(_v3_config(manifest))
    assert isinstance(defense, RTCv3Defense)
    assert defense.version == "rtc_v3"


def test_v3_rejects_unknown_runtime_and_manifest_fields():
    manifest = build_manifest(params=_params())
    with pytest.raises(ValueError, match="unknown custom parameters"):
        RTCv3Defense(_v3_config(manifest, mystery_switch=True))

    unknown_manifest = copy.deepcopy(manifest)
    unknown_manifest["mystery_budget"] = 1.0
    from defenses.rtc.calibration import content_hash

    unknown_manifest["content_hash"] = content_hash(unknown_manifest)
    with pytest.raises(ValueError, match="unknown fields"):
        CalibrationManifest.load(unknown_manifest)


def test_manifest_content_hash_is_fail_fast():
    manifest = build_manifest(params=_params())
    tampered = copy.deepcopy(manifest)
    tampered["clip_upper"] = 999.0

    with pytest.raises(ValueError, match="content hash mismatch"):
        CalibrationManifest.load(tampered)


def test_runtime_model_and_optimizer_mismatch_fail_before_aggregation():
    params = _params()
    manifest = build_manifest(params=params)
    defense = RTCv3Defense(_v3_config(manifest))

    with pytest.raises(ValueError, match="model_metadata_hash"):
        defense.set_context(
            1,
            ["a"],
            [np.zeros(4, dtype=np.float32)],
            server_optimizer="fedavg",
        )

    defense = RTCv3Defense(_v3_config(manifest))
    with pytest.raises(ValueError, match="server_optimizer"):
        defense.set_context(1, ["a"], params, server_optimizer="fedadam")


def test_phase0_never_silently_aggregates():
    params = _params()
    defense = RTCv3Defense(_v3_config(build_manifest(params=params)))
    defense.set_context(1, ["a"], params, server_optimizer="fedavg")

    with pytest.raises(RuntimeError, match="contract-only scaffold"):
        defense.aggregate([(params, 1)])


def test_normalized_auc_keeps_raw_compatibility():
    values = pd.Series([0.0, 0.5, 1.0])
    assert _auc(values) == pytest.approx(1.0)
    assert _normalized_auc(values) == pytest.approx(0.5)
    assert _normalized_auc(pd.Series([0.7])) == pytest.approx(0.7)

    rounds = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "planned_attack_active": [1.0, 1.0, 1.0],
            "server_asr": values,
            "server_accuracy": [0.9, 0.8, 0.7],
        }
    )
    summary = summarize_run(
        rounds,
        {
            "attack": "label_flip_targeted",
            "defense": "fedavg",
            "period": "continuous_1_0",
            "seed": 42,
            "malicious_fraction": 0.2,
            "attack_start_round": 1,
        },
    )
    assert summary["active_asr_auc"] == pytest.approx(1.0)
    assert summary["active_asr_auc_raw"] == pytest.approx(1.0)
    assert summary["active_asr_auc_normalized"] == pytest.approx(0.5)
