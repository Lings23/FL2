"""Bootstrap-manifest command tests."""

from __future__ import annotations

import pytest

from defenses.rtc.calibration import CalibrationManifest, model_metadata_hash
from experiments.rtc_v3_manifest import (
    BOOTSTRAP_LIMIT,
    build_bootstrap_manifest,
    write_bootstrap_manifest,
)
from models.model_factory import get_model, get_parameters


def test_bootstrap_manifest_matches_configured_runtime_model():
    payload = build_bootstrap_manifest("config/config.yaml")
    manifest = CalibrationManifest.load(payload)
    model = get_model(
        architecture="resnet18",
        num_classes=10,
        pretrained=False,
        dataset_name="cifar10",
    )

    params = get_parameters(model)
    manifest.validate_runtime(
        model_hash=model_metadata_hash(params),
        server_optimizer="fedavg",
        validated_mass_policy="unit_principal",
        principal_mapping_version="v1",
    )
    assert params
    assert manifest.payload["metadata"]["formal_evaluation_ready"] is False
    assert manifest.payload["stable_cones"]["sketch_algorithm_version"] == (
        "splitmix64_v1"
    )
    assert manifest.payload["stable_cones"]["coordinate_median_block_size"] == 262144
    assert manifest.payload["server_budgets"]["full"]["8"]["total"] == pytest.approx(
        BOOTSTRAP_LIMIT
    )


def test_bootstrap_writer_does_not_overwrite_without_force(tmp_path):
    output = tmp_path / "rtc_v3_manifest.json"

    written = write_bootstrap_manifest(
        config_path="config/config.yaml",
        output_path=output,
    )

    assert CalibrationManifest.load(written).payload["profile"] == (
        "bootstrap_runtime_smoke"
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_bootstrap_manifest(
            config_path="config/config.yaml",
            output_path=output,
        )
