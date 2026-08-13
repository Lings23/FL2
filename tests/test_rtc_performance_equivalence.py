"""Semantic guards for the vectorized RTC-v3 server hot paths."""

from __future__ import annotations

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from defenses.rtc.calibration import CalibrationManifest, build_manifest
from defenses.rtc.sketch import (
    LEGACY_BLAKE2B_V1,
    SPLITMIX64_V1,
    clear_sketch_cache,
    signed_count_sketch,
    signed_count_sketch_reference,
    sketch_cache_stats,
)
from defenses.rtc.weighted_stats import (
    weighted_coordinate_median,
    weighted_coordinate_median_reference,
)
from defenses.rtc.v3 import RTCv3Defense


@pytest.mark.parametrize("clients", [3, 4, 9, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vectorized_coordinate_median_matches_reference_random(clients, dtype):
    rng = np.random.default_rng(20260808 + clients)
    arrays = [rng.normal(size=(7, 11)).astype(dtype) for _ in range(clients)]
    weights = rng.uniform(0.01, 2.0, size=clients)
    principals = [f"p{index // 2}" for index in range(clients)]
    client_ids = [f"c{clients - index:02d}" for index in range(clients)]

    actual = weighted_coordinate_median(
        arrays,
        weights,
        principal_ids=principals,
        client_ids=client_ids,
        block_size=13,
    )
    expected = weighted_coordinate_median_reference(
        arrays,
        weights,
        principal_ids=principals,
        client_ids=client_ids,
    )
    np.testing.assert_array_equal(actual, expected)


def test_coordinate_median_equal_mass_fast_path_uses_lower_observation():
    arrays = [
        np.asarray([0.0, 10.0]),
        np.asarray([2.0, 20.0]),
        np.asarray([4.0, 30.0]),
        np.asarray([6.0, 40.0]),
    ]
    actual = weighted_coordinate_median(arrays, [1.0] * 4, block_size=1)
    np.testing.assert_array_equal(actual, np.asarray([2.0, 20.0]))


def test_coordinate_median_ties_zero_mass_noncontiguous_and_extremes():
    base = np.asarray(
        [
            [0.0, -0.0, 1e-30, 1e30],
            [0.0, -0.0, 1e-20, 1e20],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )
    arrays = [base[:, ::2], base[::-1, ::2], np.zeros((3, 2), dtype=np.float32)]
    weights = [0.5, 0.5, 0.0]
    kwargs = {
        "principal_ids": ["same", "same", "zero"],
        "client_ids": ["z", "a", "ignored"],
    }
    actual = weighted_coordinate_median(arrays, weights, block_size=2, **kwargs)
    expected = weighted_coordinate_median_reference(arrays, weights, **kwargs)
    np.testing.assert_array_equal(actual, expected)


def test_coordinate_median_real_resnet_tensor_shape_matches_reference():
    # This is the shape of a ResNet-18 64x64 3x3 convolution kernel.
    rng = np.random.default_rng(18)
    shape = (64, 64, 3, 3)
    arrays = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    weights = [0.2, 0.5, 0.3]
    actual = weighted_coordinate_median(arrays, weights, block_size=4096)
    expected = weighted_coordinate_median_reference(arrays, weights)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "bad_arrays,bad_weights,error",
    [
        ([np.asarray([np.nan]), np.asarray([0.0])], [1.0, 1.0], ValueError),
        ([np.asarray([np.inf]), np.asarray([0.0])], [1.0, 1.0], ValueError),
        ([np.asarray([0.0]), np.asarray([1.0])], [0.0, 0.0], ValueError),
        ([np.asarray([0.0]), np.asarray([1.0])], [1.0, -1.0], ValueError),
        ([np.asarray([0]), np.asarray([1])], [1.0, 1.0], TypeError),
    ],
)
def test_coordinate_median_invalid_inputs_keep_failing(bad_arrays, bad_weights, error):
    with pytest.raises(error):
        weighted_coordinate_median(bad_arrays, bad_weights)


@pytest.mark.parametrize("dimension", [1, 7, 32, 256])
def test_cached_legacy_count_sketch_is_bitwise_reference_equivalent(dimension):
    rng = np.random.default_rng(123)
    base = rng.normal(size=(9, 12)).astype(np.float32)
    arrays = [base[:, ::2], base[::2, 1::2], np.zeros(5, dtype=np.float64)]
    expected = signed_count_sketch_reference(arrays, dimension=dimension, seed=-17)
    actual = signed_count_sketch(
        arrays,
        dimension=dimension,
        seed=-17,
        algorithm=LEGACY_BLAKE2B_V1,
    )
    np.testing.assert_array_equal(actual, expected)


def test_splitmix_count_sketch_is_deterministic_and_reuses_bounded_cache():
    clear_sketch_cache()
    arrays = [np.arange(1000, dtype=np.float64).reshape(20, 50)]
    first = signed_count_sketch(
        arrays, dimension=64, seed=42, algorithm=SPLITMIX64_V1
    )
    after_first = sketch_cache_stats()
    second = signed_count_sketch(
        arrays, dimension=64, seed=42, algorithm=SPLITMIX64_V1
    )
    after_second = sketch_cache_stats()

    np.testing.assert_array_equal(first, second)
    assert after_first["misses"] == 1.0
    assert after_second["hits"] == 1.0
    assert after_second["entries"] == 1.0
    assert 0.0 < after_second["bytes"] <= 256 * 1024 * 1024


@pytest.mark.parametrize("algorithm", [LEGACY_BLAKE2B_V1, SPLITMIX64_V1])
def test_count_sketch_rejects_nonfinite_values(algorithm):
    with pytest.raises(ValueError, match="finite"):
        signed_count_sketch(
            [np.asarray([0.0, np.nan])], algorithm=algorithm
        )


def test_count_sketch_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="unsupported"):
        signed_count_sketch([np.asarray([1.0])], algorithm="silent-migration")


def test_old_manifest_without_sketch_version_keeps_legacy_runtime():
    payload = build_manifest(params=[np.zeros(1, dtype=np.float32)])
    defense = RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": payload,
                "implementation_phase": 1,
            },
        )
    )
    assert defense.sketch_algorithm_version == LEGACY_BLAKE2B_V1


@pytest.mark.parametrize(
    "cones,match",
    [
        (
            {"enabled": True, "dimension": 32, "sketch_algorithm_version": "bad"},
            "unsupported sketch_algorithm_version",
        ),
        (
            {"enabled": True, "dimension": 32, "coordinate_median_block_size": 0},
            "coordinate_median_block_size",
        ),
    ],
)
def test_manifest_rejects_invalid_performance_contract(cones, match):
    payload = build_manifest(
        params=[np.zeros(1, dtype=np.float32)], stable_cones=cones
    )
    with pytest.raises(ValueError, match=match):
        CalibrationManifest.load(payload)
