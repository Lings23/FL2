"""
tests/test_framework.py
------------------------
Core unit tests — run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_dataset():
    """100 samples, 10 classes, 3×8×8 images."""
    x = torch.randn(100, 3, 8, 8)
    y = torch.randint(0, 10, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def dummy_params():
    """Simulate 3 clients each with a 2-layer param list."""
    rng = np.random.default_rng(0)
    return [
        [rng.random((64, 32)).astype(np.float32),
         rng.random((64,)).astype(np.float32)]
        for _ in range(5)
    ]


# ── Dataset partitioner tests ────────────────────────────────────────────────

class TestFederatedPartitioner:
    def test_iid_coverage(self, dummy_dataset):
        from data.dataset import FederatedPartitioner
        p = FederatedPartitioner(dummy_dataset, num_clients=5, strategy="iid", seed=0)
        all_idx = p.get_all_client_indices()
        combined = set(i for ids in all_idx.values() for i in ids)
        assert len(combined) == 100, "IID must cover all samples"
        assert all(len(ids) > 0 for ids in all_idx.values()), "All clients must have data"

    def test_iid_no_overlap(self, dummy_dataset):
        from data.dataset import FederatedPartitioner
        p = FederatedPartitioner(dummy_dataset, num_clients=5, strategy="iid", seed=0)
        all_idx = p.get_all_client_indices()
        flat = [i for ids in all_idx.values() for i in ids]
        assert len(flat) == len(set(flat)), "IID splits must not overlap"

    def test_dirichlet(self, dummy_dataset):
        from data.dataset import FederatedPartitioner
        p = FederatedPartitioner(dummy_dataset, num_clients=5, strategy="dirichlet",
                                 dirichlet_alpha=0.5, seed=0)
        all_idx = p.get_all_client_indices()
        assert len(all_idx) == 5

    def test_val_split(self, dummy_dataset):
        from data.dataset import FederatedPartitioner
        p = FederatedPartitioner(dummy_dataset, num_clients=4, strategy="iid",
                                 val_split=0.2, seed=0)
        train_sub, val_sub = p.get_client_data(0)
        assert len(train_sub) > 0
        assert len(val_sub) > 0
        assert len(train_sub) + len(val_sub) <= 30  # ~100/4


class TestDatasetLoading:
    def test_dataset_config_defaults_to_huggingface(self):
        from config.config_loader import DatasetConfig

        assert DatasetConfig().download_source == "huggingface"

    def test_get_dataset_accepts_torchvision_source(self, tmp_path):
        from data.dataset import CIFAR10Dataset, get_dataset

        dataset = get_dataset("cifar10", str(tmp_path), download_source="torchvision")
        assert isinstance(dataset, CIFAR10Dataset)
        assert dataset.download_source == "torchvision"

    def test_huggingface_cifar10_decodes_image_bytes(self):
        from data.dataset import HuggingFaceCIFAR10Dataset

        buf = BytesIO()
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buf, format="PNG")

        img = HuggingFaceCIFAR10Dataset._decode_image({"bytes": buf.getvalue()})

        assert img.mode == "RGB"
        assert img.size == (32, 32)

    def test_huggingface_cifar10_uses_local_tensor_cache(self, tmp_path):
        from data.dataset import HuggingFaceCIFAR10Dataset

        cache_dir = tmp_path / "huggingface_cifar10" / "cache"
        cache_dir.mkdir(parents=True)
        torch.save(
            {
                "version": 1,
                "images": torch.zeros((2, 32, 32, 3), dtype=torch.uint8),
                "labels": torch.tensor([3, 7], dtype=torch.long),
            },
            cache_dir / "train.pt",
        )

        dataset = HuggingFaceCIFAR10Dataset(tmp_path, split="train", transform=None)
        img, label = dataset[1]

        assert len(dataset) == 2
        assert dataset.targets == [3, 7]
        assert img.mode == "RGB"
        assert img.size == (32, 32)
        assert label == 7


class TestConfigLoading:
    def test_ray_config_defaults_and_override(self):
        from config.config_loader import Config, override_config

        cfg = Config()
        assert cfg.client.batch_size == 48
        assert cfg.federation.num_clients == 20
        assert cfg.federation.clients_per_round == 10
        assert cfg.federation.min_fit_clients == 10
        assert cfg.federation.min_available_clients == 20
        assert cfg.ray.client_num_cpus == 1.0
        assert cfg.ray.client_num_gpus == 0.0
        assert cfg.ray.force_cpu is False
        assert cfg.ray.log_to_driver is False
        assert cfg.ray.include_dashboard is False
        assert cfg.ray.object_store_memory_mb == 0
        assert cfg.ray.min_available_memory_mb == 0
        assert cfg.ray.memory_wait_seconds == pytest.approx(120.0)
        assert cfg.ray.memory_poll_seconds == pytest.approx(2.0)

        override_config(cfg, {
            "ray.client_num_gpus": 0.5,
            "ray.force_cpu": True,
            "ray.log_to_driver": True,
            "ray.object_store_memory_mb": 3072,
            "ray.min_available_memory_mb": 10240,
        })
        assert cfg.ray.client_num_gpus == 0.5
        assert cfg.ray.force_cpu is True
        assert cfg.ray.log_to_driver is True
        assert cfg.ray.object_store_memory_mb == 3072
        assert cfg.ray.min_available_memory_mb == 10240


# ── Model factory tests ───────────────────────────────────────────────────────

class TestModelFactory:
    def test_resnet18_output_shape(self):
        from models.model_factory import get_model
        model = get_model("resnet18", num_classes=10, dataset_name="cifar10")
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_lightcnn_mnist(self):
        from models.model_factory import get_model
        model = get_model("cnn", num_classes=10, dataset_name="mnist")
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_mlp(self):
        from models.model_factory import get_model
        model = get_model("mlp", num_classes=10, dataset_name="mnist")
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_get_set_parameters_roundtrip(self):
        from models.model_factory import get_model, get_parameters, set_parameters
        model = get_model("resnet18", num_classes=10)
        params = get_parameters(model)
        # Perturb
        noisy = [
            p + np.random.standard_normal(size=p.shape).astype(p.dtype) * 0.01
            if np.issubdtype(p.dtype, np.floating) else p.copy()
            for p in params
        ]
        set_parameters(model, noisy)
        recovered = get_parameters(model)
        for orig, rec in zip(noisy, recovered):
            np.testing.assert_allclose(orig, rec, rtol=1e-5)

    def test_unknown_architecture_raises(self):
        from models.model_factory import get_model
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model("transformer_xl")


# ── Defense tests ─────────────────────────────────────────────────────────────

def _make_updates(vectors: list[np.ndarray], n_samples: int = 100):
    """Build UpdateList from a list of flat vectors (single-param models)."""
    return [([v.reshape(v.shape)], n_samples) for v in vectors]


class TestDefenses:
    def _uniform_updates(self, n=5, dim=20):
        rng = np.random.default_rng(42)
        return [rng.random(dim).astype(np.float32) for _ in range(n)]

    def test_fedavg_weighted(self):
        from defenses.defense_base import FedAvgDefense
        from config.config_loader import DefenseConfig
        d = FedAvgDefense(DefenseConfig())
        vecs = self._uniform_updates(4, 10)
        updates = [([v], 10) for v in vecs]
        agg = d.aggregate(updates)
        expected = np.mean(vecs, axis=0)
        np.testing.assert_allclose(agg[0], expected, rtol=1e-5)

    def test_krum_selects_one(self):
        from defenses.defense_base import KrumDefense
        from config.config_loader import DefenseConfig
        cfg = DefenseConfig(krum_num_to_select=1)
        d = KrumDefense(cfg)
        vecs = self._uniform_updates(5, 20)
        # Make last vector a clear outlier
        vecs[-1] = np.ones(20, dtype=np.float32) * 100.0
        updates = [([v], 10) for v in vecs]
        agg = d.aggregate(updates)
        # Aggregated result should not be dominated by the outlier
        assert np.linalg.norm(agg[0]) < 50.0, "Krum should exclude outlier"

    def test_trimmed_mean_excludes_extremes(self):
        from defenses.defense_base import TrimmedMeanDefense
        from config.config_loader import DefenseConfig
        cfg = DefenseConfig(trim_fraction=0.2)
        d = TrimmedMeanDefense(cfg)
        vecs = [np.ones(10, dtype=np.float32) * i for i in range(10)]
        updates = [([v], 10) for v in vecs]
        agg = d.aggregate(updates)
        # Result should be between 2 and 7 (trimmed edges)
        assert 1.5 < agg[0].mean() < 8.0

    def test_median(self):
        from defenses.defense_base import MedianDefense
        from config.config_loader import DefenseConfig
        d = MedianDefense(DefenseConfig())
        vecs = [np.full(4, float(i), dtype=np.float32) for i in range(5)]
        updates = [([v], 10) for v in vecs]
        agg = d.aggregate(updates)
        np.testing.assert_allclose(agg[0], np.full(4, 2.0), atol=0.01)

    def test_fltrust_filters_orthogonal(self):
        from defenses.defense_base import FLTrustDefense
        from config.config_loader import DefenseConfig
        d = FLTrustDefense(DefenseConfig())
        global_params = [np.zeros(10, dtype=np.float32)]
        d.set_context(1, ["good-1", "bad", "good-2"], global_params)
        server_vec = np.ones(10, dtype=np.float32)
        d.set_server_update([server_vec])
        good = np.ones(10, dtype=np.float32)
        bad  = -np.ones(10, dtype=np.float32)
        updates = [([good], 10), ([bad], 10), ([good], 10)]
        agg = d.aggregate(updates)
        assert agg[0].mean() > 0, "FLTrust should upweight aligned updates"
        assert d.last_client_aggregation_weights["bad"] == pytest.approx(0.0)

    def test_krum_rejects_invalid_byzantine_bound(self):
        from defenses.defense_base import KrumDefense
        from config.config_loader import DefenseConfig
        d = KrumDefense(DefenseConfig(krum_num_malicious=2))
        updates = [([np.ones(4, dtype=np.float32) * idx], 10) for idx in range(5)]
        with pytest.raises(ValueError, match="n >= 2f"):
            d.aggregate(updates)

    def test_foolsgold_history_uses_stable_client_ids(self):
        from defenses.defense_base import FoolsGoldDefense
        from config.config_loader import DefenseConfig

        d = FoolsGoldDefense(DefenseConfig(), num_clients=3)
        global_params = [np.zeros(2, dtype=np.float32)]
        d.set_context(1, ["a", "b", "c"], global_params)
        d.aggregate(_make_updates([
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]))
        d.set_context(2, ["c", "a"], global_params)
        d.aggregate(_make_updates([
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]))

        np.testing.assert_allclose(d._history["a"], np.array([2.0, 0.0]))
        np.testing.assert_allclose(d._history["c"], np.array([0.0, 2.0]))

    def test_foolsgold_similar_clients_are_downweighted(self):
        from defenses.defense_base import FoolsGoldDefense
        from config.config_loader import DefenseConfig

        d = FoolsGoldDefense(DefenseConfig(custom_params={"use_num_examples": False}))
        d.set_context(1, ["sybil-a", "sybil-b", "honest"], [np.zeros(3, dtype=np.float32)])
        d.aggregate(_make_updates([
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ]))

        assert d.last_client_trusts["honest"] > d.last_client_trusts["sybil-a"]
        assert d.last_round_metrics["foolsgold_fallback"] == 0.0

    def test_fltrust_all_zero_trust_uses_server_update(self):
        from defenses.defense_base import FLTrustDefense
        from config.config_loader import DefenseConfig

        d = FLTrustDefense(DefenseConfig(custom_params={"zero_trust_fallback": "server_update"}))
        d.set_context(1, ["a", "b"], [np.zeros(2, dtype=np.float32)])
        d.set_server_update([np.ones(2, dtype=np.float32) * 0.25])
        result = d.aggregate(_make_updates([
            -np.ones(2, dtype=np.float32),
            -np.ones(2, dtype=np.float32) * 2,
        ]))

        np.testing.assert_allclose(result[0], np.full(2, 0.25, dtype=np.float32))
        assert d.last_round_metrics["fltrust_fallback"] == 1.0

    def test_freqfed_selects_largest_cluster(self, monkeypatch):
        from defenses.defense_base import FreqFedDefense
        from config.config_loader import DefenseConfig

        cfg = DefenseConfig(custom_params={
            "min_cluster_size": 2,
            "min_samples": 1,
            "min_selected_clients": 2,
            "projection_dim": 0,
        })
        d = FreqFedDefense(cfg)
        d.set_context(1, ["a", "b", "c", "outlier"], [np.zeros((2, 2), dtype=np.float32)])
        monkeypatch.setattr(
            d,
            "_cluster",
            lambda _: (np.array([0, 0, 0, -1]), np.array([0.9, 0.8, 0.7, 0.0])),
        )
        updates = [
            ([np.full((2, 2), value, dtype=np.float32)], 10)
            for value in (1.0, 1.1, 0.9, 100.0)
        ]
        result = d.aggregate(updates)

        assert float(result[0].mean()) == pytest.approx(1.0, abs=1e-6)
        assert d.last_client_aggregation_weights["outlier"] == 0.0
        assert d.last_round_metrics["freqfed_selected_clients"] == 3.0

    def test_freqfed_extracts_triangular_low_frequency_region(self):
        from defenses.defense_base import FreqFedDefense
        from config.config_loader import DefenseConfig

        d = FreqFedDefense(DefenseConfig(custom_params={
            "low_frequency_ratio": 0.5,
            "low_frequency_shape": "triangular",
            "projection_dim": 0,
        }))
        spectrum = np.arange(16, dtype=np.float32).reshape(4, 4)

        low = d._low_frequency_2d(spectrum)

        np.testing.assert_array_equal(
            low,
            np.array([0, 1, 2, 4, 5, 8], dtype=np.float32),
        )

    def test_freqfed_uses_cosine_distance_matrix(self):
        from defenses.defense_base import FreqFedDefense
        from config.config_loader import DefenseConfig

        d = FreqFedDefense(DefenseConfig())
        fingerprints = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        distances = d._cosine_distance_matrix(fingerprints)

        assert distances.shape == (3, 3)
        np.testing.assert_allclose(np.diag(distances), 0.0)
        np.testing.assert_allclose(distances, distances.T)
        assert distances[0, 1] == pytest.approx(1.0)
        assert distances[0, 2] == pytest.approx(1.0 - (1.0 / np.sqrt(2.0)))

    def test_freqfed_small_round_uses_median_fallback(self):
        from defenses.defense_base import FreqFedDefense
        from config.config_loader import DefenseConfig

        d = FreqFedDefense(DefenseConfig(custom_params={
            "min_cluster_size": 3,
            "fallback": "median",
        }))
        updates = _make_updates([
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32) * 10,
        ])
        result = d.aggregate(updates)

        np.testing.assert_allclose(result[0], np.full(2, 5.0, dtype=np.float32))
        assert d.last_round_metrics["freqfed_fallback"] == 1.0

    def test_freqfed_real_hdbscan_path(self):
        pytest.importorskip("hdbscan")
        from defenses.defense_base import FreqFedDefense
        from config.config_loader import DefenseConfig

        d = FreqFedDefense(DefenseConfig(custom_params={
            "min_cluster_size": 3,
            "min_samples": 2,
        }))
        rng = np.random.default_rng(4)
        first = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (12, 1))
        second = np.tile([0.0, 1.0, 0.0, 0.0, 0.0], (6, 1))
        vectors = np.vstack([
            first + rng.normal(0, 0.08, first.shape),
            second + rng.normal(0, 0.08, second.shape),
        ])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        labels, _ = d._cluster(vectors)

        assert len(set(labels[:12])) == 1
        assert len(set(labels[12:])) == 1
        assert labels[0] >= 0 and labels[12] >= 0 and labels[0] != labels[12]

    def test_fltrust_root_selection_is_balanced(self):
        from main import _select_root_indices

        targets = np.repeat(np.arange(3), 10)
        selected = _select_root_indices(30, 6, seed=7, targets=targets)
        counts = np.bincount(targets[selected], minlength=3)

        np.testing.assert_array_equal(counts, np.array([2, 2, 2]))


class TestTimeConsistencyDefense:
    def _defense(self, **custom_params):
        from config.config_loader import DefenseConfig
        from defenses.defense_base import get_defense

        defaults = {
            "projection_dim": 8,
            "windows": {"instant": 1, "short": 3, "mid": 4, "long": 6},
            "cold_start_rounds": 3,
            "offline_reset_rounds": 1,
            "min_effective_weight": 1e-4,
        }
        defaults.update(custom_params)
        return get_defense(
            DefenseConfig(enabled=True, type="time_consistency", custom_params=defaults),
            num_clients=5,
        )

    def _updates(self, vectors, samples=10):
        return [
            ([np.asarray(v, dtype=np.float32), np.array([idx], dtype=np.int64)], samples)
            for idx, v in enumerate(vectors)
        ]

    def test_cold_start_frequency_features_are_neutral(self):
        d = self._defense()
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]
        d.set_context(1, ["7"], global_params)
        d.aggregate(self._updates([np.zeros(4, dtype=np.float32)]))

        state = d._states["7"]
        np.testing.assert_allclose(state.feature_history[-1][4:], np.zeros(3), atol=1e-6)
        assert 0.0 <= state.final_trust <= 1.0

    def test_histories_are_keyed_by_real_client_id(self):
        d = self._defense()
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]

        d.set_context(1, ["1", "2"], global_params)
        d.aggregate(self._updates([np.ones(4), -np.ones(4)]))
        d.set_context(2, ["2"], global_params)
        d.aggregate(self._updates([-np.ones(4) * 0.5]))

        assert len(d._states["1"].signature_history) == 1
        assert len(d._states["2"].signature_history) == 2

    def test_offline_client_resets_after_threshold(self):
        d = self._defense(offline_reset_rounds=1)
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]

        d.set_context(1, ["1"], global_params)
        d.aggregate(self._updates([np.ones(4)]))
        d.set_context(3, ["1"], global_params)
        d.aggregate(self._updates([np.ones(4) * 2]))

        state = d._states["1"]
        assert state.participation_count == 1
        assert len(state.signature_history) == 1

    def test_single_zero_update_and_integer_buffer_are_safe(self):
        d = self._defense()
        global_params = [np.zeros(3, dtype=np.float32), np.array([2], dtype=np.int64)]
        d.set_context(1, ["solo"], global_params)
        aggregated = d.aggregate(self._updates([np.zeros(3, dtype=np.float32)]))

        np.testing.assert_allclose(aggregated[0], np.zeros(3), atol=1e-6)
        assert aggregated[1].dtype == np.int64
        assert aggregated[1][0] == 2

    def test_scaled_integer_buffers_do_not_change_signature_length(self):
        d = self._defense()
        global_params = [
            np.zeros(3, dtype=np.float32),
            np.array([2, 3, 4], dtype=np.int64),
        ]
        updates = [
            ([np.ones(3, dtype=np.float32), np.array([2, 3, 4], dtype=np.int64)], 10),
            ([np.ones(3, dtype=np.float32) * 2.0, np.array([2.0, 3.0, 4.0], dtype=np.float64)], 10),
        ]

        d.set_context(1, ["normal", "scaled-buffer"], global_params)
        aggregated = d.aggregate(updates)

        assert len(d._states["normal"].signature_history[-1]) == 3
        assert len(d._states["scaled-buffer"].signature_history[-1]) == 3
        assert aggregated[1].dtype == np.int64
        assert sum(d.last_client_aggregation_weights.values()) == pytest.approx(1.0)

    def test_outlier_update_is_softly_downweighted(self):
        d = self._defense()
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]
        updates = self._updates([
            np.ones(4, dtype=np.float32) * 0.10,
            np.ones(4, dtype=np.float32) * 0.11,
            np.ones(4, dtype=np.float32) * 10.0,
        ])
        d.set_context(1, ["good-a", "good-b", "bad"], global_params)
        d.aggregate(updates)

        assert d.last_client_weights["bad"] > 0.0
        assert d.last_client_weights["bad"] < d.last_client_weights["good-a"]
        assert d.last_round_metrics["time_consistency_trust_min"] <= d.last_round_metrics["time_consistency_trust_mean"]

    def test_model_replacement_delta_is_clipped(self):
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]
        vectors = [np.ones(4, dtype=np.float32) * 0.10 for _ in range(6)]
        vectors += [np.ones(4, dtype=np.float32) * 1.00 for _ in range(4)]
        updates = self._updates(vectors)
        weights = [1.0] * len(updates)

        clipped_defense = self._defense(
            enable_delta_clipping=True,
            enable_trust_caps=False,
            enable_exposure_budgets=False,
            norm_clip_factor=2.0,
        )
        clipped_defense.set_context(1, [str(i) for i in range(len(updates))], global_params)
        clipped = clipped_defense._aggregate_with_effective_weights(updates, weights)

        plain_defense = self._defense(
            enable_delta_clipping=False,
            enable_trust_caps=False,
            enable_exposure_budgets=False,
        )
        plain_defense.set_context(1, [str(i) for i in range(len(updates))], global_params)
        plain = plain_defense._aggregate_with_effective_weights(updates, weights)

        assert clipped_defense._last_clipped_clients == 4
        assert clipped_defense._last_clip_norm == pytest.approx(0.4, abs=1e-6)
        assert np.linalg.norm(clipped[0]) < np.linalg.norm(plain[0]) * 0.5

    def test_low_trust_clients_are_capped(self):
        d = self._defense(low_trust_weight_cap=0.01)
        constrained = d._apply_trust_weight_constraints(
            trust_scores=[0.30, 0.80, 0.80],
            effective_weights=[100.0, 1.0, 1.0],
        )
        assert constrained[0] <= 0.01 + 1e-9
        assert d._last_capped_clients == 1
        assert d._last_quarantined_clients == 0

    def test_quarantined_clients_get_zero_weight(self):
        d = self._defense(quarantine_threshold=0.25)
        constrained = d._apply_trust_weight_constraints(
            trust_scores=[0.20, 0.80, 0.80],
            effective_weights=[100.0, 1.0, 1.0],
        )
        normalized = np.asarray(constrained) / np.sum(constrained)

        assert normalized[0] == pytest.approx(0.0)
        assert d._last_quarantined_clients == 1

    def test_all_quarantined_falls_back_safely(self):
        d = self._defense(quarantine_threshold=0.9)
        constrained = d._apply_trust_weight_constraints(
            trust_scores=[0.10, 0.20, 0.30],
            effective_weights=[1.0, 2.0, 3.0],
        )

        assert np.asarray(constrained).sum() == pytest.approx(0.0)
        assert d._last_quarantined_clients == 3

    def test_periodic_magnitude_pattern_produces_frequency_signal(self):
        d = self._defense(min_temporal_points=4, min_periodic_events=4)
        state = d._get_state("periodic")
        state.rounds.extend([1, 3, 5])
        state.risk_history.extend([0.9, 0.9, 0.9])

        risk, dominant = d.scorer._periodic_gap_risk(state, 0.9, 7)

        assert risk > 0.0
        assert dominant == pytest.approx(0.5)

    def test_logs_client_weights_each_round(self, caplog):
        import logging

        d = self._defense()
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]
        updates = self._updates([
            np.ones(4, dtype=np.float32) * 0.10,
            np.ones(4, dtype=np.float32) * 10.0,
        ])

        with caplog.at_level(logging.INFO, logger="defenses.time_consistency_defense"):
            d.set_context(1, ["benign", "suspect"], global_params)
            d.aggregate(updates)

        messages = [record.getMessage() for record in caplog.records]
        assert any("TimeConsistency round 1 client weights" in m for m in messages)
        assert any("cid=benign" in m and "aggregation_weight=" in m for m in messages)
        assert any("cid=suspect" in m and "effective_weight=" in m for m in messages)
        assert sum(d.last_client_aggregation_weights.values()) == pytest.approx(1.0)

    def test_soft_weighting_can_be_disabled_for_clip_only_ablation(self):
        d = self._defense(
            enable_soft_trust_weighting=False,
            enable_delta_clipping=False,
            enable_trust_caps=False,
        )
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]
        d.set_context(1, ["a", "b"], global_params)
        d.aggregate(self._updates([
            np.ones(4, dtype=np.float32) * 0.1,
            np.ones(4, dtype=np.float32) * 10.0,
        ]))

        assert d.last_client_aggregation_weights["a"] == pytest.approx(0.5)
        assert d.last_client_aggregation_weights["b"] == pytest.approx(0.5)



# ── Config loader tests ───────────────────────────────────────────────────────

class TestConfigLoader:
    def test_load_default_config(self, tmp_path):
        import shutil
        from config.config_loader import load_config
        shutil.copy("config/config.yaml", tmp_path / "config.yaml")
        cfg = load_config(tmp_path / "config.yaml")
        assert cfg.federation.num_rounds == 50
        assert cfg.dataset.name == "cifar10"
        assert cfg.security.attack.model_replacement_boost_factor == pytest.approx(10.0)
        assert cfg.security.attack.gaussian_noise_std == pytest.approx(0.1)
        assert cfg.security.defense.krum_num_malicious == 1

    def test_override_config(self, tmp_path):
        import shutil
        from config.config_loader import load_config, override_config
        shutil.copy("config/config.yaml", tmp_path / "config.yaml")
        cfg = load_config(tmp_path / "config.yaml")
        cfg = override_config(cfg, {
            "federation.num_rounds": 99,
            "security.defense.custom_params.enable_delta_clipping": False,
        })
        assert cfg.federation.num_rounds == 99
        assert cfg.security.defense.custom_params["enable_delta_clipping"] is False

    def test_missing_config_raises(self):
        from config.config_loader import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestPeriodicDefenseAblations:
    @staticmethod
    def _defense(**custom_params):
        from config.config_loader import DefenseConfig
        from defenses.defense_base import get_defense
        defaults = {"projection_dim": 8, "windows": {
            "instant": 1, "short": 3, "mid": 4, "long": 6,
        }}
        defaults.update(custom_params)
        return get_defense(DefenseConfig(type="time_consistency", custom_params=defaults))

    def test_temporal_ablation_zeroes_temporal_risk(self):
        d = self._defense(enable_fft_features=False)
        state = d._get_state("client")
        state.rounds.extend([1, 3, 5])
        state.risk_history.extend([0.9, 0.9, 0.9])

        risk, dominant_frequency = d.scorer._temporal_risk(
            type("Record", (), {"norm": 2.0, "impact_proxy": 1.0, "total_risk": 0.9})(),
            state,
            7,
        )

        assert risk == 0.0
        assert dominant_frequency == 0.0

    def test_direction_ablation_zeroes_direction_feature(self):
        d = self._defense(enable_direction_features=False)
        state = d._get_state("client")
        state.signature_history.append(np.ones(4, dtype=np.float32))
        risk = d.scorer._direction_risk(
            type("Record", (), {"signature": -np.ones(4, dtype=np.float32)})(),
            state,
            np.ones(4, dtype=np.float32),
        )
        assert risk == 0.0

    def test_repeated_high_risk_moves_client_to_restricted(self):
        d = self._defense(high_risk_window=3)
        state = d._get_state("client")
        state.risk_history.extend([0.8])
        record = type("Record", (), {
            "total_risk": 0.8,
            "temporal_risk": 0.0,
            "influence_risk": 0.0,
            "state": "normal",
            "trust": 1.0,
            "quarantined": False,
        })()

        d.scorer.update_state_and_trust(record, state, d.params)

        assert record.state == "quarantined"
        assert 0.0 < record.trust < 1.0

    @pytest.mark.parametrize("values, expected", [
        ([1.0, 2.0] * 3, 0.5),
        ([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], 1.0 / 6.0),
    ])
    def test_fft_identifies_configured_periods(self, values, expected):
        d = self._defense(enable_fft_features=True)
        _, _, _, dominant_frequency = d._frequency_features(values)
        assert dominant_frequency == pytest.approx(expected)

    def test_periodic_gap_risk_uses_true_round_intervals(self):
        d = self._defense(enable_fft_features=True, min_periodic_events=4)
        state = d._get_state("client")
        state.rounds.extend([2, 6, 10])
        state.risk_history.extend([0.9, 0.9, 0.9])

        risk, dominant_frequency = d.scorer._periodic_gap_risk(state, 0.9, 14)

        assert risk == pytest.approx(1.0)
        assert dominant_frequency == pytest.approx(0.25)

    def test_current_event_completes_periodic_sequence_during_scoring(self):
        d = self._defense(min_periodic_events=4, high_risk_threshold=0.7)
        state = d._get_state("client")
        state.rounds.extend([2, 6, 10])
        state.risk_history.extend([0.9, 0.9, 0.9])
        record = type("Record", (), {
            "norm": 1.0,
            "impact_proxy": 1.0,
            "total_risk": 0.0,
        })()

        risk, frequency = d.scorer._temporal_risk(
            record, state, 14, current_event_risk=0.9
        )

        assert risk == pytest.approx(1.0)
        assert frequency == pytest.approx(0.25)

    def test_temporal_risk_does_not_duplicate_nonperiodic_bursts(self):
        d = self._defense(min_periodic_events=4)
        state = d._get_state("client")
        state.rounds.extend([1, 2, 4])
        state.event_risk_history.extend([0.9, 0.9, 0.9])
        record = type("Record", (), {
            "norm": 100.0, "impact_proxy": 100.0, "total_risk": 0.0,
        })()

        risk, frequency = d.scorer._temporal_risk(
            record, state, 8, current_event_risk=0.9
        )

        assert risk == 0.0
        assert frequency == 0.0

    def test_old_periodic_events_do_not_latch_on_low_current_risk(self):
        d = self._defense(min_periodic_events=4)
        state = d._get_state("client")
        state.rounds.extend([2, 4, 6, 8])
        state.event_risk_history.extend([0.9, 0.9, 0.9, 0.9])

        risk, frequency = d.scorer._periodic_gap_risk(state, 0.2, 9)

        assert risk == 0.0
        assert frequency == 0.0

    def test_quarantine_cooldown_uses_server_round(self):
        d = self._defense(quarantine_cooldown_rounds=2)
        state = d._get_state("client")
        triggering = type("Record", (), {
            "total_risk": 0.95,
            "temporal_risk": 0.9,
            "influence_risk": 0.0,
            "state": "normal",
            "trust": 1.0,
            "quarantined": False,
        })()
        d.scorer.update_state_and_trust(triggering, state, d.params, server_round=10)
        assert state.cooldown_until == 12

        recovering = type("Record", (), {
            "total_risk": 0.0,
            "temporal_risk": 0.0,
            "influence_risk": 0.0,
            "state": "normal",
            "trust": 1.0,
            "quarantined": False,
        })()
        d.scorer.update_state_and_trust(recovering, state, d.params, server_round=12)
        assert recovering.state == "quarantined"
        d.scorer.update_state_and_trust(recovering, state, d.params, server_round=13)
        assert recovering.state == "quarantined"
        d.scorer.update_state_and_trust(recovering, state, d.params, server_round=14)
        assert recovering.state == "restricted"
        d.scorer.update_state_and_trust(recovering, state, d.params, server_round=15)
        assert recovering.state == "watch"
        d.scorer.update_state_and_trust(recovering, state, d.params, server_round=16)
        assert recovering.state == "normal"


class TestPeriodicAttackExperiment:
    def test_attack_schedule_starts_after_warmup(self):
        from experiments.periodic_attack import attack_active
        assert [attack_active(rnd, 11, 1, 1) for rnd in range(9, 15)] == [
            False, False, True, False, True, False,
        ]
        assert [attack_active(rnd, 11, 3, 3) for rnd in range(11, 18)] == [
            True, True, True, False, False, False, True,
        ]
        assert [attack_active(rnd, 11, 1, 0, end=14) for rnd in range(9, 17)] == [
            False, False, True, True, True, True, False, False,
        ]
        assert [attack_active(rnd, 3, 1, 1, end=6) for rnd in range(2, 9)] == [
            False, True, False, True, False, False, False,
        ]

    def test_screening_filters_and_records_federation_condition(self):
        from argparse import Namespace
        from experiments.periodic_attack import build_matrix, select_matrix
        args = Namespace(
            attacks="model_replacement", defenses="rtc_full,clip_only",
            periods="long_3_3", seeds="42", malicious_fractions="0.2", smoke=False,
            attack_start_round=5, attack_end_round=14,
            partition="dirichlet", dirichlet_alpha=0.3,
            participation_rate=0.5, boost_factor=5.0,
        )

        selected = select_matrix(build_matrix("main"), args)

        assert len(selected) == 2
        assert {row["defense"] for row in selected} == {"rtc_full", "clip_only"}
        assert all(row["partition"] == "dirichlet" for row in selected)
        assert all(row["attack_end_round"] == 14 for row in selected)
        assert all(row["boost_factor"] == 5.0 for row in selected)

    def test_preregistered_matrix_sizes_and_deduplication(self):
        from experiments.periodic_attack import build_matrix
        assert len(build_matrix("main")) == 288
        assert len(build_matrix("ablation")) == 300
        assert len(build_matrix("untargeted")) == 216
        assert len(build_matrix("all")) == 504

    def test_label_flip_specs_carry_protocol_but_not_boost(self):
        from argparse import Namespace
        from experiments.periodic_attack import build_matrix, select_matrix

        args = Namespace(
            attacks="label_flip_targeted", defenses="fedavg",
            periods="short_1_1", seeds="42", malicious_fractions="0.2",
            smoke=False, attack_start_round=11, attack_end_round=-1,
            partition="iid", dirichlet_alpha=0.5, participation_rate=0.5,
            boost_factor=99.0, label_flip_source_label=5,
            label_flip_target_label=3, label_flip_poison_fraction=1.0,
        )

        selected = select_matrix(build_matrix("main"), args)

        assert len(selected) == 1
        assert selected[0]["label_flip_source_label"] == 5
        assert selected[0]["label_flip_target_label"] == 3
        assert selected[0]["label_flip_poison_fraction"] == pytest.approx(1.0)
        assert "boost_factor" not in selected[0]

    def test_smoke_matrix_excludes_legacy_rtc_v2_ablations(self):
        from experiments.periodic_attack import build_matrix, clean_baselines
        matrix = build_matrix(smoke=True)
        defenses = {row["defense"] for row in matrix}
        assert defenses == {"fedavg", "rtc_full", "clip_only", "freqfed"}
        assert len(clean_baselines(matrix, smoke=True)) == 3

    def test_round_summary_splits_active_and_residual_asr(self):
        from experiments.periodic_attack import summarize_run, tracker_to_rounds
        spec = {"attack": "backdoor", "period": "short_1_1", "on_rounds": 1,
                "off_rounds": 1, "malicious_fraction": 0.2, "seed": 42,
                "defense": "rtc_full", "defense_type": "time_consistency",
                "custom_params": {}, "attack_start_round": 3}
        records = []
        for rnd, asr in enumerate([0.0, 0.0, 0.8, 0.2, 0.6, 0.1], 1):
            records.append({"round": rnd, "split": "server", "accuracy": 0.8, "asr": asr})
            records.append({"round": rnd, "split": "fit", "active_attacker_weight_share": 0.1})
        result = summarize_run(tracker_to_rounds(records, spec), spec)
        assert result["active_asr"] == pytest.approx(0.7)
        assert result["residual_asr"] == pytest.approx(0.15)

    def test_label_flip_summary_uses_summed_exposure_counts(self):
        from experiments.periodic_attack import summarize_run

        spec = {
            "attack": "label_flip_targeted", "period": "continuous_1_0",
            "on_rounds": 1, "off_rounds": 0, "malicious_fraction": 0.2,
            "seed": 42, "defense": "fedavg", "defense_type": "none",
            "custom_params": {}, "attack_start_round": 1,
            "label_flip_source_label": 5, "label_flip_target_label": 3,
            "label_flip_poison_fraction": 1.0,
        }
        rounds = pd.DataFrame({
            "round": [1, 2],
            "planned_attack_active": [1.0, 1.0],
            "server_accuracy": [0.2, 0.3],
            "server_asr": [0.5, 0.4],
            "fit_label_flip_poisoned_exposures": [10, 30],
            "fit_label_flip_eligible_exposures": [10, 30],
            "fit_selected_training_exposures": [100, 300],
            "fit_label_flip_active_malicious_exposure_rate": [0.2, 0.2],
        })

        result = summarize_run(rounds, spec)

        assert result["label_flip_poisoned_exposures"] == 40
        assert result["label_flip_global_exposure_rate"] == pytest.approx(0.1)
        assert result["label_flip_eligible_poison_rate"] == pytest.approx(1.0)
        assert result["label_flip_active_malicious_exposure_rate"] == pytest.approx(0.2)

    def test_all_reverse_summary_reports_objective_asr_without_reclassifying_attack(self):
        from experiments.periodic_attack import summarize_run

        spec = {
            "attack": "label_flip_all_reverse", "attack_group": "untargeted",
            "period": "continuous_1_0", "on_rounds": 1, "off_rounds": 0,
            "malicious_fraction": 0.2, "seed": 42, "defense": "rtc_v3",
            "defense_type": "rtc_v3_candidate", "custom_params": {},
            "attack_start_round": 1,
        }
        rounds = pd.DataFrame({
            "round": [1, 2],
            "planned_attack_active": [1.0, 1.0],
            "server_accuracy": [0.8, 0.7],
            "server_asr": [0.1, 0.2],
        })

        result = summarize_run(rounds, spec)

        assert result["active_asr"] == pytest.approx(0.15)
        assert result["active_asr_auc"] == pytest.approx(0.15)
        assert result["peak_asr"] == pytest.approx(0.2)
        assert result["attack_group"] == "untargeted"

    def test_bootstrap_summary_reports_freqfed_diagnostics(self):
        from experiments.periodic_attack import bootstrap_summary

        rows = []
        for seed, fallback, selected, rejected, noise, clusters, ratio, seconds in (
            (42, 0.0, 4.0, 6.0, 6.0, 1.0, 0.4, 3.0),
            (43, 0.2, 6.0, 4.0, 4.0, 2.0, 0.6, 5.0),
        ):
            rows.append({
                "attack": "none", "period": "clean", "malicious_fraction": 0.0,
                "partition": "iid", "dirichlet_alpha": 0.5,
                "participation_rate": 0.5, "boost_factor": 10.0,
                "defense": "freqfed", "seed": seed,
                "freqfed_fallback": fallback,
                "freqfed_selected_clients": selected,
                "freqfed_rejected_clients": rejected,
                "freqfed_noise_clients": noise,
                "freqfed_cluster_count": clusters,
                "freqfed_selected_ratio": ratio,
                "aggregation_time_seconds": seconds,
            })

        report = bootstrap_summary(pd.DataFrame(rows), seed=7).set_index("metric")
        expected = {
            "freqfed_fallback": 0.1,
            "freqfed_selected_clients": 5.0,
            "freqfed_rejected_clients": 5.0,
            "freqfed_noise_clients": 5.0,
            "freqfed_cluster_count": 1.5,
            "freqfed_selected_ratio": 0.5,
            "aggregation_time_seconds": 4.0,
        }
        assert set(expected) <= set(report.index)
        for metric, mean in expected.items():
            assert report.loc[metric, "mean"] == pytest.approx(mean)
            assert report.loc[metric, "count"] == 2

    def test_first_state_lag_uses_first_active_attack_round(self):
        from experiments.periodic_attack import first_state_lag
        rounds = pd.DataFrame({
            "round": [1, 2, 3, 4, 5, 6],
            "planned_attack_active": [0, 0, 1, 0, 1, 0],
            "fit_malicious_restricted_rate": [0, 0, 0, 0, 0.5, 0],
        })

        assert first_state_lag(rounds, "fit_malicious_restricted_rate") == 2.0

    def test_deterministic_sampling_is_seeded_by_round(self):
        from strategies.fed_strategy import FedSecStrategy
        class Client:
            def __init__(self, cid): self.cid = cid
        class Manager:
            def __init__(self): self.clients = {str(i): Client(str(i)) for i in range(10)}
            def all(self): return self.clients
        manager = Manager()
        first = FedSecStrategy._deterministic_sample(manager, 5, 5, 43)
        repeated = FedSecStrategy._deterministic_sample(manager, 5, 5, 43)
        other_round = FedSecStrategy._deterministic_sample(manager, 5, 5, 44)
        assert [client.cid for client in first] == [client.cid for client in repeated]
        assert [client.cid for client in first] != [client.cid for client in other_round]


class TestPeriodicExperimentV2:
    def test_target_acceptance_uses_mean_ratio_and_seed_direction(self):
        from experiments.periodic_attack import validate_target_acceptance
        rows = []
        for seed, rtc, fedavg, clip in [(42, 0.2, 1.0, 0.8), (43, 0.4, 1.0, 0.9)]:
            for defense, value in (("rtc_full", rtc), ("fedavg", fedavg), ("clip_only", clip)):
                rows.append({"attack": "model_replacement", "period": "short_1_1",
                             "malicious_fraction": 0.2, "partition": "iid",
                             "dirichlet_alpha": 0.5, "participation_rate": 0.5,
                             "boost_factor": 10.0, "seed": seed, "defense": defense,
                             "active_asr_auc": value, "benign_quarantine_rate": 0.0})
        gates = validate_target_acceptance(pd.DataFrame(rows))
        assert gates
        assert all(gate["passed"] for gate in gates)

    def test_target_acceptance_checks_clean_only_summary(self):
        from experiments.periodic_attack import validate_target_acceptance
        clean = pd.DataFrame([
            {"attack": "none", "seed": 42, "defense": "fedavg", "final_accuracy": 0.80},
            {"attack": "none", "seed": 42, "defense": "rtc_full", "final_accuracy": 0.78},
        ])

        gates = validate_target_acceptance(clean)

        assert [gate["gate"] for gate in gates] == ["target_clean_accuracy_drop"]
        assert gates[0]["passed"] is True

    def test_sampling_uses_partition_id_not_random_proxy_cid(self):
        from strategies.fed_strategy import FedSecStrategy

        class Client:
            def __init__(self, cid, partition_id):
                self.cid = cid
                self.partition_id = partition_id

        class Manager:
            def __init__(self, prefix):
                self.clients = {
                    f"{prefix}-{i}": Client(f"{prefix}-{i}", i) for i in reversed(range(10))
                }
            def all(self):
                return self.clients

        first = FedSecStrategy._deterministic_sample(Manager("node-a"), 5, 5, 99)
        second = FedSecStrategy._deterministic_sample(Manager("node-b"), 5, 5, 99)
        assert [client.partition_id for client in first] == [client.partition_id for client in second]

    def test_metric_tracker_preserves_decimal_experiment_name(self, tmp_path):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(str(tmp_path), "attack__m0.2__defense")
        tracker.log(round=1, accuracy=0.5)
        tracker.save()

        assert (tmp_path / "attack__m0.2__defense.csv").exists()
        assert (tmp_path / "attack__m0.2__defense.json").exists()
        assert not (tmp_path / "attack__m0.csv").exists()

    def test_freqfed_median_fallback_has_no_fake_client_weights(self):
        from config.config_loader import DefenseConfig
        from defenses.defense_base import FreqFedDefense

        defense = FreqFedDefense(DefenseConfig(custom_params={
            "min_cluster_size": 3, "fallback": "median",
        }))
        defense.set_context(1, ["a", "b"], [np.zeros(2, dtype=np.float32)])
        defense.aggregate(_make_updates([
            np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32),
        ]))

        assert defense.last_client_weights == {}
        assert defense.last_client_aggregation_weights == {}
        assert defense.last_round_metrics["aggregation_mode"] == "median_fallback"
        assert defense.last_round_metrics["freqfed_fallback_reason"] == "insufficient_valid_fingerprints"

    def test_freqfed_allow_single_cluster_is_forwarded(self, monkeypatch):
        import hdbscan
        from config.config_loader import DefenseConfig
        from defenses.defense_base import FreqFedDefense

        captured = {}
        class FakeClusterer:
            probabilities_ = np.ones(2)
            def __init__(self, **kwargs): captured.update(kwargs)
            def fit_predict(self, values): return np.zeros(len(values), dtype=np.int64)
        monkeypatch.setattr(hdbscan, "HDBSCAN", FakeClusterer)
        defense = FreqFedDefense(DefenseConfig(custom_params={
            "min_cluster_size": 2, "min_samples": 1, "allow_single_cluster": True,
        }))
        defense._cluster(np.eye(2, dtype=np.float32))
        assert captured["allow_single_cluster"] is True
        assert captured["metric"] == "precomputed"

    def test_sampling_manifest_mismatch_fails_gate(self, tmp_path):
        import json
        from experiments.periodic_attack import validate_sampling_manifests

        rounds = tmp_path / "rounds"
        raw = tmp_path / "raw"
        rounds.mkdir(); raw.mkdir()
        common = {"attack": "backdoor", "period": "short_1_1",
                  "malicious_fraction": 0.2, "seed": 42}
        for name, selected in (("a", "0,1"), ("b", "0,2")):
            pd.DataFrame([{**common, "round": 1, "defense": name,
                           "fit_selected_partition_ids": selected,
                           "fit_planned_partition_ids": selected}]).to_csv(
                rounds / f"{name}.csv", index=False
            )
        manifest = {"seed": 42, "sha256": "same"}
        (raw / "a_data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        gates = validate_sampling_manifests(rounds, raw)
        assert next(g for g in gates if g["gate"] == "selected_partition_ids_match")["passed"] is False

    def test_strict_attack_owned_clean_plans_are_not_cross_compared(self, tmp_path):
        import json
        from experiments.periodic_attack import validate_sampling_manifests
        from experiments.trial_plan import SCHEMA_VERSION, sha256_json

        rounds = tmp_path / "rounds"
        raw = tmp_path / "raw"
        plans = tmp_path / "plans"
        rounds.mkdir(); raw.mkdir(); plans.mkdir()
        for index, selected in enumerate((["0", "1"], ["2", "3"])):
            digest = f"fit-digest-{index}"
            payload = {
                "schema_version": SCHEMA_VERSION,
                "pairing_group_id": f"attack-owner-{index}",
                "sampling_protocol": "principal_uniform",
                "malicious_partition_ids": [],
                "rounds": [{
                    "round": 1,
                    "partition_ids": selected,
                    "fit_seed_digest": digest,
                }],
            }
            payload["trial_plan_hash"] = sha256_json(payload)
            plan_path = plans / f"plan-{index}.json"
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            joined = ",".join(selected)
            pd.DataFrame([{
                "attack": "none",
                "period": "clean",
                "malicious_fraction": 0.0,
                "seed": 42,
                "defense": "fedavg",
                "pairing_mode": "strict",
                "round": 1,
                "trial_plan_hash": payload["trial_plan_hash"],
                "trial_plan_path": str(plan_path),
                "fit_trial_plan_hash": payload["trial_plan_hash"],
                "fit_selected_partition_ids": joined,
                "fit_planned_partition_ids": joined,
                "fit_planned_partition_ids_json": json.dumps(selected),
                "fit_completed_partition_ids_json": json.dumps(selected),
                "fit_fit_seed_digest": digest,
            }]).to_csv(rounds / f"clean-{index}.csv", index=False)
            (raw / f"clean-{index}_data_manifest.json").write_text(
                json.dumps({"seed": 42, "sha256": "same-data"}),
                encoding="utf-8",
            )

        gates = validate_sampling_manifests(rounds, raw)
        values = {row["gate"]: row["passed"] for row in gates}
        assert values["selected_partition_ids_match"] is True
        assert values["trial_plan_hash_match"] is True
        assert values["trial_plan_sequence_match"] is True
        assert values["completed_sequence_equals_plan"] is True
        assert values["client_random_stream_match"] is True

    def test_small_sample_statistics_are_descriptive_only(self):
        from experiments.periodic_attack import statistical_tests

        rows = []
        for defense, value in (("rtc_full", 0.1), ("freqfed", 0.2)):
            rows.append({"attack": "backdoor", "period": "short_1_1",
                         "malicious_fraction": 0.2, "seed": 42,
                         "defense": defense, "active_asr_auc": value})
        result = statistical_tests(pd.DataFrame(rows))
        assert bool(result.iloc[0]["descriptive_only"])
        assert pd.isna(result.iloc[0]["p_value"])
        assert pd.isna(result.iloc[0]["p_holm"])


# ── Server ASR tests ──────────────────────────────────────────────────────────

class TargetModel(torch.nn.Module):
    def __init__(self, target: int = 2, num_classes: int = 4):
        super().__init__()
        self.target = target
        self.num_classes = num_classes

    def forward(self, x):
        logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        logits[:, self.target] = 1.0
        return logits


class TestMetricTracker:
    def test_saves_dedicated_client_metrics_csv(self, tmp_path):
        import csv
        from utils.metrics import MetricTracker

        tracker = MetricTracker(str(tmp_path), "instrumented")
        tracker.log(round=1, split="server", accuracy=0.8)
        tracker.log(
            round=1,
            split="client",
            cid="7",
            is_malicious=True,
            raw_delta_norm=10.0,
            clipped_delta_norm=2.0,
            aggregation_weight=0.1,
            impact_norm=0.2,
        )
        tracker.save()

        output = tmp_path / "instrumented_clients.csv"
        assert output.exists()
        with open(output, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["cid"] == "7"
        assert float(rows[0]["impact_norm"]) == pytest.approx(0.2)


class TestLabelFlipExposureMetrics:
    def test_raw_exposure_counts_are_summed_without_sample_reweighting(self):
        from strategies.fed_strategy import FedSecStrategy

        records = [
            {
                "is_malicious": True,
                "attack_active": True,
                "num_examples": 100,
                "local_epochs": 5,
                "label_flip_total_examples": 100,
                "label_flip_eligible_examples": 10,
                "label_flip_poisoned_examples": 10,
                "aggregation_weight": None,
                "impact_norm": None,
                "trust": None,
                "state": None,
                "clipped": False,
                "quarantined": False,
            },
            {
                "is_malicious": False,
                "attack_active": False,
                "num_examples": 300,
                "local_epochs": 5,
                "label_flip_total_examples": None,
                "label_flip_eligible_examples": None,
                "label_flip_poisoned_examples": None,
                "aggregation_weight": None,
                "impact_norm": None,
                "trust": None,
                "state": None,
                "clipped": False,
                "quarantined": False,
            },
        ]

        metrics = FedSecStrategy._security_round_metrics(records)

        assert metrics["label_flip_poisoned_exposures"] == 50
        assert metrics["label_flip_eligible_exposures"] == 50
        assert metrics["selected_training_exposures"] == 2000
        assert metrics["label_flip_global_exposure_rate"] == pytest.approx(0.025)
        assert metrics["label_flip_eligible_poison_rate"] == pytest.approx(1.0)
        assert metrics["label_flip_active_malicious_exposure_rate"] == pytest.approx(0.1)


class TestServerASR:
    def test_targeted_label_flip_metrics_use_source_only(self):
        from config.config_loader import AttackConfig
        from server.fl_server import label_flip_metrics_from_confusion

        confusion = np.asarray([
            [4, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 6, 0],
            [0, 0, 2, 2],
        ])
        cfg = AttackConfig(
            enabled=True,
            type="label_flip_targeted",
            source_label=3,
            target_label=2,
        )

        metrics = label_flip_metrics_from_confusion(confusion, cfg)

        assert metrics["asr"] == pytest.approx(0.5)
        assert metrics["source_recall"] == pytest.approx(0.5)
        assert metrics["target_precision"] == pytest.approx(0.75)
        assert metrics["asr_total"] == 4
        assert metrics["source_to_target_count"] == 2
        assert np.asarray(__import__("json").loads(metrics["confusion_matrix_json"])).sum() == 19

    def test_targeted_label_flip_rejects_missing_source_class(self):
        from config.config_loader import AttackConfig
        from server.fl_server import label_flip_metrics_from_confusion

        cfg = AttackConfig(
            enabled=True,
            type="label_flip_targeted",
            source_label=3,
            target_label=2,
        )
        confusion = np.eye(4, dtype=int)
        confusion[3] = 0
        with pytest.raises(RuntimeError, match="no source label 3"):
            label_flip_metrics_from_confusion(confusion, cfg)

    def test_all_reverse_reports_reverse_mapping_asr(self):
        from config.config_loader import AttackConfig
        from server.fl_server import label_flip_metrics_from_confusion

        cfg = AttackConfig(enabled=True, type="label_flip_all_reverse")
        confusion = np.asarray([
            [0, 0, 4],
            [0, 5, 0],
            [3, 0, 0],
        ])
        metrics = label_flip_metrics_from_confusion(confusion, cfg)

        assert metrics["macro_recall"] == pytest.approx(1 / 3)
        assert metrics["asr"] == pytest.approx(1.0)
        assert metrics["reverse_mapping_rate"] == pytest.approx(1.0)
        assert metrics["reverse_mapping_count"] == 12
        assert metrics["asr_total"] == 12

    def test_dba_full_trigger_stamps_all_fragments(self):
        from config.config_loader import AttackConfig
        from server.fl_server import stamp_dba_full_trigger
        from attacks.attack_client import get_dba_trigger_coords

        cfg = AttackConfig(
            enabled=True,
            type="dba",
            trigger_size=2,
            trigger_value=1.0,
            dba_trigger_num=3,
            dba_pattern_mode="legacy_blocks",
            dba_trigger_value_mode="scalar",
            dba_gap=1,
        )
        x = torch.zeros(1, 3, 8, 8)
        triggered = stamp_dba_full_trigger(x, cfg)
        stamped = {tuple(coord) for coord in torch.nonzero(triggered[0, 0] == 1.0).tolist()}
        expected = set().union(*[
            set(get_dba_trigger_coords(i, (3, 8, 8), trigger_size=2,
                                       dba_trigger_num=3, pattern_mode="legacy_blocks", gap=1))
            for i in range(3)
        ])
        assert stamped == expected
        assert torch.count_nonzero(triggered[..., -2:, -2:]) == 0

    def test_backdoor_trigger_stamps_bottom_right_only(self):
        from config.config_loader import AttackConfig
        from server.fl_server import stamp_backdoor_trigger

        cfg = AttackConfig(enabled=True, type="backdoor", trigger_size=2, trigger_value=1.0)
        x = torch.zeros(1, 3, 8, 8)
        triggered = stamp_backdoor_trigger(x, cfg)
        assert triggered[..., -2:, -2:].min().item() == pytest.approx(1.0)
        assert torch.count_nonzero(triggered[..., :-2, :]) == 0
        assert torch.count_nonzero(triggered[..., :, :-2]) == 0

    def test_targeted_asr_excludes_target_label_samples(self):
        from config.config_loader import AttackConfig
        from server.fl_server import evaluate_targeted_asr

        x = torch.zeros(4, 1, 8, 8)
        y = torch.tensor([0, 2, 1, 3], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        cfg = AttackConfig(
            enabled=True,
            type="backdoor",
            backdoor_target_label=2,
            trigger_size=2,
            trigger_value=1.0,
        )
        metrics = evaluate_targeted_asr(TargetModel(target=2), loader, torch.device("cpu"), cfg)
        assert metrics is not None
        assert metrics["asr_total"] == 3
        assert metrics["asr"] == pytest.approx(1.0)
        assert metrics["attack_type"] == "backdoor"

    def test_model_replacement_reports_backdoor_asr(self):
        from config.config_loader import AttackConfig
        from server.fl_server import evaluate_targeted_asr

        x = torch.zeros(3, 1, 8, 8)
        y = torch.tensor([0, 1, 3], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        cfg = AttackConfig(
            enabled=True,
            type="model_replacement",
            backdoor_target_label=2,
        )
        metrics = evaluate_targeted_asr(
            TargetModel(target=2), loader, torch.device("cpu"), cfg
        )

        assert metrics is not None
        assert metrics["asr"] == pytest.approx(1.0)
        assert metrics["attack_type"] == "model_replacement"


# ── Attack dataset tests ──────────────────────────────────────────────────────

class TestAttackDatasets:
    def test_targeted_label_flip_changes_only_source_samples(self):
        from attacks.attack_client import LabelFlipDataset

        base = TensorDataset(torch.zeros(8, 1), torch.tensor([5, 1, 5, 3, 5, 0, 4, 5]))
        ds = LabelFlipDataset(
            base,
            attack_type="label_flip_targeted",
            num_classes=10,
            source=5,
            target=3,
            poison_fraction=1.0,
            seed=7,
        )

        labels = [int(ds[i][1]) for i in range(len(ds))]
        assert labels == [3, 1, 3, 3, 3, 0, 4, 3]
        assert ds.eligible_examples == 4
        assert ds.poisoned_examples == 4

    def test_all_reverse_flips_every_selected_label(self):
        from attacks.attack_client import LabelFlipDataset

        labels = torch.arange(10)
        base = TensorDataset(torch.zeros(10, 1), labels)
        ds = LabelFlipDataset(
            base,
            attack_type="label_flip_all_reverse",
            num_classes=10,
            poison_fraction=1.0,
            seed=7,
        )

        assert [int(ds[i][1]) for i in range(len(ds))] == list(reversed(range(10)))
        assert ds.eligible_examples == 10
        assert ds.poisoned_examples == 10

    def test_partial_label_flip_is_deterministic(self):
        from attacks.attack_client import LabelFlipDataset

        base = TensorDataset(torch.zeros(20, 1), torch.full((20,), 5))
        kwargs = dict(
            attack_type="label_flip_targeted",
            num_classes=10,
            source=5,
            target=3,
            poison_fraction=0.25,
            seed=42,
        )
        left = LabelFlipDataset(base, **kwargs)
        right = LabelFlipDataset(base, **kwargs)

        assert left.poison_indices == right.poison_indices
        assert left.poisoned_examples == 5

    @pytest.mark.parametrize(
        "attack_type, kwargs, message",
        [
            ("label_flip_targeted", {"source": 3, "target": 3}, "must be different"),
            ("label_flip_targeted", {"source": -1, "target": 3}, "source_label"),
            ("label_flip_all_reverse", {"poison_fraction": 1.1}, "must be in"),
            ("label_flip", {}, "no longer supported"),
        ],
    )
    def test_invalid_label_flip_configuration_fails(self, attack_type, kwargs, message):
        from attacks.attack_client import LabelFlipDataset

        base = TensorDataset(torch.zeros(4, 1), torch.arange(4))
        with pytest.raises(ValueError, match=message):
            LabelFlipDataset(
                base,
                attack_type=attack_type,
                num_classes=4,
                **kwargs,
            )

    def test_backdoor_trigger_stamped(self, dummy_dataset):
        from attacks.attack_client import BackdoorDataset
        ds = BackdoorDataset(dummy_dataset, target_label=0, poison_fraction=1.0,
                             trigger_size=2, trigger_value=1.0)
        x, y = ds[0]
        assert y == 0, "All samples should be relabelled to target"
        assert x[..., -2:, -2:].min().item() == pytest.approx(1.0, abs=1e-5), \
            "Trigger pixels must be stamped"

    def test_dba_relabels_poisoned_samples(self):
        from attacks.attack_client import DBADataset
        base = TensorDataset(torch.zeros(4, 3, 8, 8), torch.ones(4, dtype=torch.long))
        ds = DBADataset(base, target_label=7, fragment_index=0,
                        poison_fraction=1.0, trigger_size=2, trigger_value=1.0,
                        pattern_mode="legacy_blocks", trigger_value_mode="scalar")
        _, y = ds[0]
        assert int(y) == 7, "DBA poisoned samples should use target label"

    def test_dba_fragment_indices_have_different_positions(self):
        from attacks.attack_client import get_dba_trigger_coords
        coords_0 = set(get_dba_trigger_coords(0, (3, 8, 8), trigger_size=2,
                                              dba_trigger_num=4, pattern_mode="legacy_blocks", gap=1))
        coords_1 = set(get_dba_trigger_coords(1, (3, 8, 8), trigger_size=2,
                                              dba_trigger_num=4, pattern_mode="legacy_blocks", gap=1))
        assert coords_0
        assert coords_1
        assert coords_0 != coords_1
        assert coords_0.isdisjoint(coords_1)

    def test_dba_local_fragment_only_stamps_own_coords(self):
        from attacks.attack_client import DBADataset, get_dba_trigger_coords
        base = TensorDataset(torch.zeros(1, 3, 8, 8), torch.zeros(1, dtype=torch.long))
        ds = DBADataset(base, target_label=3, fragment_index=1,
                        poison_fraction=1.0, trigger_size=2, trigger_value=1.0,
                        dba_trigger_num=4, pattern_mode="legacy_blocks",
                        trigger_value_mode="scalar", gap=1)
        x, _ = ds[0]
        stamped = {tuple(coord) for coord in torch.nonzero(x[0] == 1.0, as_tuple=False).tolist()}
        own = set(get_dba_trigger_coords(1, tuple(x.shape), trigger_size=2,
                                         dba_trigger_num=4, pattern_mode="legacy_blocks", gap=1))
        full = set().union(*[
            set(get_dba_trigger_coords(i, tuple(x.shape), trigger_size=2,
                                       dba_trigger_num=4, pattern_mode="legacy_blocks", gap=1))
            for i in range(4)
        ])
        assert stamped == own
        assert stamped != full

    def test_dba_scales_model_update(self):
        from attacks.attack_client import DBAClient
        client = DBAClient.__new__(DBAClient)
        client.client_id = 0
        client.attack_cfg = type("Cfg", (), {
            "dba_scale_update": True,
            "dba_boost_factor": 3.0,
        })()
        client._global_params_cache = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array(5, dtype=np.int64),
        ]
        local = [
            np.array([2.0, 4.0], dtype=np.float32),
            np.array(9, dtype=np.int64),
        ]
        scaled = client.on_after_fit(local, {})
        np.testing.assert_allclose(scaled[0], np.array([4.0, 8.0], dtype=np.float32))
        assert scaled[1].dtype == np.int64
        assert scaled[1].shape == ()
        assert scaled[1].item() == 9

    def test_gaussian_noise_preserves_scalar_int_buffers(self):
        from attacks.attack_client import GaussianNoiseClient

        client = GaussianNoiseClient.__new__(GaussianNoiseClient)
        client.client_id = 0
        params = [
            np.zeros((2, 2), dtype=np.float32),
            np.array(3, dtype=np.int64),
            np.array(1.0, dtype=np.float32),
        ]

        noisy = client.on_after_fit(params, {})

        assert noisy[0].dtype == np.float32
        assert noisy[0].shape == (2, 2)
        assert noisy[1].dtype == np.int64
        assert noisy[1].shape == ()
        assert noisy[1].item() == 3
        assert noisy[2].dtype == np.float32
        assert noisy[2].shape == ()

    def test_byzantine_preserves_scalar_int_buffers(self):
        from attacks.attack_client import ByzantineClient

        client = ByzantineClient.__new__(ByzantineClient)
        client.client_id = 0
        params = [
            np.zeros((2, 2), dtype=np.float32),
            np.array(3, dtype=np.int64),
            np.array(1.0, dtype=np.float32),
        ]

        random_params = client.on_after_fit(params, {})

        assert random_params[0].dtype == np.float32
        assert random_params[0].shape == (2, 2)
        assert random_params[1].dtype == np.int64
        assert random_params[1].shape == ()
        assert random_params[1].item() == 3
        assert random_params[2].dtype == np.float32
        assert random_params[2].shape == ()

    def test_model_replacement_preserves_scalar_int_buffers(self):
        from attacks.attack_client import ModelReplacementClient

        client = ModelReplacementClient.__new__(ModelReplacementClient)
        client.client_id = 0
        client.boost_factor = 4.0
        client._global_params_cache = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array(5, dtype=np.int64),
        ]
        local = [
            np.array([2.0, 4.0], dtype=np.float32),
            np.array(9, dtype=np.int64),
        ]

        scaled = client.on_after_fit(local, {})

        np.testing.assert_allclose(scaled[0], np.array([5.0, 10.0], dtype=np.float32))
        assert scaled[1].dtype == np.int64
        assert scaled[1].shape == ()
        assert scaled[1].item() == 9

    def test_dba_registered(self):
        from attacks.attack_client import DBAClient, get_attack_client_class
        assert get_attack_client_class("dba") is DBAClient


def test_aggregate_update_evidence_has_exact_norm_and_deterministic_sketch():
    from strategies.fed_strategy import aggregate_update_evidence

    previous = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array(7, dtype=np.int64),
    ]
    aggregated = [
        np.array([4.0, 6.0], dtype=np.float32),
        np.array(9, dtype=np.int64),
    ]
    norm, sketch = aggregate_update_evidence(previous, aggregated)
    repeated_norm, repeated = aggregate_update_evidence(previous, aggregated)
    assert norm == pytest.approx(5.0)
    assert repeated_norm == pytest.approx(norm)
    assert sketch.shape == (256,)
    assert np.linalg.norm(sketch) == pytest.approx(1.0)
    np.testing.assert_array_equal(sketch, repeated)
