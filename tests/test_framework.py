"""
tests/test_framework.py
------------------------
Core unit tests — run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

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
        server_vec = np.ones(10, dtype=np.float32)
        d.set_server_update([server_vec])
        good = np.ones(10, dtype=np.float32)
        bad  = -np.ones(10, dtype=np.float32)
        updates = [([good], 10), ([bad], 10), ([good], 10)]
        agg = d.aggregate(updates)
        assert agg[0].mean() > 0, "FLTrust should upweight aligned updates"


# ── Config loader tests ───────────────────────────────────────────────────────

class TestConfigLoader:
    def test_load_default_config(self, tmp_path):
        import shutil
        from config.config_loader import load_config
        shutil.copy("config/config.yaml", tmp_path / "config.yaml")
        cfg = load_config(tmp_path / "config.yaml")
        assert cfg.federation.num_rounds == 50
        assert cfg.dataset.name == "cifar10"

    def test_override_config(self, tmp_path):
        import shutil
        from config.config_loader import load_config, override_config
        shutil.copy("config/config.yaml", tmp_path / "config.yaml")
        cfg = load_config(tmp_path / "config.yaml")
        cfg = override_config(cfg, {"federation.num_rounds": 99})
        assert cfg.federation.num_rounds == 99

    def test_missing_config_raises(self):
        from config.config_loader import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


# ── Attack dataset tests ──────────────────────────────────────────────────────

class TestAttackDatasets:
    def test_label_flip(self, dummy_dataset):
        from attacks.attack_client import LabelFlipDataset
        ds = LabelFlipDataset(dummy_dataset, source=0, target=9)
        assert len(ds) == len(dummy_dataset)
        # At least one flip should have occurred
        flipped = sum(1 for i in range(len(ds))
                      if dummy_dataset[i][1] == 0 and ds[i][1] == 9)
        original_zeros = sum(1 for i in range(len(dummy_dataset))
                             if dummy_dataset[i][1] == 0)
        assert flipped == original_zeros

    def test_backdoor_trigger_stamped(self, dummy_dataset):
        from attacks.attack_client import BackdoorDataset
        ds = BackdoorDataset(dummy_dataset, target_label=0, poison_fraction=1.0,
                             trigger_size=2, trigger_value=1.0)
        x, y = ds[0]
        assert y == 0, "All samples should be relabelled to target"
        assert x[..., -2:, -2:].min().item() == pytest.approx(1.0, abs=1e-5), \
            "Trigger pixels must be stamped"
