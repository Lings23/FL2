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
        assert cfg.ray.client_num_cpus == 1.0
        assert cfg.ray.client_num_gpus == 0.0
        assert cfg.ray.log_to_driver is False

        override_config(cfg, {"ray.client_num_gpus": 0.5, "ray.log_to_driver": True})
        assert cfg.ray.client_num_gpus == 0.5
        assert cfg.ray.log_to_driver is True


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
        assert aggregated[1][0] == 0

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
            norm_clip_factor=2.0,
        )
        clipped_defense.set_context(1, [str(i) for i in range(len(updates))], global_params)
        clipped = clipped_defense._aggregate_with_effective_weights(updates, weights)

        plain_defense = self._defense(enable_delta_clipping=False, enable_trust_caps=False)
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
        normalized = np.asarray(constrained) / np.sum(constrained)

        assert normalized[0] <= 0.01 + 1e-9
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
        normalized = np.asarray(constrained) / np.sum(constrained)

        assert np.isfinite(normalized).all()
        assert normalized.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(normalized, np.array([1.0, 2.0, 3.0]) / 6.0)

    def test_periodic_magnitude_pattern_produces_frequency_signal(self):
        d = self._defense(windows={"instant": 1, "short": 3, "mid": 4, "long": 6})
        global_params = [np.zeros(4, dtype=np.float32), np.array([0], dtype=np.int64)]
        for rnd, scale in enumerate([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], start=1):
            d.set_context(rnd, ["periodic"], global_params)
            d.aggregate(self._updates([np.ones(4, dtype=np.float32) * scale]))

        state = d._states["periodic"]
        assert state.feature_history[-1][4] > 0.0
        assert state.feature_history[-1][5] > 0.0

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


class TestServerASR:
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
            dba_gap=1,
        )
        x = torch.zeros(1, 3, 8, 8)
        triggered = stamp_dba_full_trigger(x, cfg)
        stamped = {tuple(coord) for coord in torch.nonzero(triggered[0, 0] == 1.0).tolist()}
        expected = set().union(*[
            set(get_dba_trigger_coords(i, (3, 8, 8), trigger_size=2,
                                       dba_trigger_num=3, gap=1))
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

    def test_dba_relabels_poisoned_samples(self):
        from attacks.attack_client import DBADataset
        base = TensorDataset(torch.zeros(4, 3, 8, 8), torch.ones(4, dtype=torch.long))
        ds = DBADataset(base, target_label=7, fragment_index=0,
                        poison_fraction=1.0, trigger_size=2, trigger_value=1.0)
        _, y = ds[0]
        assert int(y) == 7, "DBA poisoned samples should use target label"

    def test_dba_fragment_indices_have_different_positions(self):
        from attacks.attack_client import get_dba_trigger_coords
        coords_0 = set(get_dba_trigger_coords(0, (3, 8, 8), trigger_size=2,
                                              dba_trigger_num=4, gap=1))
        coords_1 = set(get_dba_trigger_coords(1, (3, 8, 8), trigger_size=2,
                                              dba_trigger_num=4, gap=1))
        assert coords_0
        assert coords_1
        assert coords_0 != coords_1
        assert coords_0.isdisjoint(coords_1)

    def test_dba_local_fragment_only_stamps_own_coords(self):
        from attacks.attack_client import DBADataset, get_dba_trigger_coords
        base = TensorDataset(torch.zeros(1, 3, 8, 8), torch.zeros(1, dtype=torch.long))
        ds = DBADataset(base, target_label=3, fragment_index=1,
                        poison_fraction=1.0, trigger_size=2, trigger_value=1.0,
                        dba_trigger_num=4, gap=1)
        x, _ = ds[0]
        stamped = {tuple(coord) for coord in torch.nonzero(x[0] == 1.0, as_tuple=False).tolist()}
        own = set(get_dba_trigger_coords(1, tuple(x.shape), trigger_size=2,
                                         dba_trigger_num=4, gap=1))
        full = set().union(*[
            set(get_dba_trigger_coords(i, tuple(x.shape), trigger_size=2,
                                       dba_trigger_num=4, gap=1))
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
