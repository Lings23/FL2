"""
attacks/attack_client.py
-------------------------
Malicious client variants for security experiment simulation.

Implemented attacks
-------------------
• label_flip        — flip source → target labels during local training
• backdoor          — stamp a pixel trigger + relabel to target class
• gaussian_noise    — add Gaussian noise to uploaded model weights
• model_replacement — scale update to replace global model (Bagdasaryan et al.)
• byzantine         — send random weights (worst-case adversary)

Extension interface
-------------------
1. Subclass FedSecClient and override on_before_fit() and/or on_after_fit()
2. Register in ATTACK_REGISTRY at the bottom
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from client.fl_client import FedSecClient
from config.config_loader import AttackConfig, ClientConfig, DPConfig
from models.model_factory import get_parameters, set_parameters

logger = logging.getLogger(__name__)


# ── Helper: poisoned data loaders ────────────────────────────────────────────

class LabelFlipDataset(Dataset):
    """Wraps a dataset and flips source_label → target_label."""

    def __init__(self, base: Dataset, source: int, target: int):
        self.base = base
        self.source = source
        self.target = target

    def __len__(self): return len(self.base)  # type: ignore

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.target if y == self.source else y


class BackdoorDataset(Dataset):
    """
    Stamps a small pixel-block trigger in the bottom-right corner
    and relabels triggered samples to target_label.

    poison_fraction: fraction of training samples that get triggered.
    """

    def __init__(
        self,
        base: Dataset,
        target_label: int,
        poison_fraction: float = 0.1,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        seed: int = 0,
    ):
        self.base = base
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        rng = np.random.default_rng(seed)
        n = len(base)  # type: ignore
        n_poison = max(1, int(n * poison_fraction))
        self.poison_indices = set(rng.choice(n, n_poison, replace=False).tolist())

    def __len__(self): return len(self.base)  # type: ignore

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poison_indices:
            x = x.clone()
            # Stamp trigger bottom-right
            x[..., -self.trigger_size:, -self.trigger_size:] = self.trigger_value
            y = self.target_label
        return x, y


# ── Attack client implementations ─────────────────────────────────────────────

class LabelFlipClient(FedSecClient):
    """Flips labels during local training."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        poisoned_ds = LabelFlipDataset(
            self.train_loader.dataset,
            source=self.attack_cfg.source_label,
            target=self.attack_cfg.target_label,
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.debug("Client %d: label flip %d→%d activated",
                     self.client_id,
                     self.attack_cfg.source_label,
                     self.attack_cfg.target_label)


class BackdoorClient(FedSecClient):
    """Injects backdoor trigger and trains with poisoned data."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        poisoned_ds = BackdoorDataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.debug("Client %d: backdoor attack activated (target=%d)",
                     self.client_id, self.attack_cfg.backdoor_target_label)


class GaussianNoiseClient(FedSecClient):
    """Adds Gaussian noise to the model update before uploading."""

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        noisy = [p + np.random.normal(0, 0.1, p.shape).astype(p.dtype)
                 for p in parameters]
        logger.debug("Client %d: Gaussian noise injected", self.client_id)
        return noisy


class ByzantineClient(FedSecClient):
    """Sends completely random weights — worst-case adversary."""

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        random_params = [np.random.randn(*p.shape).astype(p.dtype)
                         for p in parameters]
        logger.debug("Client %d: Byzantine (random weights) attack", self.client_id)
        return random_params


class ModelReplacementClient(FedSecClient):
    """
    Model replacement attack (Bagdasaryan et al. 2020).
    Scales update so that aggregation results in the malicious model.
    Requires knowledge of the aggregation fraction (num_clients / clients_per_round).
    """

    def __init__(self, *args, boost_factor: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.boost_factor = boost_factor

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        # Retrieve the global params that were set at fit start
        global_params = getattr(self, "_global_params_cache", parameters)
        # Compute update and amplify
        scaled = [
            global_params[i] + self.boost_factor * (parameters[i] - global_params[i])
            for i in range(len(parameters))
        ]
        logger.debug("Client %d: model replacement (boost=%.1f)",
                     self.client_id, self.boost_factor)
        return scaled

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        # Cache the received global params
        self._global_params_cache = [p.copy() for p in parameters]
        # Also do backdoor data poisoning
        poisoned_ds = BackdoorDataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
            poison_fraction=1.0,   # all samples poisoned
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )


# ── Registry ──────────────────────────────────────────────────────────────────

ATTACK_REGISTRY: Dict[str, Type[FedSecClient]] = {
    "label_flip":        LabelFlipClient,
    "backdoor":          BackdoorClient,
    "gaussian_noise":    GaussianNoiseClient,
    "byzantine":         ByzantineClient,
    "model_replacement": ModelReplacementClient,
    # ── Extension point ────────────────────────────────────────────
    # "your_attack": YourAttackClient,
}


def get_attack_client_class(attack_type: str) -> Type[FedSecClient]:
    t = attack_type.lower()
    if t not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack {t!r}. Available: {list(ATTACK_REGISTRY)}"
        )
    return ATTACK_REGISTRY[t]
