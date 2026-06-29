"""
attacks/attack_client.py
-------------------------
Malicious client variants for security experiment simulation.

Implemented attacks
-------------------
• label_flip        — flip source → target labels during local training
• backdoor          — stamp a pixel trigger + relabel to target class
• dba               — distributed backdoor with per-client trigger fragments
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


# ── Helper: dtype-safe parameter transforms ──────────────────────────────────

def _is_floating_array(param: np.ndarray) -> bool:
    return np.issubdtype(param.dtype, np.floating)


def _apply_to_floating_params(
    parameters: List[np.ndarray],
    fn,
) -> List[np.ndarray]:
    """Apply fn only to floating tensors and preserve non-floating buffers.

    PyTorch state dicts include integer scalar buffers such as BatchNorm
    ``num_batches_tracked``. Attacks should not randomize or scale those
    buffers; doing so can either crash on 0-D arrays or corrupt aggregation.
    """
    transformed: List[np.ndarray] = []
    for param in parameters:
        if _is_floating_array(param):
            arr = np.asarray(fn(param), dtype=param.dtype)
            transformed.append(arr.reshape(param.shape).astype(param.dtype, copy=False))
        else:
            transformed.append(param.copy())
    return transformed


def _scale_floating_update(
    global_params: List[np.ndarray],
    local_params: List[np.ndarray],
    boost_factor: float,
) -> List[np.ndarray]:
    """Scale floating model updates while preserving non-floating buffers."""
    scaled: List[np.ndarray] = []
    for global_param, local_param in zip(global_params, local_params):
        if _is_floating_array(global_param):
            arr = global_param + boost_factor * (local_param - global_param)
            scaled.append(arr.astype(global_param.dtype, copy=False))
        else:
            scaled.append(local_param.copy())
    return scaled


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
        poison_fraction = min(max(float(poison_fraction), 0.0), 1.0)
        n_poison = int(n * poison_fraction)
        if poison_fraction > 0.0 and n > 0:
            n_poison = max(1, n_poison)
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


def get_dba_trigger_coords(
    fragment_index: int,
    image_shape: Tuple[int, ...],
    trigger_size: int = 3,
    dba_trigger_num: int = 4,
    gap: int = 3,
    base_row: int = 0,
    base_col: int = 0,
) -> List[Tuple[int, int]]:
    """
    Return clipped pixel coordinates for one DBA trigger fragment.

    Fragments are laid out left-to-right from (base_row, base_col). The
    returned coordinates are spatial (row, col); all channels are stamped by
    DBADataset.
    """
    if len(image_shape) < 2:
        raise ValueError(f"DBA expects image tensors with spatial dims, got {image_shape}")

    height, width = image_shape[-2], image_shape[-1]
    if height <= 0 or width <= 0:
        raise ValueError(f"DBA expects positive spatial dims, got {image_shape}")

    trigger_size = max(1, int(trigger_size))
    fragment_index = int(fragment_index) % max(1, int(dba_trigger_num))
    gap = max(0, int(gap))

    start_row = int(base_row)
    start_col = int(base_col) + fragment_index * (trigger_size + gap)
    coords = []
    for row in range(start_row, start_row + trigger_size):
        for col in range(start_col, start_col + trigger_size):
            clipped_row = min(max(row, 0), height - 1)
            clipped_col = min(max(col, 0), width - 1)
            coords.append((clipped_row, clipped_col))
    return sorted(set(coords))


class DBADataset(Dataset):
    """
    Stamps one DBA trigger fragment and relabels poisoned samples to target.

    Each malicious client owns only one local fragment. The full trigger is
    formed across clients after aggregation, not inside a single client.
    """

    def __init__(
        self,
        base: Dataset,
        target_label: int,
        fragment_index: int,
        poison_fraction: float = 0.1,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        dba_trigger_num: int = 4,
        gap: int = 3,
        base_row: int = 0,
        base_col: int = 0,
        seed: int = 0,
    ):
        self.base = base
        self.target_label = target_label
        self.fragment_index = fragment_index
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.dba_trigger_num = dba_trigger_num
        self.gap = gap
        self.base_row = base_row
        self.base_col = base_col

        rng = np.random.default_rng(seed)
        n = len(base)  # type: ignore
        poison_fraction = min(max(float(poison_fraction), 0.0), 1.0)
        n_poison = int(n * poison_fraction)
        if poison_fraction > 0.0 and n > 0:
            n_poison = max(1, n_poison)
        self.poison_indices = set(rng.choice(n, n_poison, replace=False).tolist())

    def __len__(self): return len(self.base)  # type: ignore

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poison_indices:
            x = x.clone()
            coords = get_dba_trigger_coords(
                fragment_index=self.fragment_index,
                image_shape=tuple(x.shape),
                trigger_size=self.trigger_size,
                dba_trigger_num=self.dba_trigger_num,
                gap=self.gap,
                base_row=self.base_row,
                base_col=self.base_col,
            )
            for row, col in coords:
                x[..., row, col] = self.trigger_value
            y = self.target_label
        return x, y


# ── Attack client implementations ─────────────────────────────────────────────

class LabelFlipClient(FedSecClient):
    """Flips labels during local training."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if not getattr(self, "_attack_active", True):
            return
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
        if not getattr(self, "_attack_active", True):
            return
        poisoned_ds = BackdoorDataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
            poison_fraction=self.attack_cfg.poison_fraction,
            trigger_size=self.attack_cfg.trigger_size,
            trigger_value=self.attack_cfg.trigger_value,
            seed=self.client_id,
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.debug("Client %d: backdoor attack activated (target=%d)",
                     self.client_id, self.attack_cfg.backdoor_target_label)


class DBAClient(FedSecClient):
    """Distributed backdoor attack with per-client local trigger fragments."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if not getattr(self, "_attack_active", True):
            return
        self._global_params_cache = [p.copy() for p in parameters]
        fragment_index = self.client_id % max(1, self.attack_cfg.dba_trigger_num)
        poisoned_ds = DBADataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
            fragment_index=fragment_index,
            poison_fraction=self.attack_cfg.poison_fraction,
            trigger_size=self.attack_cfg.trigger_size,
            trigger_value=self.attack_cfg.trigger_value,
            dba_trigger_num=self.attack_cfg.dba_trigger_num,
            gap=self.attack_cfg.dba_gap,
            base_row=self.attack_cfg.dba_base_row,
            base_col=self.attack_cfg.dba_base_col,
            seed=self.client_id,
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.debug(
            "Client %d: DBA attack activated (fragment=%d target=%d)",
            self.client_id,
            fragment_index,
            self.attack_cfg.backdoor_target_label,
        )

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True) or not self.attack_cfg.dba_scale_update:
            return parameters

        global_params = getattr(self, "_global_params_cache", parameters)
        scaled = _scale_floating_update(
            global_params,
            parameters,
            self.attack_cfg.dba_boost_factor,
        )
        logger.debug(
            "Client %d: DBA update scaled (boost=%.1f)",
            self.client_id,
            self.attack_cfg.dba_boost_factor,
        )
        return scaled


class GaussianNoiseClient(FedSecClient):
    """Adds Gaussian noise to the model update before uploading."""

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True):
            return parameters
        attack_cfg = getattr(self, "attack_cfg", None)
        std = float(getattr(attack_cfg, "gaussian_noise_std", 0.1))
        noisy = _apply_to_floating_params(
            parameters,
            lambda p: p + np.random.normal(0, std, size=p.shape).astype(p.dtype),
        )
        logger.debug("Client %d: Gaussian noise injected (std=%.4f)", self.client_id, std)
        return noisy


class ByzantineClient(FedSecClient):
    """Sends completely random weights — worst-case adversary."""

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True):
            return parameters
        random_params = _apply_to_floating_params(
            parameters,
            lambda p: np.random.standard_normal(size=p.shape),
        )
        logger.debug("Client %d: Byzantine (random weights) attack", self.client_id)
        return random_params


class ModelReplacementClient(FedSecClient):
    """
    Model replacement attack (Bagdasaryan et al. 2020).
    Scales update so that aggregation results in the malicious model.
    Requires knowledge of the aggregation fraction (num_clients / clients_per_round).
    """

    def __init__(self, *args, boost_factor: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        configured = getattr(self.attack_cfg, "model_replacement_boost_factor", 10.0)
        self.boost_factor = float(configured if boost_factor is None else boost_factor)

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True):
            return parameters
        # Retrieve the global params that were set at fit start
        global_params = getattr(self, "_global_params_cache", parameters)
        # Compute update and amplify
        scaled = _scale_floating_update(global_params, parameters, self.boost_factor)
        logger.debug("Client %d: model replacement (boost=%.1f)",
                     self.client_id, self.boost_factor)
        return scaled

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if not getattr(self, "_attack_active", True):
            return
        # Cache the received global params
        self._global_params_cache = [p.copy() for p in parameters]
        # Also do backdoor data poisoning
        poisoned_ds = BackdoorDataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
            poison_fraction=1.0,   # all samples poisoned
            trigger_size=self.attack_cfg.trigger_size,
            trigger_value=self.attack_cfg.trigger_value,
            seed=self.client_id,
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
    "dba":               DBAClient,
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
