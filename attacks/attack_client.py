"""
attacks/attack_client.py
-------------------------
Malicious client variants for security experiment simulation.

Implemented attacks
-------------------
• label_flip_targeted   — flip source → target labels during local training
• label_flip_all_reverse — flip every selected label with y' = C - 1 - y
• backdoor          — stamp a pixel trigger + relabel to target class
• dba               — distributed backdoor with per-client trigger fragments
• gaussian_noise    — generic additive Gaussian perturbation of the uploaded update
• model_replacement — scale update to replace global model (Bagdasaryan et al.)
• mpaf              — fake-client drift toward a fixed low-accuracy base model
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
    scalable_indices: Optional[set[int]] = None,
) -> List[np.ndarray]:
    """Scale selected floating model deltas and preserve all other buffers."""
    scaled: List[np.ndarray] = []
    for index, (global_param, local_param) in enumerate(
        zip(global_params, local_params)
    ):
        if (
            _is_floating_array(global_param)
            and (scalable_indices is None or index in scalable_indices)
        ):
            arr = global_param + boost_factor * (local_param - global_param)
            scaled.append(arr.astype(global_param.dtype, copy=False))
        else:
            scaled.append(local_param.copy())
    return scaled


def _trainable_state_indices(model: nn.Module) -> set[int]:
    """Map registered parameters to their positions in state-dict order."""
    trainable_names = {str(name) for name, _ in model.named_parameters()}
    return {
        index
        for index, name in enumerate(model.state_dict().keys())
        if str(name) in trainable_names
    }


def _client_scalable_indices(client: FedSecClient) -> Optional[set[int]]:
    """Return a real client's trainable layout; allow lightweight test doubles."""
    model = getattr(client, "model", None)
    return _trainable_state_indices(model) if model is not None else None


# ── Helper: poisoned data loaders ────────────────────────────────────────────

LABEL_FLIP_ATTACKS = {"label_flip_targeted", "label_flip_all_reverse"}


def validate_label_flip_config(
    attack_cfg: AttackConfig,
    num_classes: int,
) -> None:
    """Fail fast on ambiguous or invalid label-flip configurations."""
    attack_type = str(attack_cfg.type).lower()
    if attack_type == "label_flip":
        raise ValueError(
            "Attack type 'label_flip' is no longer supported; use "
            "'label_flip_targeted' or 'label_flip_all_reverse'."
        )
    if attack_type not in LABEL_FLIP_ATTACKS:
        return

    fraction = float(attack_cfg.label_flip_poison_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("label_flip_poison_fraction must be in [0, 1]")
    if int(num_classes) < 2:
        raise ValueError("Label flip requires dataset.num_classes >= 2")
    if attack_type == "label_flip_targeted":
        source = int(attack_cfg.source_label)
        target = int(attack_cfg.target_label)
        if not 0 <= source < int(num_classes):
            raise ValueError(f"source_label must be in [0, {int(num_classes) - 1}]")
        if not 0 <= target < int(num_classes):
            raise ValueError(f"target_label must be in [0, {int(num_classes) - 1}]")
        if source == target:
            raise ValueError("source_label and target_label must be different")


def validate_attack_config(attack_cfg: AttackConfig, num_classes: int) -> None:
    """Validate the complete versioned attack contract before simulation."""
    from attacks.spec import get_attack_spec

    validate_label_flip_config(attack_cfg, num_classes)
    attack_type = str(attack_cfg.type).lower()
    get_attack_spec(attack_type)
    if not 0.0 <= float(attack_cfg.malicious_fraction) <= 1.0:
        raise ValueError("malicious_fraction must be in [0, 1]")
    if not 0.0 <= float(attack_cfg.poison_fraction) <= 1.0:
        raise ValueError("poison_fraction must be in [0, 1]")
    if float(attack_cfg.gaussian_noise_std) < 0.0:
        raise ValueError("gaussian_noise_std must be non-negative")
    if float(attack_cfg.random_noise_scale) < 0.0:
        raise ValueError("random_noise_scale must be non-negative")
    if str(attack_cfg.random_noise_distribution).lower() not in {
        "rademacher",
        "gaussian",
    }:
        raise ValueError(
            "random_noise_distribution must be rademacher or gaussian"
        )
    if float(attack_cfg.sign_flip_scale) < 0.0:
        raise ValueError("sign_flip_scale must be non-negative")
    if (
        not np.isfinite(float(attack_cfg.replacement_gain))
        or float(attack_cfg.replacement_gain) <= 0.0
    ):
        raise ValueError("replacement_gain must be finite and positive")
    if bool(attack_cfg.aggregation_aware_scaling) and attack_type not in {
        "dba", "model_replacement", "scaling_backdoor"
    }:
        raise ValueError("aggregation_aware_scaling is only valid for scaled backdoors")
    if float(attack_cfg.lie_z) < 0.0:
        raise ValueError("lie_z must be non-negative; zero selects automatic z")
    if str(attack_cfg.coordinated_attack_knowledge).lower() not in {
        "all_updates",
        "malicious_only",
    }:
        raise ValueError(
            "coordinated_attack_knowledge must be all_updates or malicious_only"
        )
    if str(attack_cfg.optimization_perturbation).lower() not in {
        "inverse_std",
        "inverse_unit",
        "inverse_sign",
    }:
        raise ValueError("unsupported optimization_perturbation")
    if float(attack_cfg.optimization_gamma_init) <= 0.0:
        raise ValueError("optimization_gamma_init must be positive")
    if float(attack_cfg.optimization_tolerance) <= 0.0:
        raise ValueError("optimization_tolerance must be positive")
    if int(attack_cfg.optimization_max_iterations) <= 0:
        raise ValueError("optimization_max_iterations must be positive")
    if (
        not np.isfinite(float(attack_cfg.optimization_gamma_max))
        or float(attack_cfg.optimization_gamma_max) <= 0.0
    ):
        raise ValueError("optimization_gamma_max must be finite and positive")
    if not 0.0 < float(attack_cfg.optimization_gamma_fraction) <= 1.0:
        raise ValueError("optimization_gamma_fraction must be in (0, 1]")
    if int(attack_cfg.dba_trigger_num) <= 0:
        raise ValueError("dba_trigger_num must be positive")
    if str(attack_cfg.dba_pattern_mode).lower() not in {
        "paper_cifar_1x6_2x2",
        "legacy_blocks",
    }:
        raise ValueError("unsupported dba_pattern_mode")
    if str(attack_cfg.dba_trigger_value_mode).lower() not in {
        "cifar10_normalized_white",
        "scalar",
    }:
        raise ValueError("unsupported dba_trigger_value_mode")
    if (
        str(attack_cfg.dba_pattern_mode).lower() == "paper_cifar_1x6_2x2"
        and int(attack_cfg.dba_trigger_num) != 4
    ):
        raise ValueError("paper_cifar_1x6_2x2 requires dba_trigger_num=4")
    if int(attack_cfg.trigger_size) <= 0:
        raise ValueError("trigger_size must be positive")
    if int(attack_cfg.dba_gap) < 0:
        raise ValueError("dba_gap must be non-negative")


class LabelFlipDataset(Dataset):
    """Deterministically poison a subset of labels in a local dataset."""

    def __init__(
        self,
        base: Dataset,
        *,
        attack_type: str,
        num_classes: int,
        poison_fraction: float = 1.0,
        source: int = 5,
        target: int = 3,
        seed: int = 0,
    ):
        self.base = base
        self.attack_type = str(attack_type).lower()
        self.num_classes = int(num_classes)
        self.source = int(source)
        self.target = int(target)

        validation_cfg = AttackConfig(
            enabled=True,
            type=self.attack_type,
            source_label=self.source,
            target_label=self.target,
            label_flip_poison_fraction=float(poison_fraction),
        )
        validate_label_flip_config(validation_cfg, self.num_classes)

        if self.attack_type == "label_flip_targeted":
            eligible = [
                idx for idx in range(len(base))
                if int(base[idx][1]) == self.source  # type: ignore[index]
            ]
        else:
            eligible = list(range(len(base)))  # type: ignore[arg-type]

        fraction = float(poison_fraction)
        n_poison = int(len(eligible) * fraction)
        if fraction > 0.0 and eligible:
            n_poison = max(1, n_poison)
        rng = np.random.default_rng(int(seed))
        selected = (
            rng.choice(np.asarray(eligible, dtype=np.int64), n_poison, replace=False)
            if n_poison > 0 else np.asarray([], dtype=np.int64)
        )
        self.poison_indices = {int(idx) for idx in selected.tolist()}
        self.eligible_examples = len(eligible)
        self.poisoned_examples = len(self.poison_indices)
        self.total_examples = len(base)  # type: ignore[arg-type]

    def __len__(self): return len(self.base)  # type: ignore

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y_int = int(y)
        if idx not in self.poison_indices:
            return x, y
        if self.attack_type == "label_flip_targeted":
            poisoned_label = self.target
        else:
            poisoned_label = self.num_classes - 1 - y_int
        if torch.is_tensor(y):
            return x, torch.as_tensor(poisoned_label, dtype=y.dtype, device=y.device)
        if isinstance(y, np.generic):
            return x, y.dtype.type(poisoned_label)
        return x, poisoned_label


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
    pattern_mode: str = "paper_cifar_1x6_2x2",
    gap: int = 3,
    base_row: int = 0,
    base_col: int = 0,
) -> List[Tuple[int, int]]:
    """
    Return clipped pixel coordinates for one DBA trigger fragment.

    ``paper_cifar_1x6_2x2`` reproduces the official CIFAR configuration:
    four 1x6 line fragments at rows 0/4 and columns 0--5/9--14.  The old
    left-to-right square layout remains available only as ``legacy_blocks``.
    Returned coordinates are spatial (row, col); all channels are stamped.
    """
    if len(image_shape) < 2:
        raise ValueError(f"DBA expects image tensors with spatial dims, got {image_shape}")

    height, width = image_shape[-2], image_shape[-1]
    if height <= 0 or width <= 0:
        raise ValueError(f"DBA expects positive spatial dims, got {image_shape}")

    trigger_size = max(1, int(trigger_size))
    trigger_num = max(1, int(dba_trigger_num))
    fragment_index = int(fragment_index) % trigger_num
    gap = max(0, int(gap))

    mode = str(pattern_mode).lower()
    if mode == "paper_cifar_1x6_2x2":
        if trigger_num != 4:
            raise ValueError("paper_cifar_1x6_2x2 requires dba_trigger_num=4")
        starts = (
            (int(base_row), int(base_col)),
            (int(base_row), int(base_col) + 6 + gap),
            (int(base_row) + 1 + gap, int(base_col)),
            (int(base_row) + 1 + gap, int(base_col) + 6 + gap),
        )
        start_row, start_col = starts[fragment_index]
        if start_row < 0 or start_col < 0 or start_row >= height or start_col + 5 >= width:
            raise ValueError(
                "paper_cifar_1x6_2x2 does not fit image shape "
                f"{image_shape} at base ({base_row}, {base_col}) with gap {gap}"
            )
        return [(start_row, col) for col in range(start_col, start_col + 6)]
    if mode != "legacy_blocks":
        raise ValueError(f"Unsupported DBA pattern mode: {pattern_mode!r}")

    start_row = int(base_row)
    start_col = int(base_col) + fragment_index * (trigger_size + gap)
    coords = []
    for row in range(start_row, start_row + trigger_size):
        for col in range(start_col, start_col + trigger_size):
            clipped_row = min(max(row, 0), height - 1)
            clipped_col = min(max(col, 0), width - 1)
            coords.append((clipped_row, clipped_col))
    return sorted(set(coords))


def stamp_dba_coords_(
    tensor: torch.Tensor,
    coords: List[Tuple[int, int]],
    *,
    trigger_value: float,
    value_mode: str,
) -> torch.Tensor:
    """Stamp DBA coordinates in-place and return ``tensor``.

    CIFAR-10 tensors in this project are normalized before DBADataset sees
    them.  The official raw white value therefore has to be transformed per
    RGB channel rather than written as a scalar 1.0.
    """
    mode = str(value_mode).lower()
    if mode == "scalar":
        value: float | torch.Tensor = float(trigger_value)
    elif mode == "cifar10_normalized_white":
        if tensor.dim() < 3 or int(tensor.shape[-3]) != 3:
            raise ValueError(
                "cifar10_normalized_white requires a channel-first RGB tensor"
            )
        means = tensor.new_tensor((0.4914, 0.4822, 0.4465))
        stds = tensor.new_tensor((0.2023, 0.1994, 0.2010))
        value = (tensor.new_tensor(1.0) - means) / stds
    else:
        raise ValueError(f"Unsupported DBA trigger value mode: {value_mode!r}")
    for row, col in coords:
        tensor[..., row, col] = value
    return tensor


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
        pattern_mode: str = "paper_cifar_1x6_2x2",
        trigger_value_mode: str = "cifar10_normalized_white",
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
        self.pattern_mode = pattern_mode
        self.trigger_value_mode = trigger_value_mode
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
                pattern_mode=self.pattern_mode,
                gap=self.gap,
                base_row=self.base_row,
                base_col=self.base_col,
            )
            stamp_dba_coords_(
                x,
                coords,
                trigger_value=self.trigger_value,
                value_mode=self.trigger_value_mode,
            )
            y = self.target_label
        return x, y


# ── Attack client implementations ─────────────────────────────────────────────

class LabelFlipClient(FedSecClient):
    """Flips labels during local training."""

    def _poisoned_dataset(self) -> LabelFlipDataset:
        seed = int(getattr(self, "_current_attack_seed", self.experiment_seed))
        cache = getattr(self, "_label_flip_datasets", {})
        cached = cache.get(seed)
        if cached is None:
            cached = LabelFlipDataset(
                self._clean_train_loader.dataset,
                attack_type=self.attack_cfg.type,
                num_classes=self.num_classes,
                poison_fraction=self.attack_cfg.label_flip_poison_fraction,
                source=self.attack_cfg.source_label,
                target=self.attack_cfg.target_label,
                seed=seed,
            )
            cache[seed] = cached
            self._label_flip_datasets = cache
        return cached

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if not getattr(self, "_attack_active", True):
            return
        poisoned_ds = self._poisoned_dataset()
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.debug(
            "Client %d: %s activated (eligible=%d poisoned=%d)",
            self.client_id,
            self.attack_cfg.type,
            poisoned_ds.eligible_examples,
            poisoned_ds.poisoned_examples,
        )

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        total = len(self._clean_train_loader.dataset)
        if getattr(self, "_attack_active", False):
            poisoned_ds = self._poisoned_dataset()
            metrics["label_flip_eligible_examples"] = poisoned_ds.eligible_examples
            metrics["label_flip_poisoned_examples"] = poisoned_ds.poisoned_examples
        else:
            metrics["label_flip_eligible_examples"] = 0
            metrics["label_flip_poisoned_examples"] = 0
        metrics["label_flip_total_examples"] = total
        return parameters


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
            seed=int(getattr(self, "_current_attack_seed", self.client_id)),
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
        configured_fragment = int(getattr(self.attack_cfg, "dba_fragment_index", -1))
        fragment_index = (
            configured_fragment
            if configured_fragment >= 0
            else self.client_id % max(1, self.attack_cfg.dba_trigger_num)
        )
        poisoned_ds = DBADataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
            fragment_index=fragment_index,
            poison_fraction=self.attack_cfg.poison_fraction,
            trigger_size=self.attack_cfg.trigger_size,
            trigger_value=self.attack_cfg.trigger_value,
            dba_trigger_num=self.attack_cfg.dba_trigger_num,
            pattern_mode=self.attack_cfg.dba_pattern_mode,
            trigger_value_mode=self.attack_cfg.dba_trigger_value_mode,
            gap=self.attack_cfg.dba_gap,
            base_row=self.attack_cfg.dba_base_row,
            base_col=self.attack_cfg.dba_base_col,
            seed=int(getattr(self, "_current_attack_seed", self.client_id)),
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self._dba_dataset = poisoned_ds
        self._dba_fragment_index = fragment_index
        logger.debug(
            "Client %d: DBA attack activated (fragment=%d target=%d)",
            self.client_id,
            fragment_index,
            self.attack_cfg.backdoor_target_label,
        )

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        aggregation_aware = bool(
            getattr(self.attack_cfg, "aggregation_aware_scaling", False)
        )
        if getattr(self, "_attack_active", True):
            poisoned_ds = getattr(self, "_dba_dataset", None)
            metrics["dba_fragment_index"] = int(
                getattr(self, "_dba_fragment_index", -1)
            )
            metrics["dba_poisoned_examples"] = int(
                len(getattr(poisoned_ds, "poison_indices", ()))
            )
            metrics["dba_total_examples"] = int(
                len(poisoned_ds) if poisoned_ds is not None else 0
            )
            metrics["dba_update_scaled"] = bool(self.attack_cfg.dba_scale_update)
            metrics["dba_scaling_mode"] = (
                "aggregation_weight_compensated"
                if aggregation_aware
                else "fixed_client_boost"
            )
            metrics["dba_replacement_gain"] = float(
                getattr(self.attack_cfg, "replacement_gain", 1.0)
            )
        if (
            not getattr(self, "_attack_active", True)
            or not self.attack_cfg.dba_scale_update
        ):
            return parameters

        # Exact FedAvg-weight compensation needs every selected client's
        # num_examples and is therefore applied once by AttackCoordinator.
        if aggregation_aware:
            return parameters

        global_params = getattr(self, "_global_params_cache", parameters)
        scaled = _scale_floating_update(
            global_params,
            parameters,
            self.attack_cfg.dba_boost_factor,
            _client_scalable_indices(self),
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
        mean = float(getattr(attack_cfg, "gaussian_noise_mean", 0.0))
        std = float(getattr(attack_cfg, "gaussian_noise_std", 0.1))
        noisy = _apply_to_floating_params(
            parameters,
            lambda p: p
            + np.random.normal(mean, std, size=p.shape).astype(p.dtype),
        )
        metrics["gaussian_noise_mean"] = mean
        metrics["gaussian_noise_std"] = std
        logger.debug(
            "Client %d: Gaussian noise injected (mean=%.4f std=%.4f)",
            self.client_id,
            mean,
            std,
        )
        return noisy


class RandomNoiseClient(FedSecClient):
    """Replaces the local floating update with norm-matched random noise."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if getattr(self, "_attack_active", True):
            self._global_params_cache = [parameter.copy() for parameter in parameters]

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True):
            return parameters
        global_params = getattr(self, "_global_params_cache", parameters)
        squared_norm = 0.0
        dimension = 0
        for global_param, local_param in zip(global_params, parameters):
            if _is_floating_array(global_param):
                delta = np.asarray(local_param, dtype=np.float64) - np.asarray(
                    global_param, dtype=np.float64
                )
                squared_norm += float(np.dot(delta.reshape(-1), delta.reshape(-1)))
                dimension += int(delta.size)
        reference_norm = float(np.sqrt(squared_norm))
        strength = float(getattr(self.attack_cfg, "random_noise_scale", 1.0))
        coordinate_scale = (
            strength * reference_norm / np.sqrt(float(dimension))
            if dimension > 0
            else 0.0
        )
        distribution = str(
            getattr(self.attack_cfg, "random_noise_distribution", "rademacher")
        ).lower()
        attacked: List[np.ndarray] = []
        for global_param, local_param in zip(global_params, parameters):
            if _is_floating_array(global_param):
                if distribution == "rademacher":
                    noise = np.random.choice((-1.0, 1.0), size=global_param.shape)
                elif distribution == "gaussian":
                    noise = np.random.standard_normal(size=global_param.shape)
                else:
                    raise ValueError(
                        "random_noise_distribution must be rademacher or gaussian"
                    )
                value = np.asarray(global_param, dtype=np.float64) + coordinate_scale * noise
                attacked.append(value.astype(global_param.dtype, copy=False))
            else:
                attacked.append(local_param.copy())
        metrics["random_noise_reference_norm"] = reference_norm
        metrics["random_noise_coordinate_scale"] = float(coordinate_scale)
        metrics["random_noise_strength"] = strength
        return attacked


class SignFlipClient(FedSecClient):
    """Reverses and optionally amplifies the client's local model delta."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if getattr(self, "_attack_active", True):
            self._global_params_cache = [parameter.copy() for parameter in parameters]

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True):
            return parameters
        scale = float(getattr(self.attack_cfg, "sign_flip_scale", 1.0))
        metrics["sign_flip_scale"] = scale
        return _scale_floating_update(
            getattr(self, "_global_params_cache", parameters),
            parameters,
            -scale,
        )


class CoordinatedAttackClient(FedSecClient):
    """Trains normally; the strategy applies the coordinated attack pre-defense."""

    pass


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
        attack_cfg = getattr(self, "attack_cfg", None)
        aggregation_aware = bool(
            getattr(attack_cfg, "aggregation_aware_scaling", False)
        )
        metrics["model_replacement_scaling_mode"] = (
            "aggregation_weight_compensated"
            if aggregation_aware
            else "fixed_client_boost"
        )
        metrics["model_replacement_gain"] = float(
            getattr(attack_cfg, "replacement_gain", 1.0)
        )
        if aggregation_aware:
            return parameters
        # Retrieve the global params that were set at fit start
        global_params = getattr(self, "_global_params_cache", parameters)
        # Compute update and amplify
        scaled = _scale_floating_update(
            global_params,
            parameters,
            self.boost_factor,
            _client_scalable_indices(self),
        )
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
            seed=int(getattr(self, "_current_attack_seed", self.client_id)),
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )


class ScalingBackdoorClient(ModelReplacementClient):
    """Train-and-scale backdoor with an explicit, configurable poison fraction."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if not getattr(self, "_attack_active", True):
            return
        self._global_params_cache = [parameter.copy() for parameter in parameters]
        poisoned_ds = BackdoorDataset(
            self.train_loader.dataset,
            target_label=self.attack_cfg.backdoor_target_label,
            poison_fraction=self.attack_cfg.poison_fraction,
            trigger_size=self.attack_cfg.trigger_size,
            trigger_value=self.attack_cfg.trigger_value,
            seed=int(getattr(self, "_current_attack_seed", self.client_id)),
        )
        self.train_loader = DataLoader(
            poisoned_ds,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )


class MPAFClient(FedSecClient):
    """MPAF-style fake client using only successive global models."""

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        if not getattr(self, "_attack_active", True):
            return
        self._global_params_cache = [parameter.copy() for parameter in parameters]
        if not hasattr(self, "_mpaf_base_params"):
            base_scale = float(getattr(self.attack_cfg, "mpaf_base_scale", 0.0))
            self._mpaf_base_params = _apply_to_floating_params(
                parameters,
                lambda parameter: base_scale * parameter,
            )

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        if not getattr(self, "_attack_active", True):
            return parameters
        global_params = getattr(self, "_global_params_cache", parameters)
        base_params = getattr(self, "_mpaf_base_params", global_params)
        attack_scale = float(getattr(self.attack_cfg, "mpaf_lambda", 1.0))
        squared_norm = 0.0
        raw_deltas: List[np.ndarray | None] = []
        for global_param, base_param in zip(global_params, base_params):
            if _is_floating_array(global_param):
                delta = attack_scale * (
                    np.asarray(base_param, dtype=np.float64)
                    - np.asarray(global_param, dtype=np.float64)
                )
                squared_norm += float(np.dot(delta.reshape(-1), delta.reshape(-1)))
                raw_deltas.append(delta)
            else:
                raw_deltas.append(None)
        norm = float(np.sqrt(squared_norm))
        ceiling = float(getattr(self.attack_cfg, "mpaf_max_update_norm", 0.0))
        applied_scale = (
            min(1.0, ceiling / norm)
            if ceiling > 0.0 and norm > 0.0
            else 1.0
        )
        attacked: List[np.ndarray] = []
        for global_param, delta in zip(global_params, raw_deltas):
            if delta is None:
                attacked.append(global_param.copy())
            else:
                value = (
                    np.asarray(global_param, dtype=np.float64)
                    + applied_scale * delta
                )
                attacked.append(value.astype(global_param.dtype, copy=False))
        metrics["mpaf_raw_update_norm"] = norm
        metrics["mpaf_applied_scale"] = applied_scale
        logger.debug(
            "Client %d: MPAF (lambda=%.3f raw_norm=%.4f applied_scale=%.4f)",
            self.client_id,
            attack_scale,
            norm,
            applied_scale,
        )
        return attacked


# ── Registry ──────────────────────────────────────────────────────────────────

ATTACK_REGISTRY: Dict[str, Type[FedSecClient]] = {
    "label_flip_targeted":    LabelFlipClient,
    "label_flip_all_reverse": LabelFlipClient,
    "backdoor":          BackdoorClient,
    "dba":               DBAClient,
    "gaussian_noise":    GaussianNoiseClient,
    "random_noise":      RandomNoiseClient,
    "sign_flip":         SignFlipClient,
    "lie":               CoordinatedAttackClient,
    "alie":              CoordinatedAttackClient,
    "min_max":           CoordinatedAttackClient,
    "min_sum":           CoordinatedAttackClient,
    "byzantine":         ByzantineClient,
    "model_replacement": ModelReplacementClient,
    "scaling_backdoor":  ScalingBackdoorClient,
    "mpaf":              MPAFClient,
    # ── Extension point ────────────────────────────────────────────
    # "your_attack": YourAttackClient,
}


def get_attack_client_class(attack_type: str) -> Type[FedSecClient]:
    t = attack_type.lower()
    if t == "label_flip":
        raise ValueError(
            "Attack type 'label_flip' is no longer supported; use "
            "'label_flip_targeted' or 'label_flip_all_reverse'."
        )
    if t not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack {t!r}. Available: {list(ATTACK_REGISTRY)}"
        )
    return ATTACK_REGISTRY[t]
