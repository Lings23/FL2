"""
data/dataset.py
---------------
Dataset loaders + federated partitioning for:
    • CIFAR-10
    • MNIST
    • FEMNIST  (Federated EMNIST — loaded from LEAF / torchvision)

Partitioning strategies:
    • IID          — random uniform split
    • Non-IID      — shard-based (2 classes per client)
    • Dirichlet    — LDA-based heterogeneous split (α controls skew)

Extension interface:
    Register a new dataset by subclassing BaseDataset and
    adding it to DATASET_REGISTRY at the bottom of this file.
"""

from __future__ import annotations

import os
import random
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


# ── Transform factories ───────────────────────────────────────────────────────

def cifar10_transforms(train: bool = True) -> Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
    ])


def mnist_transforms(train: bool = True) -> Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def femnist_transforms(train: bool = True) -> Compose:
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.9641,), (0.1592,)),
    ])


# ── Base dataset ─────────────────────────────────────────────────────────────

class BaseDataset(ABC):
    """
    Abstract base class for all federated datasets.

    Subclass and implement:
        load_train() -> Dataset
        load_test()  -> Dataset
        num_classes  -> int
        input_shape  -> Tuple[int, ...]
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """C x H x W"""
        ...

    @abstractmethod
    def load_train(self) -> Dataset:
        ...

    @abstractmethod
    def load_test(self) -> Dataset:
        ...

    def get_test_loader(self, batch_size: int = 128) -> DataLoader:
        return DataLoader(self.load_test(), batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)


# ── Concrete dataset implementations ─────────────────────────────────────────

class CIFAR10Dataset(BaseDataset):
    num_classes = 10
    input_shape = (3, 32, 32)

    def load_train(self) -> Dataset:
        return datasets.CIFAR10(
            root=self.data_dir, train=True, download=True,
            transform=cifar10_transforms(train=True))

    def load_test(self) -> Dataset:
        return datasets.CIFAR10(
            root=self.data_dir, train=False, download=True,
            transform=cifar10_transforms(train=False))


class MNISTDataset(BaseDataset):
    num_classes = 10
    input_shape = (1, 28, 28)

    def load_train(self) -> Dataset:
        return datasets.MNIST(
            root=self.data_dir, train=True, download=True,
            transform=mnist_transforms(train=True))

    def load_test(self) -> Dataset:
        return datasets.MNIST(
            root=self.data_dir, train=False, download=True,
            transform=mnist_transforms(train=False))


class FEMNISTDataset(BaseDataset):
    """
    Federated EMNIST (FEMNIST).

    Falls back to torchvision EMNIST 'byclass' split
    if the LEAF processed files are not present.

    For full LEAF partitioning (per-user splits), place
    the LEAF output under data/femnist/{train,test}/ and
    set LEAF_AVAILABLE=True.
    """
    num_classes = 62  # 10 digits + 26 lower + 26 upper
    input_shape = (1, 28, 28)
    LEAF_AVAILABLE: bool = False

    def load_train(self) -> Dataset:
        if self.LEAF_AVAILABLE:
            return self._load_leaf("train")
        logger.warning("LEAF data not found — using torchvision EMNIST 'byclass'.")
        return datasets.EMNIST(
            root=self.data_dir, split="byclass", train=True, download=True,
            transform=femnist_transforms(train=True))

    def load_test(self) -> Dataset:
        if self.LEAF_AVAILABLE:
            return self._load_leaf("test")
        return datasets.EMNIST(
            root=self.data_dir, split="byclass", train=False, download=True,
            transform=femnist_transforms(train=False))

    def _load_leaf(self, split: str) -> Dataset:
        """Load per-user LEAF FEMNIST JSONs and return a PyTorch Dataset."""
        import json

        class LEAFDataset(Dataset):
            def __init__(self, data_dir, split, transform=None):
                self.samples: List[Tuple] = []
                self.transform = transform
                leaf_dir = Path(data_dir) / "femnist" / split
                for json_file in sorted(leaf_dir.glob("*.json")):
                    with open(json_file) as f:
                        raw = json.load(f)
                    for user in raw["users"]:
                        xs = raw["user_data"][user]["x"]
                        ys = raw["user_data"][user]["y"]
                        for x, y in zip(xs, ys):
                            self.samples.append((np.array(x, dtype=np.float32)
                                                  .reshape(28, 28), int(y)))

            def __len__(self): return len(self.samples)

            def __getitem__(self, idx):
                img, label = self.samples[idx]
                img_tensor = torch.tensor(img).unsqueeze(0)
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                return img_tensor, label

        return LEAFDataset(self.data_dir, split, femnist_transforms(split == "train"))


# ── Partitioner ───────────────────────────────────────────────────────────────

class FederatedPartitioner:
    """
    Splits a training dataset into per-client subsets.

    Strategies
    ----------
    iid        : Each client gets an equal random sample.
    non_iid    : Shard-based — sort by label, cut into 2×num_clients shards,
                 assign 2 shards per client (classic McMahan et al.).
    dirichlet  : LDA with concentration α (lower = more heterogeneous).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        strategy: str = "iid",
        dirichlet_alpha: float = 0.5,
        seed: int = 42,
        val_split: float = 0.1,
    ):
        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy
        self.alpha = dirichlet_alpha
        self.seed = seed
        self.val_split = val_split
        self._rng = np.random.default_rng(seed)

        self._targets = self._extract_targets()
        self._client_indices: Optional[Dict[int, List[int]]] = None

    def _extract_targets(self) -> np.ndarray:
        if hasattr(self.dataset, "targets"):
            t = self.dataset.targets
            return np.array(t.numpy() if isinstance(t, torch.Tensor) else t)
        # Fallback: iterate (slow)
        return np.array([self.dataset[i][1] for i in range(len(self.dataset))])

    # ── Public API ────────────────────────────────────────────────────────────

    def get_client_data(self, client_id: int) -> Tuple[Subset, Subset]:
        """Return (train_subset, val_subset) for the given client."""
        if self._client_indices is None:
            self._client_indices = self._partition()
        indices = self._client_indices[client_id]
        n_val = max(1, int(len(indices) * self.val_split))
        self._rng.shuffle(indices)
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        return Subset(self.dataset, train_idx), Subset(self.dataset, val_idx)

    def get_all_client_indices(self) -> Dict[int, List[int]]:
        if self._client_indices is None:
            self._client_indices = self._partition()
        return self._client_indices

    def summary(self) -> str:
        all_idx = self.get_all_client_indices()
        lines = [f"Partition: {self.strategy}  |  Clients: {self.num_clients}"]
        for cid, idx in all_idx.items():
            labels, counts = np.unique(self._targets[idx], return_counts=True)
            dist = {int(l): int(c) for l, c in zip(labels, counts)}
            lines.append(f"  Client {cid:3d}: n={len(idx):5d}  dist={dist}")
        return "\n".join(lines)

    # ── Partition strategies ──────────────────────────────────────────────────

    def _partition(self) -> Dict[int, List[int]]:
        if self.strategy == "iid":
            return self._iid()
        if self.strategy == "non_iid":
            return self._non_iid_shards()
        if self.strategy == "dirichlet":
            return self._dirichlet()
        raise ValueError(f"Unknown partition strategy: {self.strategy!r}")

    def _iid(self) -> Dict[int, List[int]]:
        n = len(self.dataset)
        idx = self._rng.permutation(n).tolist()
        return {i: idx[i::self.num_clients] for i in range(self.num_clients)}

    def _non_iid_shards(self, shards_per_client: int = 2) -> Dict[int, List[int]]:
        """McMahan et al. 2017 non-IID construction."""
        sorted_idx = np.argsort(self._targets)
        num_shards = shards_per_client * self.num_clients
        shard_size = len(self.dataset) // num_shards
        shards = [sorted_idx[i * shard_size:(i + 1) * shard_size].tolist()
                  for i in range(num_shards)]
        shard_ids = self._rng.permutation(num_shards).tolist()
        result = {}
        for cid in range(self.num_clients):
            assigned = shard_ids[cid * shards_per_client:(cid + 1) * shards_per_client]
            result[cid] = [idx for s in assigned for idx in shards[s]]
        return result

    def _dirichlet(self) -> Dict[int, List[int]]:
        """LDA-based split. Lower α → more heterogeneous."""
        classes = np.unique(self._targets)
        client_indices: Dict[int, List[int]] = defaultdict(list)
        for c in classes:
            c_idx = np.where(self._targets == c)[0]
            self._rng.shuffle(c_idx)
            proportions = self._rng.dirichlet(
                np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
            for cid, split in enumerate(np.split(c_idx, proportions)):
                client_indices[cid].extend(split.tolist())
        return dict(client_indices)


# ── Factory ───────────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, type] = {
    "cifar10": CIFAR10Dataset,
    "mnist":   MNISTDataset,
    "femnist": FEMNISTDataset,
    # ── Extension point ────────────────────────────────────────────
    # "your_dataset": YourDatasetClass,
}


def get_dataset(name: str, data_dir: str = "data/") -> BaseDataset:
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](data_dir=data_dir)
