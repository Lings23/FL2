"""
models/model_factory.py
------------------------
Model zoo for the federated security framework.

Provided architectures
----------------------
• ResNet-18   — standard torchvision, adapted for CIFAR/MNIST input sizes
• LightCNN    — small ConvNet for MNIST / FEMNIST baselines
• MLP         — fully-connected baseline

Extension interface
-------------------
Register a new model:
    1. Subclass nn.Module and implement forward()
    2. Add it to MODEL_REGISTRY at the bottom

All models expose a `get_parameters()` / `set_parameters()` pair
that returns / receives a flat list of numpy arrays — the
interface expected by Flower.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


# ── Parameter helpers ─────────────────────────────────────────────────────────

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as list of numpy arrays (Flower interface)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load parameters from list of numpy arrays into model (Flower interface)."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# ── ResNet-18 ─────────────────────────────────────────────────────────────────

class ResNet18(nn.Module):
    """
    ResNet-18 adapted for small images (CIFAR: 32×32, MNIST: 28×28).

    Changes vs. torchvision default:
        • First conv: 3×3, stride 1, no padding (instead of 7×7, stride 2)
        • MaxPool removed
        • Final FC replaced with num_classes output
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        in_channels: int = 3,
    ):
        super().__init__()
        self.base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Adapt for small images
        self.base.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.base.maxpool = nn.Identity()  # type: ignore[assignment]
        self.base.fc = nn.Linear(512, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self) -> List[np.ndarray]:
        return get_parameters(self)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self, parameters)

    # ── Extension hooks ───────────────────────────────────────────────────────

    def get_feature_extractor(self) -> nn.Module:
        """Return backbone without classification head (for feature analysis)."""
        extractor = nn.Sequential(*list(self.base.children())[:-1])
        return extractor

    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hook-based intermediate feature extraction (useful for defenses)."""
        features: Dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(module, inp, out):
                features[name] = out.detach()
            return hook

        hooks = [
            self.base.layer1.register_forward_hook(make_hook("layer1")),
            self.base.layer2.register_forward_hook(make_hook("layer2")),
            self.base.layer3.register_forward_hook(make_hook("layer3")),
            self.base.layer4.register_forward_hook(make_hook("layer4")),
        ]
        _ = self(x)
        for h in hooks:
            h.remove()
        return features


# ── Light CNN (MNIST / FEMNIST) ───────────────────────────────────────────────

class LightCNN(nn.Module):
    """
    Lightweight ConvNet — fast baseline for MNIST / FEMNIST.

    Architecture: Conv(32) → Conv(64) → FC(128) → FC(num_classes)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def get_parameters(self) -> List[np.ndarray]:
        return get_parameters(self)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self, parameters)


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Fully-connected MLP — simplest possible baseline."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Flatten()]
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_parameters(self) -> List[np.ndarray]:
        return get_parameters(self)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self, parameters)


# ── Registry & Factory ────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "resnet18": ResNet18,
    "lightcnn": LightCNN,
    "cnn":      LightCNN,   # alias
    "mlp":      MLP,
    # ── Extension point ────────────────────────────────────────────
    # "your_model": YourModelClass,
}

# Default in_channels per dataset
_DATASET_CHANNELS = {"cifar10": 3, "mnist": 1, "femnist": 1}
_DATASET_CLASSES  = {"cifar10": 10, "mnist": 10, "femnist": 62}


def get_model(
    architecture: str,
    num_classes: int = 10,
    pretrained: bool = False,
    dataset_name: str = "cifar10",
) -> nn.Module:
    """
    Instantiate a model from the registry with sensible defaults.

    Parameters
    ----------
    architecture : str
        Key into MODEL_REGISTRY, e.g. "resnet18", "cnn", "mlp"
    num_classes : int
        Number of output classes.
    pretrained : bool
        Use ImageNet pretrained weights (ResNet only).
    dataset_name : str
        Used to infer in_channels automatically.
    """
    arch = architecture.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture {arch!r}. Available: {list(MODEL_REGISTRY)}"
        )

    in_channels = _DATASET_CHANNELS.get(dataset_name.lower(), 3)
    cls = MODEL_REGISTRY[arch]

    if arch == "resnet18":
        return cls(num_classes=num_classes, pretrained=pretrained,
                   in_channels=in_channels)
    if arch in ("lightcnn", "cnn"):
        return cls(num_classes=num_classes, in_channels=in_channels)
    if arch == "mlp":
        # Flat input dim: inferred from in_channels × H × W
        input_dims = {"cifar10": 3 * 32 * 32, "mnist": 784, "femnist": 784}
        return cls(input_dim=input_dims.get(dataset_name.lower(), 784),
                   num_classes=num_classes)

    # Generic fallback — try common signatures
    try:
        return cls(num_classes=num_classes)
    except TypeError:
        return cls()
