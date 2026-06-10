"""
models/model_factory.py
------------------------
Central registry for all model architectures used in FL experiments.

Dataset -> architecture recommendations
----------------------------------------
Dataset          Resolution    Architecture     Params      Serialized
--------         ----------    ------------     ------      ----------
mnist            28x28x1       mnist_cnn        ~94K        0.39 MB
fashion_mnist    28x28x1       mnist_cnn        ~94K        0.39 MB
emnist           28x28x1       mnist_cnn        ~107K       0.43 MB
cifar10          32x32x3       resnet18         ~11M        44.7 MB
cifar100         32x32x3       resnet18         ~11M        44.8 MB
svhn             32x32x3       resnet18         ~11M        44.7 MB
tiny_imagenet    64x64x3       resnet18         ~11M        45.0 MB
imagenet         224x224x3     resnet50         ~25M        102  MB
celeba           64x64x3       mobilenet_v2     ~2.2M       9.0  MB
shakespeare      sequence      lstm             ~800K       3.2  MB

ResNet-18 on small inputs (<=64px)
-----------------------------------
The standard torchvision ResNet-18 has conv1(kernel=7, stride=2) + MaxPool(stride=2),
which downsamples the input 4x before the first residual block.

  32x32 input -> 16 (conv1) -> 8 (pool) -> ... -> 1x1 after layer3
  28x28 input -> 14 (conv1) -> 7 (pool) -> ... -> 0x0 (collapses entirely)

For datasets with resolution <= 64px, conv1 is replaced with (kernel=3, stride=1,
padding=1) and MaxPool is removed. This is the standard CIFAR-ResNet used in all
FL benchmark papers (He et al. 2016, McMahan et al. 2017).
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

from models.mnist_cnn import MnistCNN
from models.char_lstm import CharLSTM

logger = logging.getLogger(__name__)

# Datasets whose resolution is too small for the standard ResNet stem
_SMALL_INPUT_DATASETS = {"mnist", "fashion_mnist", "emnist",
                          "cifar10", "cifar100", "svhn", "tiny_imagenet",
                          "celeba"}

# Datasets that are grayscale (single channel)
_GRAYSCALE_DATASETS = {"mnist", "fashion_mnist", "emnist"}


def _patch_resnet_for_small_input(model: nn.Module, in_channels: int) -> nn.Module:
    """
    Replace the ResNet stem so it works on inputs <= 64px.

    Standard stem: Conv(7x7, stride=2) + BN + ReLU + MaxPool(stride=2)
    Patched stem:  Conv(3x3, stride=1) + BN + ReLU  (no MaxPool)

    This preserves spatial resolution through the stem, matching the
    setup used in virtually every FL paper that benchmarks on CIFAR/SVHN.
    Also handles grayscale by setting in_channels on conv1.
    """
    model.conv1 = nn.Conv2d(
        in_channels, 64,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    # Replace maxpool with identity so the spatial size is not halved again
    model.maxpool = nn.Identity()
    return model


def get_model(
    architecture: str,
    num_classes: int = 10,
    pretrained: bool = False,
    dataset_name: str = "",
) -> nn.Module:
    """
    Instantiate and return the requested model architecture.

    Parameters
    ----------
    architecture : str
        One of: mnist_cnn/cnn, mlp, resnet18, resnet34, resnet50,
                mobilenet_v2, efficientnet_b0, lstm
    num_classes  : int
        Number of output classes (replaces the head if needed).
    pretrained   : bool
        Load ImageNet-pretrained weights (only for torchvision models).
        Ignored for mnist_cnn and lstm.
    dataset_name : str
        Used to automatically apply input-channel and stem patches.
        Pass the dataset name from config (e.g. "cifar10", "mnist").
    """
    import torchvision.models as tvm

    arch = architecture.lower().replace("-", "_")
    ds = dataset_name.lower()
    is_small = ds in _SMALL_INPUT_DATASETS
    in_channels = 1 if ds in _GRAYSCALE_DATASETS else 3

    # ------------------------------------------------------------------
    # mnist_cnn: lightweight CNN for 28x28 grayscale tasks
    # ------------------------------------------------------------------
    if arch in ("mnist_cnn", "cnn"):
        if ds and ds not in {"mnist", "fashion_mnist", "emnist", ""}:
            logger.warning(
                "mnist_cnn is designed for 28x28 grayscale inputs; "
                "dataset=%r may give poor results. "
                "Consider resnet18 for RGB or larger inputs.", ds
            )
        return MnistCNN(num_classes=num_classes, in_channels=in_channels)

    # ------------------------------------------------------------------
    # MLP: simple fully-connected baseline for 28x28 image datasets
    # ------------------------------------------------------------------
    elif arch == "mlp":
        if ds and ds not in {"mnist", "fashion_mnist", "emnist", ""}:
            logger.warning(
                "mlp assumes flattened 28x28 inputs; dataset=%r may not fit.",
                ds,
            )
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 28 * 28, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )

    # ------------------------------------------------------------------
    # ResNet-18
    # ------------------------------------------------------------------
    elif arch == "resnet18":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if is_small or in_channels != 3:
            model = _patch_resnet_for_small_input(model, in_channels)
        return model

    # ------------------------------------------------------------------
    # ResNet-34
    # ------------------------------------------------------------------
    elif arch == "resnet34":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if is_small or in_channels != 3:
            model = _patch_resnet_for_small_input(model, in_channels)
        return model

    # ------------------------------------------------------------------
    # ResNet-50: for ImageNet-scale datasets
    # ------------------------------------------------------------------
    elif arch == "resnet50":
        weights = "IMAGENET1K_V2" if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if is_small or in_channels != 3:
            model = _patch_resnet_for_small_input(model, in_channels)
        return model

    # ------------------------------------------------------------------
    # MobileNetV2: lightweight option for celeba / medium-res datasets
    # ------------------------------------------------------------------
    elif arch == "mobilenet_v2":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.mobilenet_v2(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        if in_channels != 3:
            first_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
        return model

    # ------------------------------------------------------------------
    # EfficientNet-B0: efficient option for medium/large-res datasets
    # ------------------------------------------------------------------
    elif arch == "efficientnet_b0":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        if in_channels != 3:
            first_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
        return model

    # ------------------------------------------------------------------
    # CharLSTM: for text datasets (Shakespeare, sent140)
    # ------------------------------------------------------------------
    elif arch in ("lstm", "char_lstm"):
        return CharLSTM(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Available: mnist_cnn/cnn, mlp, resnet18, resnet34, resnet50, "
            f"mobilenet_v2, efficientnet_b0, lstm"
        )


# ---------------------------------------------------------------------------
# Parameter serialization helpers (used by FedSecClient)
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> List:
    """Extract model state dict as a list of numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List) -> None:
    """Load a list of numpy arrays into a model's state dict."""
    state_dict = model.state_dict()
    new_state = {
        k: torch.tensor(v, dtype=existing.dtype)
        for (k, existing), v in zip(state_dict.items(), parameters)
    }
    model.load_state_dict(new_state, strict=True)
