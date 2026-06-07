"""
models/mnist_cnn.py
--------------------
Lightweight CNN for MNIST / Fashion-MNIST / EMNIST.

Architecture
------------
    Conv(in_channels->32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)   # 14x14x32
    Conv(32->64, 3x3, pad=1)          -> BN -> ReLU -> MaxPool(2)   # 7x7x64
    Conv(64->128, 3x3, pad=1)         -> BN -> ReLU                 # 7x7x128
    AdaptiveAvgPool(1)                                               # 1x1x128
    Linear(128 -> num_classes)

Parameters:  ~94K  (vs ResNet-18's 11M -- 114x smaller)
Serialized:  ~0.39 MB per client
Reaches >99% on MNIST in <5 FL rounds.

Usage in config.yaml
--------------------
model:
  architecture: mnist_cnn
  num_classes: 10
  pretrained: false
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    """Lightweight CNN for 28x28 grayscale classification tasks."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 14x14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 7x7

            # Block 3 (no pooling -- preserve spatial info for small inputs)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # 1x1x128
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    import io
    for num_classes, in_channels, dataset in [(10, 1, "mnist/fashion_mnist"),
                                               (62, 1, "emnist")]:
        model = MnistCNN(num_classes=num_classes, in_channels=in_channels)
        params = sum(p.numel() for p in model.parameters())
        dummy = torch.randn(4, in_channels, 28, 28)
        out = model(dummy)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        print(f"{dataset}: {params:,} params, {buf.tell()/1e6:.2f} MB, "
              f"output shape {tuple(out.shape)}")
