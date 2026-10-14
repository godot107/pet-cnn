"""Two CNNs: a small one trained from scratch (the Trello card's "build architecture") and an
ImageNet-pretrained ResNet-18 (transfer learning, what the Kaggle crowd uses)."""
from __future__ import annotations

import torch
from torch import nn

ARCHS = ("smallcnn", "resnet18")


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 conv blocks (32-64-128-256) + global average pool. ~1.2M parameters; any input size >= 16."""

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "smallcnn":
        return SmallCNN(n_classes)
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    raise ValueError(f"arch must be one of {ARCHS}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
