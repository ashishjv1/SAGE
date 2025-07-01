"""model_factory.py  – v4 (full)
================================================
One‑stop factory for CNN backbones that require **zero extra pip installs**.

Highlights
----------
* **Built‑in ResNeXt‑29 CIFAR family** (32×4d, 8×64d, 4×64d, 2×64d) – no more
  cloning Kuang‑Liu or installing from Git. The reference implementation is
  embedded below (MIT license).
* **TorchVision** backbones: ResNet‑18, ResNeXt‑50/101.
* **timm** models work out‑of‑the‑box if you have timm installed – the factory
  forwards unknown names to `timm.create_model()`.

Basic usage
-----------
```python
from model_factory import create_model
model = create_model("resnext", num_classes=100).to(device)  # loads ResNeXt‑29 32×4d
model = create_model("resnet18", num_classes=100, pretrained=True)
```

Discover all aliases
--------------------
```python
from model_factory import list_supported
print(list_supported())
```
"""
from __future__ import annotations

import inspect
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional dependency : timm (for extra models) --------------------------------
try:
    import timm  # noqa: F401 – may be absent
except ImportError:  # pragma: no cover
    timm = None  # type: ignore

# -----------------------------------------------------------------------------
# 🔧  ResNeXt‑29 implementation for 32×32 images (CIFAR) -----------------------
# Lightly adapted from https://github.com/kuangliu/pytorch-cifar (MIT)
# -----------------------------------------------------------------------------

class _RxBlock(nn.Module):
    """Bottleneck block with grouped conv (ResNeXt)."""

    expansion = 2  # output channel multiplier

    def __init__(self, in_planes: int, cardinality: int, bottleneck_width: int, stride: int):
        super().__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        self.conv2 = nn.Conv2d(
            group_width,
            group_width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class _ResNeXtCIFAR(nn.Module):
    """ResNeXt‑29 for CIFAR‑10/100 (32×32 inputs)."""

    def __init__(
        self,
        input_channel: int,
        num_blocks: List[int],
        cardinality: int,
        bottleneck_width: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, n_classes)

    # ------------------------------------------------------------------
    def _make_layer(self, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_RxBlock(self.in_planes, self.cardinality, self.bottleneck_width, s))
            self.in_planes = _RxBlock.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2  # widen after each stage
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, *, last: bool = False, freeze: bool = False):
        fn = torch.no_grad if freeze else (lambda: torch.enable_grad())
        with fn():
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.pool(out).view(out.size(0), -1)
        logits = self.linear(out)
        return (logits, out) if last else logits


# -----------------------------------------------------------------------------
# Public constructors for ResNeXt‑29 variants ----------------------------------

def ResNeXt29_2x64d(input_channel: int, n_classes: int):
    return _ResNeXtCIFAR(input_channel, [3, 3, 3], 2, 64, n_classes)


def ResNeXt29_4x64d(input_channel: int, n_classes: int):
    return _ResNeXtCIFAR(input_channel, [3, 3, 3], 4, 64, n_classes)


def ResNeXt29_8x64d(input_channel: int, n_classes: int):
    return _ResNeXtCIFAR(input_channel, [3, 3, 3], 8, 64, n_classes)


def ResNeXt29_32x4d(input_channel: int, n_classes: int):
    return _ResNeXtCIFAR(input_channel, [3, 3, 3], 32, 4, n_classes)


_CUSTOM_CIFAR = {
    "cifar_resnext29_32x4d": ResNeXt29_32x4d,
    "resnext29_32x4d": ResNeXt29_32x4d,
    "cifar_resnext29_8x64d": ResNeXt29_8x64d,
    "resnext29_8x64d": ResNeXt29_8x64d,
    "cifar_resnext29_4x64d": ResNeXt29_4x64d,
    "resnext29_4x64d": ResNeXt29_4x64d,
    "cifar_resnext29_2x64d": ResNeXt29_2x64d,
    "resnext29_2x64d": ResNeXt29_2x64d,
}

# -----------------------------------------------------------------------------
# TorchVision helpers ----------------------------------------------------------
from torchvision import models as tvm
from torchvision.models import (
    ResNet18_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
)

_ALIAS: Dict[str, str] = {
    # CIFAR ResNeXt‑29 family
    "resnext": "cifar_resnext29_32x4d",
    "resnext29_32x4d": "cifar_resnext29_32x4d",
    "resnext29_8x64d": "cifar_resnext29_8x64d",
    "resnext29_4x64d": "cifar_resnext29_4x64d",
    "resnext29_2x64d": "cifar_resnext29_2x64d",

    # TorchVision backbones
    "resnet18": "resnet18",
    "resnext50": "resnext50_32x4d",
    "resnext101": "resnext101_32x8d",
}

_TORCHVISION_NAMES = {"resnet18", "resnext50_32x4d", "resnext101_32x8d"}


def _build_torchvision(name: str, num_classes: int, pretrained: bool):
    """Return a TorchVision backbone with new `num_classes` classifier."""

    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnext50_32x4d":
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnext101_32x8d":
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnext101_32x8d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise KeyError(f"Unknown TorchVision model '{name}'")


# -----------------------------------------------------------------------------
# 🎁  Public factory -----------------------------------------------------------

def create_model(
    arch: str,
    num_classes: int = 100,
    pretrained: bool | str = True,
    **kwargs,
):
    """Generic model creator.

    Parameters
    ----------
    arch : str
        Alias or full model name (see `list_supported()`).
    num_classes : int
        Number of classes for the final classifier layer.
    pretrained : bool or str
        `True` → use ImageNet weights if available (TorchVision, timm). If a
        string, pass it to timm as a checkpoint path.
    **kwargs : Any
        Extra keyword arguments forwarded to `timm.create_model()`.
    """

    name = _ALIAS.get(arch.lower(), arch.lower())

    # 1. TorchVision -----------------------------------------------------------
    if name in _TORCHVISION_NAMES:
        return _build_torchvision(name, num_classes, bool(pretrained))

    # 2. Built‑in CIFAR ResNeXt‑29 -------------------------------------------
    if name in _CUSTOM_CIFAR:
        fn = _CUSTOM_CIFAR[name]
        return fn(input_channel=3, n_classes=num_classes)  # type: ignore[arg-type]

    # 3. timm (only if installed) ---------------------------------------------
    if timm is not None and hasattr(timm, "is_model") and timm.is_model(name):
        return timm.create_model(
            name,
            pretrained=pretrained if isinstance(pretrained, bool) else False,
            num_classes=num_classes,
            **kwargs,
        )

    # Legacy timm name (strip "cifar_" prefix) ------------------------------
    if name.startswith("cifar_") and timm is not None:
        legacy = name.replace("cifar_", "")
        if timm.is_model(legacy):
            return timm.create_model(legacy, pretrained=False, num_classes=num_classes, **kwargs)

    # 4. No luck → helpful error ---------------------------------------------
    lines = [f"Unknown model '{name}'."]
    if timm is not None:
        matches = timm.list_models(f"*{name.split('_')[0]}*")
        if matches:
            lines.append(f"Closest timm matches: {matches}")
    else:
        lines.append("timm not installed; `pip install timm` for extra models.")
    raise RuntimeError("\n".join(lines))


# -----------------------------------------------------------------------------
# 🔍 Introspection utilities ---------------------------------------------------

def list_supported() -> List[str]:
    """Return sorted list of aliases recognised by the factory."""
    return sorted(_ALIAS)


def describe(arch: str) -> str:
    """Return docstring of the underlying constructor (TorchVision · timm · CIFAR impl)."""
    name = _ALIAS.get(arch.lower(), arch.lower())

    if name in _TORCHVISION_NAMES:
        fn = getattr(tvm, name)
    elif name in _CUSTOM_CIFAR:
        fn = _CUSTOM_CIFAR[name]
    elif timm is not None and hasattr(timm, "create_model"):
        fn = timm.create_model  # type: ignore[assignment]
    else:
        return "<no backend available>"

    return inspect.getdoc(fn) or "<no docstring>"


__all__ = [
    "create_model",
    "list_supported",
    "describe",
    # public CIFAR variants
    "ResNeXt29_32x4d",
    "ResNeXt29_8x64d",
    "ResNeXt29_4x64d",
    "ResNeXt29_2x64d",
]
