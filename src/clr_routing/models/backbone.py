"""Backbone networks. Define an abstract interface so the routing/replay
machinery is independent of vision vs. text encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import timm
import torch
from torch import nn


class BackboneBase(nn.Module, ABC):
    """Abstract backbone. Subclasses must expose:

    - `features(x)`: per-token (or per-position) features used by the classifier.
    - `representation(x)`: a single pooled vector r(x) used by the router.
    - `embed_dim`: the feature dimension.
    - `target_modules`: names of nn.Linear modules eligible for LoRA injection.
    """

    @abstractmethod
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return features used downstream by the classifier."""

    @abstractmethod
    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """Return a pooled (B, embed_dim) representation for routing."""

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def target_modules(self) -> tuple[str, ...]:
        """Substrings identifying nn.Linear modules where LoRA should be injected."""


class ViTBackbone(BackboneBase):
    """timm ViT wrapper.

    Uses the [CLS] token as the classifier feature and mean-pools patch tokens
    for the routing representation. The mean over patch tokens (excluding CLS)
    aligns with the "mean pooling of encoder outputs" specified in the report.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self._vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self._embed_dim = self._vit.num_features

        if freeze_base:
            for p in self._vit.parameters():
                p.requires_grad = False

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ViT forward and return per-token features (B, 1+N, D)."""
        # timm ViTs expose forward_features that returns (B, 1+N, D) by default.
        return self._vit.forward_features(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._forward_tokens(x)
        return tokens[:, 0]  # CLS (classification i.e. summary over the batch) token

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._forward_tokens(x)
        return tokens[:, 1:].mean(
            dim=1
        )  # mean over patch tokens, don't include index 0 as that's the CLS token

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def target_modules(self) -> tuple[str, ...]:
        # Inject LoRA into the QKV projections of each attention block.
        # `attn.qkv` is the standard timm ViT attribute name.
        return ("attn.qkv",)
