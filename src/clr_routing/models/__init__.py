"""Model components: backbone, LoRA experts, router, and the composed learner."""

from __future__ import annotations

import torch
from torch import nn

from clr_routing.models.backbone import BackboneBase, ViTBackbone
from clr_routing.models.lora import LoRAExpertBank, MultiExpertLoRALinear
from clr_routing.models.router import (
    EntropyGate,
    FixedTopKRouter,
    PrototypeMemory,
    PrototypeRouter,
    RoutingDecision,
    RoutingStrategy,
)


class ContinualLearner(nn.Module):
    """Composes backbone, LoRA expert bank, router, and classifier into one model.

    Forward pass:
        1. Compute representation via backbone (no LoRA active).
        2. Route: get expert weights from `RoutingStrategy`.
        3. Forward through backbone with LoRA active per routing weights.
        4. Classify.

    Training-time bookkeeping (prototype updates, sample assignment for replay)
    is performed by the `ContinualTrainer`, not here, so that this module
    remains a pure function from input to logits during inference.
    """

    def __init__(
        self,
        backbone: BackboneBase,
        lora_bank: LoRAExpertBank,
        router: RoutingStrategy,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lora_bank = lora_bank
        self.router = router
        self.classifier = classifier

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute r(x) = mean-pooled encoder outputs, with LoRA disabled."""
        with self.lora_bank.disabled():
            return self.backbone.representation(x)

    def forward(
        self, x: torch.Tensor, return_routing: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, RoutingDecision]:
        """Standard forward returning logits.

        If `return_routing=True`, also returns the `RoutingDecision` so the
        trainer can update prototypes and replay buffers.
        """
        r = self.representation(x)
        decision = self.router.route(r)

        with self.lora_bank.activated(decision.weights):
            features = self.backbone.features(x)

        logits = self.classifier(features)
        if return_routing:
            return logits, decision
        return logits


__all__ = [
    "BackboneBase",
    "ContinualLearner",
    "EntropyGate",
    "FixedTopKRouter",
    "LoRAExpertBank",
    "MultiExpertLoRALinear",
    "PrototypeMemory",
    "PrototypeRouter",
    "RoutingDecision",
    "RoutingStrategy",
    "ViTBackbone",
]
