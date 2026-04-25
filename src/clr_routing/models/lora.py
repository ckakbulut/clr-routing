"""Multi-expert LoRA modules.

Standard LoRA replaces W with W + (BA) where A is (rank, in) and B is (out, rank).
Here we maintain `num_experts` independent (A, B) pairs and combine their outputs
according to per-sample routing weights w_b (shape (B, num_experts)) provided
by the router:

    y_lora[b, s, o] = scaling * sum_k w[b, k] * (B[k] @ A[k] @ x[b, s])[o]

When `disabled()` is active, the LoRA contribution is skipped entirely so the
backbone produces standard pretrained features (used for routing representation).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class _LoRAState:
    """Mutable, per-forward-pass routing state shared across all LoRA layers.

    Held by `LoRAExpertBank` and read by every `MultiExpertLoRALinear`. Using a
    shared state object avoids passing routing weights through every forward()
    call, which would require modifying the timm ViT internals.
    """

    def __init__(self) -> None:
        self.expert_weights: torch.Tensor | None = None  # (B, E)
        self.enabled: bool = True


class MultiExpertLoRALinear(nn.Module):
    """Linear layer augmented with `num_experts` LoRA adapters.

    Args:
        base: The frozen pretrained `nn.Linear` to wrap.
        num_experts: Number of independent LoRA experts (k in the report).
        rank: LoRA rank.
        alpha: LoRA scaling factor; effective scale is `alpha / rank`.
        state: Shared `_LoRAState` from the bank.
    """

    def __init__(
        self,
        base: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: float,
        state: _LoRAState,
    ) -> None:
        super().__init__()
        self._base = base
        for p in self._base.parameters():
            p.requires_grad = False

        self._num_experts = num_experts
        self._rank = rank
        self._scaling = alpha / rank
        self._state = state

        in_f, out_f = base.in_features, base.out_features
        # A is initialized with kaiming uniform (matches LoRA paper).
        # B is zero-initialized so the LoRA contribution starts at zero.
        self.lora_A = nn.Parameter(torch.empty(num_experts, rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self._base(x)

        if (
            not self._state.enabled
            or self._state.expert_weights is None
            or x.shape[0] == 0
        ):
            return base_out

        weights = self._state.expert_weights  # (B, E)
        if weights.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"Expert weights batch ({weights.shape[0]}) "
                f"does not match input batch ({x.shape[0]})."
            )

        # x can be (B, in) or (B, S, in). Normalize to 3D for einsum, restore later.
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(1)  # (B, 1, in)

        # Step 1: A_x[b, s, e, r] = sum_i A[e, r, i] * x[b, s, i]
        a_x = torch.einsum("eri,bsi->bser", self.lora_A, x)
        # Step 2: BA_x[b, s, e, o] = sum_r B[e, o, r] * A_x[b, s, e, r]
        ba_x = torch.einsum("eor,bser->bseo", self.lora_B, a_x)
        # Step 3: weighted sum over experts
        lora_out = torch.einsum("be,bseo->bso", weights, ba_x)
        lora_out = lora_out * self._scaling

        if squeezed:
            lora_out = lora_out.squeeze(1)

        return base_out + lora_out


class LoRAExpertBank(nn.Module):
    """Manages all `MultiExpertLoRALinear` layers across a backbone.

    Responsibilities:
        - Inject LoRA layers into the backbone at modules matching
          `target_modules` substrings.
        - Hold the shared `_LoRAState` so routing weights flow into every layer.
        - Provide `activated(weights)` and `disabled()` context managers (RAII)
          for safe state management — state is always restored on exit.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_modules: tuple[str, ...],
        num_experts: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self._num_experts = num_experts
        self._state = _LoRAState()
        self._injected = self._inject(backbone, target_modules, num_experts, rank, alpha)

    def _inject(
        self,
        backbone: nn.Module,
        target_modules: tuple[str, ...],
        num_experts: int,
        rank: int,
        alpha: float,
    ) -> nn.ModuleList:
        """Replace matching nn.Linear modules in `backbone` with multi-expert versions.

        Returns a ModuleList of the injected modules so they are tracked as
        parameters of this bank (pretrained backbone params remain frozen).
        """
        injected = []
        for name, module in list(backbone.named_modules()):
            if not any(name.endswith(t) for t in target_modules):
                continue
            if not isinstance(module, nn.Linear):
                continue

            new_layer = MultiExpertLoRALinear(
                base=module,
                num_experts=num_experts,
                rank=rank,
                alpha=alpha,
                state=self._state,
            )
            self._replace_in_parent(backbone, name, new_layer)
            injected.append(new_layer)

        if not injected:
            raise RuntimeError(
                f"No modules matched target patterns {target_modules}. "
                "Check Backbone.target_modules."
            )
        return nn.ModuleList(injected)

    @staticmethod
    def _replace_in_parent(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @contextmanager
    def activated(self, expert_weights: torch.Tensor) -> Iterator[None]:
        """Enable LoRA with the given per-sample routing weights.

        The previous state is restored on exit, even if an exception is raised.
        This is the RAII pattern for the routing state.
        """
        prev_weights = self._state.expert_weights
        prev_enabled = self._state.enabled
        self._state.expert_weights = expert_weights
        self._state.enabled = True
        try:
            yield
        finally:
            self._state.expert_weights = prev_weights
            self._state.enabled = prev_enabled

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Disable LoRA entirely (used for computing the routing representation)."""
        prev_enabled = self._state.enabled
        self._state.enabled = False
        try:
            yield
        finally:
            self._state.enabled = prev_enabled
