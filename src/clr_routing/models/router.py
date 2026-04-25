"""Routing strategies and prototype memory.

The `RoutingStrategy` ABC lets the trainer remain agnostic to whether routing is
prototype-guided (the proposed method) or fixed top-K (baselines). Both strategies
return a `RoutingDecision` carrying the per-sample expert weights and any
auxiliary tensors needed for logging or losses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RoutingDecision:
    """Output of a routing strategy.

    Attributes:
        weights: (B, num_experts) — combination weights to feed into LoRA.
            For top-N selection, inactive experts have weight 0; active experts
            sum to 1.
        distribution: (B, num_experts) — full softmax distribution before
            top-N selection. Used by the routing KL loss.
        entropy: (B,) — entropy of `distribution`. Used by the entropy gate
            and for diagnostics.
        num_active: (B,) — integer number of experts activated per sample.
    """

    weights: torch.Tensor
    distribution: torch.Tensor
    entropy: torch.Tensor
    num_active: torch.Tensor


class PrototypeMemory(nn.Module):
    """EMA-updated prototypes for tasks (training only) and experts.

    Stored as buffers (not parameters) so they are not affected by optimizers
    but are included in `state_dict()` for checkpointing.
    """

    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        max_tasks: int,
        expert_ema_beta: float = 0.1,
        task_ema_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self._beta = expert_ema_beta
        self._gamma = task_ema_gamma

        # L2-normalized at use-time; stored unnormalized to keep EMA simple.
        self.register_buffer("expert_prototypes", torch.zeros(num_experts, embed_dim))
        self.register_buffer("expert_initialized", torch.zeros(num_experts, dtype=torch.bool))
        self.register_buffer("task_prototypes", torch.zeros(max_tasks, embed_dim))
        self.register_buffer("task_initialized", torch.zeros(max_tasks, dtype=torch.bool))

    @torch.no_grad()
    def update_expert(self, expert_id: int, samples: torch.Tensor) -> None:
        """EMA-update expert prototype with mean of `samples` (already representations).

        On first update, initialize directly to the mean rather than
        EMA-blending with a zero prior.
        """
        new = samples.mean(dim=0)
        if bool(self.expert_initialized[expert_id]):
            self.expert_prototypes[expert_id] = (
                (1 - self._beta) * self.expert_prototypes[expert_id] + self._beta * new
            )
        else:
            self.expert_prototypes[expert_id] = new
            self.expert_initialized[expert_id] = True

    @torch.no_grad()
    def update_task(self, task_id: int, samples: torch.Tensor) -> None:
        new = samples.mean(dim=0)
        if bool(self.task_initialized[task_id]):
            self.task_prototypes[task_id] = (
                (1 - self._gamma) * self.task_prototypes[task_id] + self._gamma * new
            )
        else:
            self.task_prototypes[task_id] = new
            self.task_initialized[task_id] = True

    def expert_prototypes_normalized(self) -> torch.Tensor:
        return _safe_normalize(self.expert_prototypes)

    def task_prototype_normalized(self, task_id: int) -> torch.Tensor:
        return _safe_normalize(self.task_prototypes[task_id : task_id + 1]).squeeze(0)


class EntropyGate:
    """Maps per-sample entropy to a top-N selection (1, 2, or 3 experts).

    Thresholds are configurable. The default policy:
        H(q) < entropy_low                    -> 1 expert
        entropy_low <= H(q) < entropy_high    -> 2 experts
        H(q) >= entropy_high                  -> 3 experts
    """

    def __init__(self, entropy_low: float, entropy_high: float, max_n: int = 3) -> None:
        if entropy_low > entropy_high:
            raise ValueError("entropy_low must be <= entropy_high")
        self._low = entropy_low
        self._high = entropy_high
        self._max_n = max_n

    def __call__(self, entropy: torch.Tensor) -> torch.Tensor:
        """Return per-sample integer N in {1, ..., max_n}."""
        n = torch.ones_like(entropy, dtype=torch.long)
        n = torch.where(entropy >= self._low, n + 1, n)
        n = torch.where(entropy >= self._high, n + 1, n)
        return torch.clamp(n, max=self._max_n)


class RoutingStrategy(nn.Module, ABC):
    """Abstract routing strategy."""

    @abstractmethod
    def route(self, representation: torch.Tensor) -> RoutingDecision: ...

    @property
    @abstractmethod
    def num_experts(self) -> int: ...


class PrototypeRouter(RoutingStrategy):
    """Cosine-similarity routing over expert prototypes with entropy-adaptive top-N.

    s_k(x) = cos(r(x), p_k)
    q(x)   = softmax(s(x) / temperature)
    Number of selected experts is determined by the entropy of q(x) via
    `EntropyGate`.
    """

    def __init__(
        self,
        memory: PrototypeMemory,
        gate: EntropyGate,
        num_experts: int,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._gate = gate
        self._num_experts = num_experts
        self._temperature = temperature

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def route(self, representation: torch.Tensor) -> RoutingDecision:
        r = _safe_normalize(representation)  # (B, D)
        p = self._memory.expert_prototypes_normalized()  # (E, D)

        # If no expert has been initialized yet, fall back to uniform routing.
        if not bool(self._memory.expert_initialized.any()):
            uniform = torch.full(
                (representation.shape[0], self._num_experts),
                1.0 / self._num_experts,
                device=representation.device,
            )
            entropy = _entropy(uniform)
            n_active = self._gate(entropy)
            return RoutingDecision(
                weights=uniform,
                distribution=uniform,
                entropy=entropy,
                num_active=n_active,
            )

        scores = r @ p.T  # (B, E)
        distribution = torch.softmax(scores / self._temperature, dim=-1)
        entropy = _entropy(distribution)
        n_active = self._gate(entropy)
        weights = _select_top_n(distribution, n_active)
        return RoutingDecision(
            weights=weights,
            distribution=distribution,
            entropy=entropy,
            num_active=n_active,
        )


class FixedTopKRouter(RoutingStrategy):
    """Baseline: always activate the same K experts (e.g. K=1 round-robin per task).

    For task-incremental baselines you would configure this to deterministically
    select the expert assigned to the current task. For task-agnostic baselines,
    you can use the same prototype-based scoring but with a fixed K.
    """

    def __init__(
        self,
        memory: PrototypeMemory,
        num_experts: int,
        k: int,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if k < 1 or k > num_experts:
            raise ValueError(f"k={k} must be in [1, {num_experts}]")
        self._memory = memory
        self._num_experts = num_experts
        self._k = k
        self._temperature = temperature

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def route(self, representation: torch.Tensor) -> RoutingDecision:
        r = _safe_normalize(representation)
        p = self._memory.expert_prototypes_normalized()

        if not bool(self._memory.expert_initialized.any()):
            uniform = torch.full(
                (representation.shape[0], self._num_experts),
                1.0 / self._num_experts,
                device=representation.device,
            )
            entropy = _entropy(uniform)
            n_active = torch.full_like(entropy, self._k, dtype=torch.long)
            return RoutingDecision(uniform, uniform, entropy, n_active)

        scores = r @ p.T
        distribution = torch.softmax(scores / self._temperature, dim=-1)
        entropy = _entropy(distribution)
        n_active = torch.full((distribution.shape[0],), self._k,
                              dtype=torch.long, device=distribution.device)
        weights = _select_top_n(distribution, n_active)
        return RoutingDecision(weights, distribution, entropy, n_active)


# ---------- helpers ----------

def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * (p.clamp_min(eps).log())).sum(dim=-1)


def _select_top_n(distribution: torch.Tensor, n_active: torch.Tensor) -> torch.Tensor:
    """Zero out all but the top-`n_active[b]` entries of each row, renormalize.

    Args:
        distribution: (B, E) softmax distribution.
        n_active: (B,) integer number of experts to keep per row.

    Returns:
        (B, E) sparse weights summing to 1 along the expert dim.
    """
    b, e = distribution.shape
    # For each row, find the threshold value (the n_active-th largest).
    # We sort descending and gather the threshold per row.
    sorted_vals, _ = distribution.sort(dim=-1, descending=True)
    # n_active[b] is in [1, E]. We pick sorted_vals[b, n_active[b] - 1].
    idx = (n_active.clamp(min=1, max=e) - 1).unsqueeze(-1)  # (B, 1)
    threshold = sorted_vals.gather(dim=-1, index=idx)  # (B, 1)

    mask = distribution >= threshold
    weights = distribution * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights
