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
from torch.nn import functional as F


class RoutingProjection(nn.Module):
    """Small residual MLP that maps r(x) into the prototype space for routing.

    Without this projection, both r(x) (frozen backbone output) and p_k
    (no-grad EMA buffer) carry no gradient w.r.t. any trainable parameter, so
    the routing distribution q(x) is gradient-free and any routing supervision
    on q(x) (eqs. (9)-(10) of the report) collapses to a no-op. Inserting a
    trainable projection on r(x) — i.e. s_k(x) = cos(MLP(r(x)), p_k) — is the
    smallest change that restores the gradient path through s_k(x) without
    altering the prototype-EMA semantics.

    The block is implemented as a residual MLP with the second linear's
    weights/bias zero-initialized, so at step 0 the projection is the identity
    and routing behavior matches the no-projection baseline. Training then
    deforms the routing geometry to match the routing-loss target.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim is not None else embed_dim
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Zero-init the residual branch so MLP(x) = x at step 0.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.dropout(F.gelu(self.fc1(x))))


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
    """Maps per-sample entropy to a top-N selection in {1, ..., max_n}.

    Generalizes the three-bin policy (low/mid/high entropy) to any `max_n`.
    Thresholds are placed at `max_n - 1` linearly spaced points in
    [entropy_low, entropy_high]; entropies at or above the i-th threshold
    activate at least i+1 experts.

    Default (max_n=3):
        H(q) < entropy_low                    -> 1 expert
        entropy_low <= H(q) < entropy_high    -> 2 experts
        H(q) >= entropy_high                  -> 3 experts
    """

    def __init__(self, entropy_low: float, entropy_high: float, max_n: int = 3) -> None:
        if entropy_low > entropy_high:
            raise ValueError("entropy_low must be <= entropy_high")
        if max_n < 1:
            raise ValueError("max_n must be >= 1")
        self._low = entropy_low
        self._high = entropy_high
        self._max_n = max_n
        if max_n == 1:
            self._thresholds = torch.empty(0)
        else:
            self._thresholds = torch.linspace(entropy_low, entropy_high, max_n - 1)

    def __call__(self, entropy: torch.Tensor) -> torch.Tensor:
        """Return per-sample integer N in {1, ..., max_n}."""
        if self._thresholds.numel() == 0:
            return torch.ones_like(entropy, dtype=torch.long)
        thresholds = self._thresholds.to(entropy.device)
        # Count how many thresholds the entropy meets/exceeds; offset by 1.
        passed = (entropy.unsqueeze(-1) >= thresholds).long().sum(dim=-1)
        return passed + 1


class RoutingStrategy(nn.Module, ABC):
    """Abstract routing strategy."""

    @abstractmethod
    def route(self, representation: torch.Tensor) -> RoutingDecision: ...

    @property
    @abstractmethod
    def num_experts(self) -> int: ...


class PrototypeRouter(RoutingStrategy):
    """Cosine-similarity routing over expert prototypes with entropy-adaptive top-N.

    With a `RoutingProjection` φ:
        s_k(x) = cos(φ(r(x)), p_k)
        q(x)   = softmax(s(x) / temperature)
    Without the projection (φ = identity), this reduces to the report's
    s_k(x) = cos(r(x), p_k), but the routing distribution carries no gradient
    w.r.t. any trainable parameter (frozen backbone + buffer prototypes), so
    any routing supervision on q(x) is a silent no-op. Passing in a
    `RoutingProjection` restores the gradient path through s_k(x).

    Number of selected experts is determined by the entropy of q(x) via
    `EntropyGate`. Note that the prototypes are still updated in raw r-space
    by the trainer; the projection learns to align φ(r(x)) with the EMA
    prototypes during training.
    """

    def __init__(
        self,
        memory: PrototypeMemory,
        gate: EntropyGate,
        num_experts: int,
        temperature: float = 0.5,
        projection: RoutingProjection | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._gate = gate
        self._num_experts = num_experts
        self._temperature = temperature
        # Registered as a submodule so its parameters appear in
        # `learner.parameters()` and are picked up by the optimizer.
        self.projection = projection

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def route(self, representation: torch.Tensor) -> RoutingDecision:
        if self.projection is not None:
            representation = self.projection(representation)
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
        # Mask uninitialized experts: their zero prototype rows would otherwise
        # contribute exp(0)/Z to the softmax and pollute routing.
        init_mask = self._memory.expert_initialized.to(scores.device)
        scores = scores.masked_fill(~init_mask.unsqueeze(0), float("-inf"))
        distribution = torch.softmax(scores / self._temperature, dim=-1)
        entropy = _entropy(distribution)
        n_active = self._gate(entropy)
        # Cap active experts by the number actually initialized so we don't
        # try to select more "real" experts than exist.
        n_active = torch.minimum(n_active, init_mask.sum().to(n_active))
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
        init_mask = self._memory.expert_initialized.to(scores.device)
        scores = scores.masked_fill(~init_mask.unsqueeze(0), float("-inf"))
        distribution = torch.softmax(scores / self._temperature, dim=-1)
        entropy = _entropy(distribution)
        # Cap k by the number of initialized experts in the early phase.
        k_eff = min(self._k, int(init_mask.sum().item()))
        n_active = torch.full((distribution.shape[0],), k_eff,
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
    # Rank-based selection (handles ties): for each row, sort descending and
    # mark the first `n_active[b]` ranks as kept. A threshold-based mask would
    # over-select when multiple entries equal the threshold (e.g., uniform).
    _, sort_idx = distribution.sort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(sort_idx)
    arange_e = torch.arange(e, device=distribution.device).unsqueeze(0).expand(b, -1)
    ranks.scatter_(1, sort_idx, arange_e)
    mask = ranks < n_active.clamp(min=1, max=e).unsqueeze(-1)
    weights = distribution * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights
