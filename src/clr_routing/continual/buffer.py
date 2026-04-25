"""Replay buffers.

The `ReplayBuffer` ABC unifies global vs. per-expert variants behind a single
interface, so the trainer doesn't need to special-case either. Each
implementation manages its own bounded reservoir(s).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class ReplaySample:
    """A single stored example."""

    x: torch.Tensor  # already-transformed input
    y: int


class ReservoirBuffer:
    """Bounded reservoir sampler maintaining a uniform sample over the stream.

    Standard reservoir sampling (Algorithm R): for the n-th item observed,
    if `n < capacity` store it; otherwise replace a random existing item with
    probability `capacity / n`.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: list[ReplaySample] = []
        self._seen = 0

    def add(self, sample: ReplaySample) -> None:
        self._seen += 1
        if len(self._items) < self._capacity:
            self._items.append(sample)
        else:
            j = random.randint(0, self._seen - 1)
            if j < self._capacity:
                self._items[j] = sample

    def sample(self, n: int) -> list[ReplaySample]:
        if not self._items:
            return []
        n = min(n, len(self._items))
        return random.sample(self._items, n)

    def __len__(self) -> int:
        return len(self._items)


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers.

    Implementations decide where samples land (global vs. per-expert) and
    which buffers to draw from at replay time.
    """

    @abstractmethod
    def store_batch(
        self, x: torch.Tensor, y: torch.Tensor, expert_assignment: torch.Tensor
    ) -> None:
        """Store a batch. `expert_assignment[b]` is argmax_k q_k(x_b).

        Implementations may ignore expert_assignment (global buffer) or use it
        to route samples (per-expert buffer).
        """

    @abstractmethod
    def sample_batch(
        self, batch_size: int, active_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Draw a replay batch. Returns (x, y), or None if no samples available.

        `active_experts` is an optional iterable of expert indices that the
        per-expert buffer can use to bias sampling toward currently active experts.
        """

    @abstractmethod
    def __len__(self) -> int: ...


class GlobalReplayBuffer(ReplayBuffer):
    """Single reservoir over all samples regardless of expert assignment.

    Used for the global-replay ablation.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer = ReservoirBuffer(capacity)

    def store_batch(
        self, x: torch.Tensor, y: torch.Tensor, expert_assignment: torch.Tensor
    ) -> None:
        # CPU storage to avoid GPU memory bloat.
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(x.shape[0]):
            self._buffer.add(ReplaySample(x_cpu[i], int(y_cpu[i])))

    def sample_batch(
        self, batch_size: int, active_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        samples = self._buffer.sample(batch_size)
        if not samples:
            return None
        x = torch.stack([s.x for s in samples])
        y = torch.tensor([s.y for s in samples], dtype=torch.long)
        return x, y

    def __len__(self) -> int:
        return len(self._buffer)


class PerExpertReplayBuffer(ReplayBuffer):
    """One reservoir per expert. Samples are routed to argmax_k q_k(x).

    At replay time, draws roughly evenly from the buffers of the currently
    active experts (or all experts if `active_experts` is None).
    """

    def __init__(self, num_experts: int, per_expert_capacity: int) -> None:
        self._num_experts = num_experts
        self._buffers = [ReservoirBuffer(per_expert_capacity) for _ in range(num_experts)]

    def store_batch(
        self, x: torch.Tensor, y: torch.Tensor, expert_assignment: torch.Tensor
    ) -> None:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        assign_cpu = expert_assignment.detach().cpu().tolist()
        for i in range(x.shape[0]):
            self._buffers[assign_cpu[i]].add(ReplaySample(x_cpu[i], int(y_cpu[i])))

    def sample_batch(
        self, batch_size: int, active_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if active_experts is None:
            expert_ids = list(range(self._num_experts))
        else:
            expert_ids = sorted({int(e) for e in active_experts.tolist()})

        # Filter to non-empty buffers.
        expert_ids = [e for e in expert_ids if len(self._buffers[e]) > 0]
        if not expert_ids:
            return None

        # Distribute the requested batch_size across experts (roughly even).
        per = max(1, batch_size // len(expert_ids))
        all_samples: list[ReplaySample] = []
        for e in expert_ids:
            all_samples.extend(self._buffers[e].sample(per))

        if not all_samples:
            return None

        x = torch.stack([s.x for s in all_samples])
        y = torch.tensor([s.y for s in all_samples], dtype=torch.long)
        return x, y

    def occupancy(self) -> list[int]:
        """Return current sample count per expert (useful for W&B logging)."""
        return [len(b) for b in self._buffers]

    def __len__(self) -> int:
        return sum(len(b) for b in self._buffers)
