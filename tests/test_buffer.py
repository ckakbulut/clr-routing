"""Tests for replay buffers."""

from __future__ import annotations

import torch

from clr_routing.continual.buffer import (
    GlobalReplayBuffer,
    PerExpertReplayBuffer,
    ReservoirBuffer,
    ReplaySample,
)


def test_reservoir_under_capacity_keeps_all():
    buf = ReservoirBuffer(capacity=10)
    for i in range(5):
        buf.add(ReplaySample(torch.tensor(float(i)), i))
    assert len(buf) == 5


def test_reservoir_at_capacity_does_not_grow():
    buf = ReservoirBuffer(capacity=5)
    for i in range(20):
        buf.add(ReplaySample(torch.tensor(float(i)), i))
    assert len(buf) == 5


def test_per_expert_routes_by_assignment():
    buf = PerExpertReplayBuffer(num_experts=3, per_expert_capacity=10)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 2, 3, 4, 5])
    assignment = torch.tensor([0, 0, 1, 1, 2, 2])
    buf.store_batch(x, y, assignment)
    assert buf.occupancy() == [2, 2, 2]


def test_per_expert_sample_filters_to_active_experts():
    buf = PerExpertReplayBuffer(num_experts=3, per_expert_capacity=10)
    x = torch.randn(6, 4)
    y = torch.arange(6)
    assignment = torch.tensor([0, 0, 1, 1, 2, 2])
    buf.store_batch(x, y, assignment)

    # Sample only from experts 0 and 2; expect no items stored under expert 1.
    sample = buf.sample_batch(batch_size=10, active_experts=torch.tensor([0, 2]))
    assert sample is not None
    _, sampled_y = sample
    # Items in expert 1's buffer have y in {2, 3}; these must NOT appear.
    assert not bool(((sampled_y == 2) | (sampled_y == 3)).any())


def test_global_buffer_ignores_assignment():
    buf = GlobalReplayBuffer(capacity=10)
    x = torch.randn(5, 4)
    y = torch.arange(5)
    assignment = torch.tensor([0, 1, 2, 0, 1])
    buf.store_batch(x, y, assignment)
    assert len(buf) == 5


def test_empty_buffer_returns_none():
    buf = GlobalReplayBuffer(capacity=10)
    assert buf.sample_batch(batch_size=4) is None
