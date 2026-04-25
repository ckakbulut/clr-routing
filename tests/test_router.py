"""Tests for routing math: prototype memory, entropy gate, and routers."""

from __future__ import annotations

import pytest
import torch

from clr_routing.models.router import (
    EntropyGate,
    FixedTopKRouter,
    PrototypeMemory,
    PrototypeRouter,
    _entropy,
    _select_top_n,
)


def test_entropy_uniform_is_max():
    """Entropy of a uniform distribution over E experts is log(E)."""
    e = 5
    uniform = torch.full((1, e), 1.0 / e)
    h = _entropy(uniform).item()
    assert pytest.approx(h, rel=1e-5) == torch.log(torch.tensor(float(e))).item()


def test_entropy_one_hot_is_zero():
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    h = _entropy(one_hot).item()
    assert h == pytest.approx(0.0, abs=1e-6)


def test_select_top_n_keeps_correct_indices():
    dist = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    n = torch.tensor([2])
    weights = _select_top_n(dist, n)
    # Should keep indices 0 and 1, renormalized: 0.5/0.8, 0.3/0.8
    assert weights[0, 2].item() == 0.0
    assert weights[0, 3].item() == 0.0
    assert weights[0, 0].item() == pytest.approx(0.5 / 0.8, rel=1e-5)
    assert weights[0, 1].item() == pytest.approx(0.3 / 0.8, rel=1e-5)


def test_select_top_n_per_sample():
    """Different samples can have different N."""
    dist = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.4, 0.3, 0.2, 0.1]])
    n = torch.tensor([1, 3])
    weights = _select_top_n(dist, n)
    # Row 0: only top-1 active.
    assert (weights[0] > 0).sum().item() == 1
    # Row 1: top-3 active.
    assert (weights[1] > 0).sum().item() == 3


def test_entropy_gate_thresholds():
    gate = EntropyGate(entropy_low=0.5, entropy_high=1.0)
    e = torch.tensor([0.1, 0.7, 1.5])
    n = gate(e)
    assert n.tolist() == [1, 2, 3]


def test_entropy_gate_validates_thresholds():
    with pytest.raises(ValueError):
        EntropyGate(entropy_low=1.0, entropy_high=0.5)


def test_prototype_memory_first_update_initializes_directly():
    mem = PrototypeMemory(num_experts=3, embed_dim=4, max_tasks=5)
    samples = torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    mem.update_expert(0, samples)
    expected = samples.mean(dim=0)
    assert torch.allclose(mem.expert_prototypes[0], expected)
    assert bool(mem.expert_initialized[0]) is True


def test_prototype_memory_ema_after_init():
    mem = PrototypeMemory(num_experts=2, embed_dim=2, max_tasks=2, expert_ema_beta=0.5)
    s1 = torch.tensor([[2.0, 0.0]])
    s2 = torch.tensor([[0.0, 4.0]])
    mem.update_expert(0, s1)  # init -> [2, 0]
    mem.update_expert(0, s2)  # EMA -> 0.5*[2,0] + 0.5*[0,4] = [1, 2]
    assert torch.allclose(mem.expert_prototypes[0], torch.tensor([1.0, 2.0]))


def test_prototype_router_uniform_when_uninitialized():
    mem = PrototypeMemory(num_experts=4, embed_dim=8, max_tasks=2)
    gate = EntropyGate(0.4, 0.9)
    router = PrototypeRouter(mem, gate, num_experts=4, temperature=0.5)
    r = torch.randn(3, 8)
    decision = router.route(r)
    assert torch.allclose(
        decision.distribution, torch.full((3, 4), 0.25), atol=1e-6
    )


def test_prototype_router_routes_to_nearest_prototype():
    mem = PrototypeMemory(num_experts=3, embed_dim=4, max_tasks=2)
    # Initialize prototypes to canonical basis vectors.
    e0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    e1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    e2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mem.update_expert(0, e0)
    mem.update_expert(1, e1)
    mem.update_expert(2, e2)

    gate = EntropyGate(0.4, 0.9)
    router = PrototypeRouter(mem, gate, num_experts=3, temperature=0.1)
    # Input strongly aligned with expert 1 should route there.
    r = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    decision = router.route(r)
    assert decision.distribution[0, 1].item() > 0.9


def test_fixed_topk_activates_exactly_k():
    mem = PrototypeMemory(num_experts=4, embed_dim=4, max_tasks=2)
    for k in range(4):
        v = torch.zeros(1, 4); v[0, k] = 1.0
        mem.update_expert(k, v)

    router = FixedTopKRouter(mem, num_experts=4, k=2, temperature=0.5)
    r = torch.randn(3, 4)
    decision = router.route(r)
    assert (decision.weights > 0).sum(dim=-1).tolist() == [2, 2, 2]
