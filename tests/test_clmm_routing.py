"""Tests for routing utilities: select_experts, entropy, KL loss."""

import math

import pytest
import torch

from clmm.config import Config
from clmm.routing import (
    routing_entropy,
    normalize_entropy,
    select_experts,
    load_balance_loss,
    routing_kl_loss,
)


def make_cfg(**kwargs) -> Config:
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_routing_entropy_uniform():
    K = 4
    q = torch.full((2, K), 1.0 / K)
    ent = routing_entropy(q)
    expected = math.log(K)
    assert ent.shape == (2,)
    assert pytest.approx(ent[0].item(), rel=1e-5) == expected


def test_routing_entropy_one_hot():
    q = torch.tensor([[1.0, 0.0, 0.0]])
    ent = routing_entropy(q)
    assert ent[0].item() == pytest.approx(0.0, abs=1e-6)


def test_normalize_entropy():
    K = 3
    raw = torch.tensor([math.log(K)])
    norm = normalize_entropy(raw, K)
    assert norm[0].item() == pytest.approx(1.0, rel=1e-5)


def test_select_experts_top1():
    cfg = make_cfg(num_experts=3, routing_mode="top1", use_normalized_entropy=False)
    q = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    idx, weights, ent = select_experts(q, "top1", cfg)
    assert idx.shape == (2, 1)
    assert weights.shape == (2, 1)
    assert idx[0, 0].item() == 0
    assert idx[1, 0].item() == 1
    assert weights.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])


def test_select_experts_top2():
    cfg = make_cfg(num_experts=3, routing_mode="top2", use_normalized_entropy=False)
    q = torch.tensor([[0.7, 0.2, 0.1]])
    idx, weights, ent = select_experts(q, "top2", cfg)
    assert idx.shape == (1, 2)
    assert (weights > 0).sum().item() == 2
    assert weights.sum(dim=-1).item() == pytest.approx(1.0, rel=1e-5)


def test_select_experts_top3():
    cfg = make_cfg(num_experts=3, routing_mode="top3", use_normalized_entropy=False)
    q = torch.tensor([[0.5, 0.3, 0.2]])
    idx, weights, ent = select_experts(q, "top3", cfg)
    assert idx.shape == (1, 3)
    assert (weights > 0).sum().item() == 3


def test_select_experts_entropy_adaptive_low_entropy_gives_1():
    cfg = make_cfg(
        num_experts=3,
        entropy_threshold_low=0.5,
        entropy_threshold_high=0.9,
        use_normalized_entropy=True,
    )
    # Very concentrated -> low entropy -> 1 expert
    q = torch.tensor([[0.98, 0.01, 0.01]])
    idx, weights, ent = select_experts(q, "entropy_adaptive", cfg)
    assert (weights > 0).sum().item() == 1


def test_select_experts_entropy_adaptive_high_entropy_gives_3():
    cfg = make_cfg(
        num_experts=3,
        entropy_threshold_low=0.5,
        entropy_threshold_high=0.9,
        use_normalized_entropy=True,
    )
    # Uniform -> max entropy (normalized = 1.0) -> 3 experts
    q = torch.full((1, 3), 1.0 / 3)
    idx, weights, ent = select_experts(q, "entropy_adaptive", cfg)
    assert (weights > 0).sum().item() == 3


def test_select_experts_unknown_mode_raises():
    cfg = make_cfg(num_experts=3, use_normalized_entropy=False)
    q = torch.full((1, 3), 1.0 / 3)
    with pytest.raises(ValueError):
        select_experts(q, "invalid_mode", cfg)


def test_weights_sum_to_one():
    cfg = make_cfg(num_experts=3, use_normalized_entropy=False)
    q = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    for mode in ["top1", "top2", "top3"]:
        _, weights, _ = select_experts(q, mode, cfg)
        assert weights.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0], rel=1e-5)


def test_load_balance_loss_shape():
    cfg = make_cfg(num_experts=3)
    q = torch.softmax(torch.randn(8, 3), dim=-1)
    loss = load_balance_loss(q, cfg)
    assert loss.shape == ()


def test_routing_kl_loss_zero_when_same():
    q = torch.softmax(torch.randn(4, 3), dim=-1)
    loss = routing_kl_loss(q, q)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_routing_kl_loss_positive_when_different():
    q = torch.softmax(torch.randn(4, 3), dim=-1)
    q_tilde = torch.softmax(torch.randn(4, 3), dim=-1)
    loss = routing_kl_loss(q, q_tilde)
    assert loss.item() >= 0.0


def test_routing_kl_loss_detaches_target():
    q = torch.softmax(torch.randn(4, 3), dim=-1).requires_grad_(True)
    q_tilde = torch.softmax(torch.randn(4, 3), dim=-1).requires_grad_(True)
    loss = routing_kl_loss(q, q_tilde)
    loss.backward()
    assert q_tilde.grad is None
