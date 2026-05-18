"""Tests for ClassBalancedBuffer and ReplayManager."""

import pytest
import torch

from clmm.config import Config
from clmm.replay import ClassBalancedBuffer, ReplayManager


def make_image(c=3, h=4, w=4):
    return torch.randn(c, h, w)


def test_buffer_add_and_sample():
    buf = ClassBalancedBuffer(capacity=10)
    for i in range(5):
        buf.add(make_image(), y=i % 2)
    assert len(buf) == 5
    result = buf.sample(3)
    assert result is not None
    xs, ys = result
    assert xs.shape[0] == 3


def test_buffer_respects_capacity():
    buf = ClassBalancedBuffer(capacity=5)
    for i in range(20):
        buf.add(make_image(), y=i % 3)
    assert len(buf) == 5


def test_buffer_sample_none_when_empty():
    buf = ClassBalancedBuffer(capacity=10)
    assert buf.sample(5) is None


def test_buffer_sample_zero_returns_none():
    buf = ClassBalancedBuffer(capacity=10)
    buf.add(make_image(), y=0)
    assert buf.sample(0) is None


def test_buffer_class_counts():
    buf = ClassBalancedBuffer(capacity=20)
    for _ in range(5):
        buf.add(make_image(), y=0)
    for _ in range(3):
        buf.add(make_image(), y=1)
    counts = buf.class_counts()
    assert counts[0] == 5
    assert counts[1] == 3


def test_buffer_class_balanced_replacement():
    buf = ClassBalancedBuffer(capacity=4)
    for _ in range(4):
        buf.add(make_image(), y=0)
    # All 4 slots taken by class 0; adding class 1 should replace a class-0 sample
    buf.add(make_image(), y=1)
    counts = buf.class_counts()
    assert counts.get(1, 0) == 1
    assert len(buf) == 4


def make_cfg(**kwargs):
    cfg = Config()
    cfg.num_experts = 3
    cfg.buffer_size_per_expert = 10
    cfg.global_buffer_size = 20
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_replay_manager_add_batch_fills_expert_buffer():
    cfg = make_cfg()
    rm = ReplayManager(cfg)

    B = 6
    x = torch.randn(B, 3, 4, 4)
    y = torch.arange(B)
    selected = torch.zeros(B, 1, dtype=torch.long)  # all routed to expert 0

    rm.add_batch(x, y, selected)

    assert len(rm.expert_buffers[0]) == B
    assert len(rm.expert_buffers[1]) == 0
    assert len(rm.global_buffer) == B


def test_replay_manager_add_batch_routes_to_correct_expert():
    cfg = make_cfg()
    rm = ReplayManager(cfg)

    x = torch.randn(3, 3, 4, 4)
    y = torch.arange(3)
    selected = torch.tensor([[0], [1], [2]])

    rm.add_batch(x, y, selected)

    assert len(rm.expert_buffers[0]) == 1
    assert len(rm.expert_buffers[1]) == 1
    assert len(rm.expert_buffers[2]) == 1


def test_replay_manager_sample_expert_mode():
    cfg = make_cfg()
    rm = ReplayManager(cfg)

    x = torch.randn(6, 3, 4, 4)
    y = torch.arange(6)
    selected = torch.zeros(6, 1, dtype=torch.long)
    rm.add_batch(x, y, selected)

    active = torch.tensor([[0]])
    result = rm.sample_for_selected(active, total_n=4, mode="expert")
    assert result is not None
    xr, yr = result
    assert xr.shape[0] <= 4


def test_replay_manager_sample_global_mode():
    cfg = make_cfg()
    rm = ReplayManager(cfg)

    x = torch.randn(5, 3, 4, 4)
    y = torch.arange(5)
    selected = torch.zeros(5, 1, dtype=torch.long)
    rm.add_batch(x, y, selected)

    result = rm.sample_for_selected(selected, total_n=3, mode="global")
    assert result is not None
    xr, yr = result
    assert xr.shape[0] == 3


def test_replay_manager_sample_none_mode():
    cfg = make_cfg()
    rm = ReplayManager(cfg)
    selected = torch.zeros(4, 1, dtype=torch.long)
    result = rm.sample_for_selected(selected, total_n=4, mode="none")
    assert result is None


def test_replay_manager_expert_sizes():
    cfg = make_cfg()
    rm = ReplayManager(cfg)
    sizes = rm.expert_sizes()
    assert sizes == [0, 0, 0]
