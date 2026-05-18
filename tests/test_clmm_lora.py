"""Tests for ExpertLoRALinear."""

import pytest
import torch
import torch.nn as nn

from clmm.lora import ExpertLoRALinear, iter_lora_layers


def make_layer(in_f=8, out_f=16, num_experts=3, rank=4, alpha=4):
    base = nn.Linear(in_f, out_f, bias=True)
    return ExpertLoRALinear(base, num_experts=num_experts, rank=rank, alpha=alpha)


def test_b_initialized_to_zero():
    layer = make_layer()
    assert layer.B.abs().sum().item() == 0.0


def test_base_params_frozen():
    layer = make_layer()
    for p in layer.base.parameters():
        assert not p.requires_grad


def test_disabled_by_default_returns_base_output():
    layer = make_layer(in_f=8, out_f=16)
    x = torch.randn(4, 8)
    out = layer(x)
    expected = layer.base(x)
    assert torch.allclose(out, expected)


def test_enabled_with_zero_b_still_returns_base_output():
    layer = make_layer(in_f=8, out_f=16, num_experts=2, rank=4)
    selected = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    weights = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]])
    layer.set_routing(selected, weights)

    x = torch.randn(4, 8)
    out = layer(x)
    expected = layer.base(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_enabled_with_nonzero_b_adds_delta():
    layer = make_layer(in_f=8, out_f=16, num_experts=2, rank=4, alpha=4)
    nn.init.normal_(layer.B, std=0.1)

    selected = torch.tensor([[0], [1]])
    weights = torch.ones(2, 1)
    layer.set_routing(selected, weights)

    x = torch.randn(2, 8)
    out = layer(x)
    base_out = layer.base(x)
    assert not torch.allclose(out, base_out, atol=1e-6)


def test_2d_input_shape():
    layer = make_layer(in_f=8, out_f=16, num_experts=2, rank=4)
    nn.init.normal_(layer.B, std=0.1)
    selected = torch.tensor([[0], [1], [0]])
    weights = torch.ones(3, 1)
    layer.set_routing(selected, weights)

    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 16)


def test_3d_input_shape():
    layer = make_layer(in_f=8, out_f=16, num_experts=2, rank=4)
    nn.init.normal_(layer.B, std=0.1)
    selected = torch.tensor([[0], [1]])
    weights = torch.ones(2, 1)
    layer.set_routing(selected, weights)

    x = torch.randn(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 16)


def test_set_routing_none_disables():
    layer = make_layer()
    selected = torch.tensor([[0]])
    weights = torch.ones(1, 1)
    layer.set_routing(selected, weights)
    assert layer.enabled

    layer.set_routing(None, None)
    assert not layer.enabled
    assert layer.selected_experts is None


def test_iter_lora_layers_finds_all():
    class FakeVit(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                ExpertLoRALinear(nn.Linear(4, 8), 2, 2, 2.0),
                ExpertLoRALinear(nn.Linear(8, 4), 2, 2, 2.0),
            ])

    vit = FakeVit()
    found = list(iter_lora_layers(vit))
    assert len(found) == 2
