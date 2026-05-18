"""Tests for PrototypeManager."""

import pytest
import torch
import torch.nn.functional as F

from clmm.config import Config
from clmm.model import PrototypeManager


def make_proto_mgr(num_experts=3, z_dim=8):
    cfg = Config()
    cfg.num_experts = num_experts
    cfg.beta_proto = 0.1
    cfg.gamma_task_proto = 0.1
    cfg.tau = 0.1
    device = torch.device("cpu")
    return PrototypeManager(cfg, z_dim=z_dim, device=device)


def test_assign_class_to_first_unused_expert():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    proto = torch.randn(4)
    pm.assign_class_to_expert(cls_id=0, class_proto=proto)
    assert pm.class_to_expert[0] == 0
    assert bool(pm.expert_initialized[0])
    assert not bool(pm.expert_initialized[1])
    assert not bool(pm.expert_initialized[2])


def test_assign_class_fills_experts_in_order():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    for i in range(3):
        pm.assign_class_to_expert(cls_id=i, class_proto=torch.randn(4))
    assert pm.class_to_expert[0] == 0
    assert pm.class_to_expert[1] == 1
    assert pm.class_to_expert[2] == 2
    assert pm.expert_initialized.all()


def test_assign_class_uses_closest_when_full():
    pm = make_proto_mgr(num_experts=2, z_dim=4)
    pm.assign_class_to_expert(0, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    pm.assign_class_to_expert(1, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    # All experts initialized; class 2 should go to closest expert
    pm.assign_class_to_expert(2, torch.tensor([0.9, 0.1, 0.0, 0.0]))
    assert pm.class_to_expert[2] == 0


def test_update_class_initializes_new_class():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    y = torch.tensor([0])
    z = torch.randn(1, 4)
    pm.update_class(y, z)
    assert 0 in pm.class_prototypes
    assert 0 in pm.class_to_expert


def test_update_class_ema_updates_existing():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    y = torch.tensor([0])
    z1 = torch.randn(1, 4)
    pm.update_class(y, z1)
    proto_before = pm.class_prototypes[0].clone()

    z2 = torch.randn(1, 4)
    pm.update_class(y, z2)
    proto_after = pm.class_prototypes[0]
    assert not torch.allclose(proto_after, proto_before)


def test_update_experts_updates_initialized_only():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    pm.expert_initialized[0] = True
    pm.expert_prototypes[0] = F.normalize(torch.randn(4), dim=0)

    z = torch.randn(4, 4)
    selected = torch.tensor([[0], [0], [1], [1]])
    before = pm.expert_prototypes[0].clone()
    pm.update_experts(z, selected)

    # Expert 0 should change (initialized + samples assigned)
    assert not torch.allclose(pm.expert_prototypes[0], before)
    # Expert 2 should not change (not initialized)
    assert not bool(pm.expert_initialized[2])


def test_routing_target_returns_none_when_no_classes():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    y = torch.tensor([0, 1])
    q_tilde, mask = pm.routing_target_from_labels(y, tau=0.1)
    assert q_tilde is None
    assert mask is None


def test_routing_target_shape_after_class_init():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    y = torch.tensor([0, 1, 0])
    z = torch.randn(3, 4)
    pm.update_class(y, z)

    q_tilde, mask = pm.routing_target_from_labels(y, tau=0.1)
    assert q_tilde is not None
    assert q_tilde.shape == (3, 3)
    assert mask.shape == (3,)
    assert mask.all()


def test_routing_target_sums_to_one():
    pm = make_proto_mgr(num_experts=3, z_dim=4)
    y = torch.tensor([0])
    z = torch.randn(1, 4)
    pm.update_class(y, z)

    q_tilde, mask = pm.routing_target_from_labels(y, tau=0.1)
    assert q_tilde[0].sum().item() == pytest.approx(1.0, abs=1e-5)
