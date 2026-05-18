from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from clmm.config import Config
from clmm.lora import ExpertLoRALinear, replace_vit_mlp_with_lora, iter_lora_layers
from clmm.routing import select_experts


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, r):
        return F.normalize(self.net(r), dim=-1)


class PrototypeManager:
    def __init__(self, cfg: Config, z_dim: int, device):
        self.cfg = cfg
        self.device = device
        self.z_dim = z_dim

        self.expert_prototypes = F.normalize(
            torch.randn(cfg.num_experts, z_dim, device=device),
            dim=-1
        )

        self.expert_initialized = torch.zeros(
            cfg.num_experts,
            dtype=torch.bool,
            device=device
        )

        self.class_prototypes: Dict[int, torch.Tensor] = {}
        self.class_to_expert: Dict[int, int] = {}

    @torch.no_grad()
    def assign_class_to_expert(self, cls_id: int, class_proto: torch.Tensor):
        """
        Assign a new class to an expert.

        Rule:
        1. If unused experts exist, use the first unused expert.
        2. Otherwise, use the closest existing expert prototype.
        """
        class_proto = F.normalize(class_proto.detach(), dim=0)

        unused = torch.where(~self.expert_initialized)[0]

        if len(unused) > 0:
            expert_id = int(unused[0].item())

            self.class_to_expert[cls_id] = expert_id
            self.expert_initialized[expert_id] = True

            self.expert_prototypes[expert_id] = class_proto.clone()

        else:
            sims = F.cosine_similarity(
                class_proto.unsqueeze(0),
                self.expert_prototypes,
                dim=-1
            )
            expert_id = int(torch.argmax(sims).item())
            self.class_to_expert[cls_id] = expert_id

    @torch.no_grad()
    def update_class(self, y: torch.Tensor, z: torch.Tensor):
        """
        Update class prototypes using EMA.
        New classes initialize or attach to experts.
        """
        y = y.detach()
        z = z.detach()

        for cls in y.unique():
            cls_id = int(cls.item())
            mask = (y == cls)

            mean_z = F.normalize(z[mask].mean(dim=0), dim=0)

            if cls_id not in self.class_prototypes:
                self.class_prototypes[cls_id] = mean_z.clone()
                self.assign_class_to_expert(cls_id, mean_z)
            else:
                c = self.class_prototypes[cls_id]
                c = (
                    (1 - self.cfg.gamma_task_proto) * c
                    + self.cfg.gamma_task_proto * mean_z
                )
                self.class_prototypes[cls_id] = F.normalize(c, dim=0)

    @torch.no_grad()
    def update_experts(self, z: torch.Tensor, selected_experts: torch.Tensor):
        """
        Update expert prototypes using samples routed to each expert.
        Only update initialized experts.
        """
        z_det = z.detach()

        for k in range(self.cfg.num_experts):
            if not bool(self.expert_initialized[k]):
                continue

            mask = (selected_experts == k).any(dim=1)

            if mask.any():
                mean_z = F.normalize(z_det[mask].mean(dim=0), dim=0)
                p = self.expert_prototypes[k]

                p = (1 - self.cfg.beta_proto) * p + self.cfg.beta_proto * mean_z
                self.expert_prototypes[k] = F.normalize(p, dim=0)

    @torch.no_grad()
    def routing_target_from_labels(self, y: torch.Tensor, tau: float):
        """
        Build routing targets from class prototypes and expert assignments.

        Returns:
            q_tilde: [B, K]
            valid_mask: [B]
        """
        y = y.detach()
        p = self.expert_prototypes.detach()

        targets = []
        valid_mask = []

        for cls in y:
            cls_id = int(cls.item())

            if cls_id not in self.class_prototypes:
                targets.append(torch.zeros(self.cfg.num_experts, device=p.device))
                valid_mask.append(False)
                continue

            c = self.class_prototypes[cls_id].detach().to(p.device)

            sims = F.cosine_similarity(c.unsqueeze(0), p, dim=-1) * 10

            q_tilde = F.softmax(sims / tau, dim=-1)

            if cls_id in self.class_to_expert:
                primary = self.class_to_expert[cls_id]
                one_hot = torch.zeros_like(q_tilde)
                one_hot[primary] = 1.0

                q_tilde = 0.6 * q_tilde + 0.4 * one_hot
                q_tilde = q_tilde / q_tilde.sum().clamp_min(1e-8)

            targets.append(q_tilde)
            valid_mask.append(True)

        q_tilde = torch.stack(targets, dim=0)
        valid_mask = torch.tensor(valid_mask, device=p.device, dtype=torch.bool)

        if not valid_mask.any():
            return None, None

        return q_tilde, valid_mask


class CLLoRAViT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.vit = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0
        )
        self.embed_dim = self.vit.num_features

        for p in self.vit.parameters():
            p.requires_grad = False

        replace_vit_mlp_with_lora(self.vit, cfg)

        self.router = nn.Identity()

        self.classifier = nn.Linear(self.embed_dim, cfg.num_classes)

        self.prototype_manager = PrototypeManager(
            cfg,
            self.embed_dim,
            device=torch.device("cpu")
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device = next(self.parameters()).device
        self.prototype_manager.device = device

        self.prototype_manager.expert_prototypes = (
            self.prototype_manager.expert_prototypes.to(device)
        )

        self.prototype_manager.expert_initialized = (
            self.prototype_manager.expert_initialized.to(device)
        )

        self.prototype_manager.class_prototypes = {
            k: v.to(device)
            for k, v in self.prototype_manager.class_prototypes.items()
        }

        return self

    def set_lora_routing(
        self,
        selected_experts: Optional[torch.Tensor],
        selected_weights: Optional[torch.Tensor]
    ):
        for layer in iter_lora_layers(self.vit):
            layer.set_routing(selected_experts, selected_weights)

    def disable_lora(self):
        self.set_lora_routing(None, None)

    def patch_mean_features(self, x):
        tokens = self.vit.forward_features(x)

        if tokens.dim() == 3 and tokens.shape[1] > 1:
            patch_tokens = tokens[:, 1:, :]
            return patch_tokens.mean(dim=1)

        return tokens

    def route_from_x(self, x):
        self.disable_lora()

        r = self.patch_mean_features(x)
        z = F.normalize(r, dim=-1)

        p = self.prototype_manager.expert_prototypes.to(z.device)

        sims = F.cosine_similarity(
            z.unsqueeze(1),
            p.unsqueeze(0),
            dim=-1
        )

        q = F.softmax(sims / self.cfg.tau, dim=-1)

        return r, z, q

    def forward_with_routing(self, x, selected_experts, selected_weights):
        self.set_lora_routing(selected_experts, selected_weights)

        features = self.patch_mean_features(x)
        logits = self.classifier(features)

        return logits, features

    def forward(self, x):
        r, z, q = self.route_from_x(x)

        selected, weights, ent = select_experts(
            q,
            self.cfg.routing_mode,
            self.cfg
        )

        logits, features = self.forward_with_routing(x, selected, weights)

        return logits, {
            "r": r,
            "z": z,
            "q": q,
            "selected": selected,
            "weights": weights,
            "entropy": ent,
        }


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
