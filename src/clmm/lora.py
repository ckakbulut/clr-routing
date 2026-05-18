from typing import Optional

import torch
import torch.nn as nn


class ExpertLoRALinear(nn.Module):
    """Frozen base Linear plus K LoRA expert adapters.

    Supports per-sample selected experts and weights.
    x shape may be [B, D] or [B, N, D].
    selected_experts shape: [B, S]
    selected_weights shape: [B, S]
    """
    def __init__(self, base: nn.Linear, num_experts: int, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = base.in_features
        out_f = base.out_features

        # A: down projection, B: up projection. Initialize B to zero, standard LoRA behavior.
        self.A = nn.Parameter(torch.randn(num_experts, in_f, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(num_experts, rank, out_f))

        self.enabled = False
        self.selected_experts = None
        self.selected_weights = None

    def set_routing(self, selected_experts: Optional[torch.Tensor], selected_weights: Optional[torch.Tensor]):
        if selected_experts is None:
            self.enabled = False
            self.selected_experts = None
            self.selected_weights = None
        else:
            self.enabled = True
            self.selected_experts = selected_experts
            self.selected_weights = selected_weights

    def forward(self, x):
        y = self.base(x)
        if not self.enabled or self.selected_experts is None:
            return y

        Bsz = x.shape[0]
        selected = self.selected_experts.to(x.device)  # [B, S]
        weights = self.selected_weights.to(x.device).to(x.dtype)  # [B, S]
        S = selected.shape[1]

        delta_total = torch.zeros_like(y)
        for slot in range(S):
            e_idx = selected[:, slot]  # [B]
            w = weights[:, slot]
            A = self.A[e_idx]          # [B, in, r]
            Bmat = self.B[e_idx]       # [B, r, out]

            if x.dim() == 3:
                # x: [B, N, in], A: [B, in, r] -> [B, N, r]
                down = torch.einsum("bni,bir->bnr", x, A)
                up = torch.einsum("bnr,bro->bno", down, Bmat)
                delta_total = delta_total + up * w.view(Bsz, 1, 1)
            elif x.dim() == 2:
                down = torch.einsum("bi,bir->br", x, A)
                up = torch.einsum("br,bro->bo", down, Bmat)
                delta_total = delta_total + up * w.view(Bsz, 1)
            else:
                raise ValueError(f"Unsupported input dim for LoRA linear: {x.shape}")

        return y + self.scaling * delta_total


def replace_vit_mlp_with_lora(vit: nn.Module, cfg):
    count = 0
    for blk in vit.blocks:
        blk.mlp.fc1 = ExpertLoRALinear(blk.mlp.fc1, cfg.num_experts, cfg.lora_rank, cfg.lora_alpha)
        blk.mlp.fc2 = ExpertLoRALinear(blk.mlp.fc2, cfg.num_experts, cfg.lora_rank, cfg.lora_alpha)
        count += 2
    print(f"Replaced {count} ViT MLP Linear layers with Expert LoRA layers.")


def iter_lora_layers(module):
    for m in module.modules():
        if isinstance(m, ExpertLoRALinear):
            yield m
