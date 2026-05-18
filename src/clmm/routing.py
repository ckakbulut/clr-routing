import math

import torch
import torch.nn.functional as F

from clmm.config import Config


def routing_entropy(q: torch.Tensor):
    return -(q * torch.log(q.clamp_min(1e-8))).sum(dim=-1)


def normalize_entropy(ent: torch.Tensor, num_experts: int):
    return ent / math.log(num_experts)


def select_experts(q: torch.Tensor, mode: str, cfg: Config):
    """Return selected experts [B, S], normalized weights [B, S], entropy [B]."""
    ent = routing_entropy(q)

    if cfg.use_normalized_entropy:
        ent = normalize_entropy(ent, cfg.num_experts)

    B = q.shape[0]

    if mode == "entropy_adaptive":
        k_per_sample = torch.where(
            ent < cfg.entropy_threshold_low,
            torch.ones_like(ent, dtype=torch.long),
            torch.where(ent < cfg.entropy_threshold_high,
                        torch.full_like(ent, 2, dtype=torch.long),
                        torch.full_like(ent, 3, dtype=torch.long))
        )
        max_k = int(k_per_sample.max().item())
    elif mode == "top1":
        k_per_sample = torch.ones(B, device=q.device, dtype=torch.long)
        max_k = 1
    elif mode == "top2":
        k_per_sample = torch.full((B,), 2, device=q.device, dtype=torch.long)
        max_k = 2
    elif mode == "top3":
        k_per_sample = torch.full((B,), 3, device=q.device, dtype=torch.long)
        max_k = 3
    else:
        raise ValueError(f"Unknown routing mode: {mode}")

    vals, idx = torch.topk(q, k=max_k, dim=-1)
    mask = torch.arange(max_k, device=q.device).unsqueeze(0) < k_per_sample.unsqueeze(1)
    vals = vals * mask.to(vals.dtype)
    weights = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return idx, weights, ent


def load_balance_loss(q: torch.Tensor, cfg: Config):
    mean_q = q.mean(dim=0)
    return cfg.num_experts * torch.sum(mean_q ** 2)


def routing_kl_loss(q: torch.Tensor, q_tilde: torch.Tensor):
    """
    KL(q_tilde || q) averaged over batch.

    q:       [B, K]
    q_tilde: [B, K]
    """
    q_tilde = q_tilde.detach().to(q.device)

    return torch.sum(
        q_tilde * (
            torch.log(q_tilde.clamp_min(1e-8)) -
            torch.log(q.clamp_min(1e-8))
        ),
        dim=-1
    ).mean()
