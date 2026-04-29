import torch
import torch.nn.functional as F


@torch.no_grad()
def pairwise_prototype_cosines(
    prototypes: torch.Tensor, init_mask: torch.Tensor | None = None
) -> dict:
    """Pairwise cosine summary across *initialized* expert prototypes."""
    if init_mask is not None:
        prototypes = prototypes[init_mask]
    if prototypes.size(0) < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n_initialized": prototypes.size(0),
        }

    p = F.normalize(prototypes, dim=-1, eps=1e-8)
    cos_matrix = p @ p.T
    K = cos_matrix.size(0)
    iu = torch.triu_indices(K, K, offset=1)
    pairs = cos_matrix[iu[0], iu[1]]
    return {
        "mean": pairs.mean().item(),
        "std": pairs.std().item() if pairs.numel() > 1 else 0.0,
        "min": pairs.min().item(),
        "max": pairs.max().item(),
        "n_initialized": K,
    }


@torch.no_grad()
def routing_score_stats(scores: torch.Tensor, init_mask: torch.Tensor | None = None) -> dict:
    """Spread of pre-softmax, pre-mask scores. Restricts to initialized columns."""
    if init_mask is not None:
        scores = scores[:, init_mask]
    if scores.size(1) < 2:
        return {"score_std_per_input_mean": 0.0, "score_range_per_input_mean": 0.0}
    return {
        "score_std_per_input_mean": scores.std(dim=1).mean().item(),
        "score_range_per_input_mean": (scores.max(dim=1).values - scores.min(dim=1).values)
        .mean()
        .item(),
    }


@torch.no_grad()
def representation_anisotropy(reprs: torch.Tensor, max_n: int = 128) -> float:
    """Mean off-diagonal cosine across a batch of input reprs."""
    r = F.normalize(reprs[:max_n], dim=-1, eps=1e-8)
    sim = r @ r.T
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    return sim[mask].mean().item()
