"""Auxiliary routing loss.

Implements equations (9) and (10) from the interim report where we talk about routing loss:

    target q̃_k = softmax(cos(φ(c_t), p_k) / τ) over experts k
    L_route = KL(q̃ || q(x))

The target uses the *current task* prototype c_t and the *current expert*
prototypes p_k, so it provides a within-task consistency signal. φ is the
shared `RoutingProjection` also used by `PrototypeRouter` to compute q(x):
applying φ to c_t keeps the target consistent with the predicted distribution
(both live in the projected space) and routes gradient through φ on the target
side — though we detach the target before calling `kl_div` so the loss only
pushes q(x) toward q̃ rather than collapsing both onto each other.

Without φ, both q(x) and q̃ are gradient-free (frozen backbone + buffer
prototypes) and `lambda_route * loss_route` contributes nothing to training.
With φ, the predicted side carries a gradient through the projection, which is
what makes eqs. (9)-(10) actually train the router.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from clr_routing.models.router import PrototypeMemory, RoutingProjection, _safe_normalize


class RoutingKLLoss(nn.Module):
    """KL(target || prediction) for routing supervision.

    Convention: KLDivLoss expects log-probabilities for the input and
    probabilities for the target. We pass log(q(x)) and target q̃.
    """

    def __init__(
        self,
        memory: PrototypeMemory,
        temperature: float = 0.5,
        projection: RoutingProjection | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._temperature = temperature
        # Stored as a submodule attribute. The same Parameter objects are
        # registered on the router too, so `learner.parameters()` already
        # picks them up — listing them here is redundant but harmless and
        # keeps device-transfer of this module self-contained.
        self.projection = projection

    def forward(self, distribution: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute the routing KL loss.

        Args:
            distribution: (B, E) softmax routing distribution q(x).
            task_id: Current task identifier for selecting c_t.

        Returns:
            Scalar loss. Returns 0 if the task or any expert prototype is
            uninitialized (cannot compute a meaningful target yet).
        """
        max_tasks = int(self._memory.task_initialized.shape[0])
        if not 0 <= task_id < max_tasks:
            raise IndexError(
                f"task_id={task_id} out of range for task buffer of size {max_tasks}"
            )
        if not bool(self._memory.task_initialized[task_id]):
            return distribution.new_zeros(())
        if not bool(self._memory.expert_initialized.all()):
            return distribution.new_zeros(())

        # Pull the raw (unnormalized) task prototype so we can apply the
        # projection in the same r-space the router uses, then normalize.
        c_t_raw = self._memory.task_prototypes[task_id]  # (D,)
        if self.projection is not None:
            c_t_raw = self.projection(c_t_raw.unsqueeze(0)).squeeze(0)
        c_t = _safe_normalize(c_t_raw.unsqueeze(0)).squeeze(0)  # (D,)
        p = self._memory.expert_prototypes_normalized()  # (E, D)
        target_logits = (p @ c_t) / self._temperature  # (E,)
        target = torch.softmax(target_logits, dim=-1)  # (E,)
        # Detach the target so the loss only pushes q(x) → q̃ rather than
        # also moving q̃ to meet q(x); otherwise the projection could
        # trivially satisfy the KL by collapsing both sides together.
        target = target.detach()
        target = target.unsqueeze(0).expand_as(distribution)  # (B, E)

        log_q = (distribution.clamp_min(1e-12)).log()
        return F.kl_div(log_q, target, reduction="batchmean")
