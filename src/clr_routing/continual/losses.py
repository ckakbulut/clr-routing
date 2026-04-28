"""Auxiliary routing loss.

Implements equations (9) and (10) from the interim report where we talk about routing loss:

    target q̃_k = softmax(cos(c_t, p_k) / τ) over experts k
    L_route = KL(q̃ || q(x))

The target uses the *current task* prototype c_t and the *current expert*
prototypes p_k, so it provides a within-task consistency signal.

Known issue — KL routing loss collapse (lambda_route = 0):
    Both q(x) and q̃ are computed from the frozen backbone representation r(x)
    and the EMA-updated prototype buffers. The buffers are registered as
    `register_buffer` (not `Parameter`) and updated under `@torch.no_grad()`,
    and the backbone is frozen; therefore neither q(x) nor q̃ carries a gradient
    w.r.t. any trainable parameter. `loss_total.backward()` produces zero
    gradient from this term, so the proposed routing supervision is a silent
    no-op during training. We disable it by setting `lambda_route = 0` at the
    config level rather than altering the methodology.
    TODO: revisit routing supervision so eqs. (9)-(10) actually contribute
    gradient — e.g., a learned projection on r(x), or learnable expert
    prototypes — and then restore lambda_route > 0.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from clr_routing.models.router import PrototypeMemory, _safe_normalize


class RoutingKLLoss(nn.Module):
    """KL(target || prediction) for routing supervision.

    Convention: KLDivLoss expects log-probabilities for the input and
    probabilities for the target. We pass log(q(x)) and target q̃.
    """

    def __init__(self, memory: PrototypeMemory, temperature: float = 0.5) -> None:
        super().__init__()
        self._memory = memory
        self._temperature = temperature

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

        c_t = self._memory.task_prototype_normalized(task_id)  # (D,)
        p = self._memory.expert_prototypes_normalized()  # (E, D)
        target_logits = (p @ c_t) / self._temperature  # (E,)
        target = torch.softmax(target_logits, dim=-1)  # (E,)
        target = target.unsqueeze(0).expand_as(distribution)  # (B, E)

        log_q = (distribution.clamp_min(1e-12)).log()
        return F.kl_div(log_q, target, reduction="batchmean")
