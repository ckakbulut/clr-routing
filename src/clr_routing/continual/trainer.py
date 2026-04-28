"""ContinualTrainer: orchestrates the task loop, replay, prototype updates, and eval.

The trainer is the single integration point for all swappable components. It
takes a `ContinualLearner`, a `ReplayBuffer`, an optimizer, and a
`RoutingKLLoss` (optional) — each can be swapped at the config level for
ablations without modifying this code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from clr_routing.continual.buffer import ReplayBuffer, PerExpertReplayBuffer
from clr_routing.continual.losses import RoutingKLLoss
from clr_routing.continual.metrics import ContinualMetrics
from clr_routing.data.cifar100_split import SplitCIFAR100, make_loader
from clr_routing.models import ContinualLearner
from clr_routing.models.router import PrototypeMemory
from clr_routing.utils.device import DeviceInfo


@dataclass
class TrainerConfig:
    epochs_per_task: int = 5
    train_batch_size: int = 64
    eval_batch_size: int = 128
    replay_batch_size: int = 32
    num_workers: int = 4

    lambda_replay: float = 1.0
    # Routing KL loss is disabled (lambda_route = 0) because the routing
    # distribution q(x) is built from the frozen backbone representation and
    # EMA-updated prototype buffers, neither of which carry a gradient. The
    # KL(q̃ || q) term therefore has no gradient path and contributes nothing
    # to training — multiplying by 0 makes that explicit.
    # TODO: re-enable once the router has a trainable parameter on the path
    # from r(x) to q(x) (e.g., a learned projection or learnable prototypes),
    # so eqs. (9)-(10) of the report actually backpropagate.
    lambda_route: float = 0.0

    # How many representations to use per prototype update step.
    prototype_update_period: int = 1
    log_every: int = 50


class ContinualTrainer:
    """Owns training and evaluation over the task stream."""

    def __init__(
        self,
        learner: ContinualLearner,
        memory: PrototypeMemory,
        buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        metrics: ContinualMetrics,
        config: TrainerConfig,
        device_info: DeviceInfo,
        routing_loss: RoutingKLLoss | None = None,
        log_callback: Callable[[dict, int], None] | None = None,
    ) -> None:
        self._device_info = device_info
        self._device = device_info.device
        self._learner = learner.to(self._device)
        self._memory = memory.to(self._device)
        self._buffer = buffer
        self._optimizer = optimizer
        self._metrics = metrics
        self._cfg = config
        self._routing_loss = routing_loss
        self._log = log_callback or (lambda metrics, step: None)
        self._global_step = 0

    # ---------- task loop ----------

    def run(self, stream: SplitCIFAR100) -> ContinualMetrics:
        """Train sequentially over all tasks, evaluating on all seen tasks
        after each task is finished.
        """
        test_loaders: dict[int, DataLoader] = {}
        for task_id, (train_ds, _) in enumerate(stream):
            train_loader = make_loader(
                train_ds,
                batch_size=self._cfg.train_batch_size,
                shuffle=True,
                num_workers=self._cfg.num_workers,
                pin_memory=self._device_info.pin_memory,
            )
            self.train_task(task_id, train_loader)

            for prior_id in range(task_id + 1):
                if prior_id not in test_loaders:
                    _, prior_test = stream.get_task(prior_id)
                    test_loaders[prior_id] = make_loader(
                        prior_test,
                        batch_size=self._cfg.eval_batch_size,
                        shuffle=False,
                        num_workers=self._cfg.num_workers,
                        pin_memory=self._device_info.pin_memory,
                    )
                acc = self.evaluate(test_loaders[prior_id])
                self._metrics.record(task_id, prior_id, acc)

            snapshot = self._metrics.snapshot(task_id)
            self._log(
                {
                    "continual/avg_accuracy": snapshot.average_accuracy,
                    "continual/avg_forgetting": snapshot.average_forgetting,
                    "continual/bwt": snapshot.backward_transfer,
                    "continual/task_completed": task_id,
                },
                self._global_step,
            )

        return self._metrics

    def train_task(self, task_id: int, loader: DataLoader) -> None:
        self._learner.train()
        non_blocking = self._device_info.supports_non_blocking
        for epoch in range(self._cfg.epochs_per_task):
            pbar = tqdm(loader, desc=f"task {task_id} epoch {epoch}", leave=False)
            for x, y in pbar:
                x = x.to(self._device, non_blocking=non_blocking)
                y = y.to(self._device, non_blocking=non_blocking)

                losses = self._step(task_id, x, y)
                self._global_step += 1

                if self._global_step % self._cfg.log_every == 0:
                    self._log({f"train/{k}": v for k, v in losses.items()}, self._global_step)

                pbar.set_postfix(loss=f"{losses['loss_total']:.3f}")

    def _step(self, task_id: int, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """One optimizer step on a current-task batch + optional replay batch."""
        self._optimizer.zero_grad(set_to_none=True)

        # --- representation (used for prototype updates and as router input) ---
        with torch.no_grad():
            r = self._learner.representation(x)
        self._memory.update_task(task_id, r)

        # --- bootstrap: ensure every expert is initialized before relying on
        # argmax-based assignment. With zero-init prototypes, the first uniform
        # decision would funnel every sample to expert 0 (argmax tie-break),
        # leaving the rest uninitialized indefinitely.
        is_bootstrap = not bool(self._memory.expert_initialized.all())
        if is_bootstrap:
            n_experts = int(self._memory.expert_initialized.shape[0])
            bootstrap_assignment = (
                torch.arange(x.shape[0], device=x.device) % n_experts
            )
            for k in range(n_experts):
                mask_k = bootstrap_assignment == k
                if mask_k.any():
                    self._memory.update_expert(int(k), r[mask_k].detach())

        # --- forward on current task (reuses precomputed r to avoid a second
        # backbone pass) ---
        logits, decision = self._learner(x, return_routing=True, representation=r)
        loss_task = F.cross_entropy(logits, y)

        # --- update expert prototypes by argmax routing assignment ---
        if is_bootstrap:
            assignment = bootstrap_assignment
        else:
            assignment = decision.distribution.argmax(dim=-1)
            for k in assignment.unique().tolist():
                mask = assignment == k
                if mask.any():
                    self._memory.update_expert(int(k), r[mask].detach())

        # --- store in replay buffer (CPU samples) ---
        self._buffer.store_batch(x, y, assignment)

        # --- replay loss ---
        loss_replay = x.new_zeros(())
        active = decision.weights.sum(dim=0).nonzero(as_tuple=False).squeeze(-1)
        replay = self._buffer.sample_batch(
            self._cfg.replay_batch_size,
            active_experts=active if isinstance(self._buffer, PerExpertReplayBuffer) else None,
        )
        if replay is not None:
            xr, yr = replay
            non_blocking = self._device_info.supports_non_blocking
            xr = xr.to(self._device, non_blocking=non_blocking)
            yr = yr.to(self._device, non_blocking=non_blocking)
            # Replay samples are stored already-augmented (one fixed crop/flip
            # per stored sample). Re-apply a per-sample random horizontal flip
            # at replay time to recover some augmentation diversity. Crop is
            # not re-applied because the cached tensor has already been cropped.
            if self._learner.training and xr.dim() == 4:
                flip_mask = torch.rand(xr.shape[0], device=xr.device) < 0.5
                if flip_mask.any():
                    flipped = torch.flip(xr, dims=[-1])
                    xr = torch.where(flip_mask.view(-1, 1, 1, 1), flipped, xr)
            replay_logits, _ = self._learner(xr, return_routing=True)
            loss_replay = F.cross_entropy(replay_logits, yr)

        # --- routing KL loss ---
        loss_route = x.new_zeros(())
        if self._routing_loss is not None:
            loss_route = self._routing_loss(decision.distribution, task_id)

        loss_total = (
            loss_task + self._cfg.lambda_replay * loss_replay + self._cfg.lambda_route * loss_route
        )
        loss_total.backward()
        self._optimizer.step()

        return {
            "loss_total": float(loss_total.detach()),
            "loss_task": float(loss_task.detach()),
            "loss_replay": float(loss_replay.detach()),
            "loss_route": float(loss_route.detach()),
            "routing_entropy_mean": float(decision.entropy.mean().detach()),
            "num_active_mean": float(decision.num_active.float().mean().detach()),
        }

    # ---------- evaluation ----------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        was_training = self._learner.training
        self._learner.eval()
        try:
            non_blocking = self._device_info.supports_non_blocking
            correct = 0
            total = 0
            for x, y in loader:
                x = x.to(self._device, non_blocking=non_blocking)
                y = y.to(self._device, non_blocking=non_blocking)
                logits = self._learner(x)
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum())
                total += y.shape[0]
            return correct / max(total, 1)
        finally:
            if was_training:
                self._learner.train()
