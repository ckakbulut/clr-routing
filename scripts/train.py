"""Main training entrypoint, configured via Hydra.

Usage:
    python scripts/train.py                              # default config
    python scripts/train.py experiment=baseline_top1     # baseline
    python scripts/train.py model.num_experts=8 train.lambda_route=0.5  # overrides
"""

from __future__ import annotations

import hydra
import json
import logging
import torch
from pathlib import Path
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from clr_routing.continual import (
    ContinualMetrics,
    ContinualTrainer,
    GlobalReplayBuffer,
    PerExpertReplayBuffer,
    RoutingKLLoss,
)
from clr_routing.continual.trainer import TrainerConfig
from clr_routing.data import SplitCIFAR100
from clr_routing.models import (
    ContinualLearner,
    EntropyGate,
    FixedTopKRouter,
    LoRAExpertBank,
    PrototypeMemory,
    PrototypeRouter,
    RoutingProjection,
    ViTBackbone,
)
from clr_routing.utils import WandBLogger, select_device, set_seed

log = logging.getLogger(__name__)


def build_routing_projection(cfg: DictConfig, embed_dim: int) -> RoutingProjection | None:
    """Build the routing projection if enabled. Only meaningful for the
    prototype router (FixedTopK has no routing supervision to drive).
    """
    proj_cfg = cfg.routing.get("projection", None)
    if proj_cfg is None or not proj_cfg.get("enabled", False):
        return None
    return RoutingProjection(
        embed_dim=embed_dim,
        hidden_dim=proj_cfg.get("hidden_dim", None),
        dropout=proj_cfg.get("dropout", 0.0),
    )


def build_router(
    cfg: DictConfig,
    memory: PrototypeMemory,
    projection: RoutingProjection | None = None,
):
    method = cfg.method
    if method == "prototype":
        gate = EntropyGate(
            entropy_low=cfg.routing.entropy_low,
            entropy_high=cfg.routing.entropy_high,
            max_n=cfg.routing.max_active,
        )
        return PrototypeRouter(
            memory=memory,
            gate=gate,
            num_experts=cfg.model.num_experts,
            temperature=cfg.routing.temperature,
            projection=projection,
        )
    if method.startswith("fixed_top"):
        k = int(method.replace("fixed_top", ""))
        return FixedTopKRouter(
            memory=memory,
            num_experts=cfg.model.num_experts,
            k=k,
            temperature=cfg.routing.temperature,
        )
    raise ValueError(f"Unknown method: {method}")


def build_buffer(cfg: DictConfig):
    if not cfg.replay.enabled:
        # Empty per-expert buffer with capacity 0 is a no-op replay source.
        return PerExpertReplayBuffer(num_experts=cfg.model.num_experts, per_expert_capacity=0)
    if cfg.replay.global_replay:
        total = cfg.replay.per_expert_capacity * cfg.model.num_experts
        return GlobalReplayBuffer(capacity=total)
    return PerExpertReplayBuffer(
        num_experts=cfg.model.num_experts,
        per_expert_capacity=cfg.replay.per_expert_capacity,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(cfg.seed)
    device_info = select_device(cfg.device)
    log.info("Using device: %s", device_info)
    print(f"[clr-routing] Using {device_info}")

    # --- data ---
    stream = SplitCIFAR100(
        num_tasks=cfg.data.num_tasks,
        classes_per_task=cfg.data.classes_per_task,
        seed=cfg.seed,
        root=cfg.data.root,
    )

    # --- model components ---
    backbone = ViTBackbone(
        model_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        freeze_base=True,
    )
    lora_bank = LoRAExpertBank(
        backbone=backbone,
        target_modules=backbone.target_modules,
        num_experts=cfg.model.num_experts,
        rank=cfg.model.lora_rank,
        alpha=cfg.model.lora_alpha,
    )
    memory = PrototypeMemory(
        num_experts=cfg.model.num_experts,
        embed_dim=backbone.embed_dim,
        max_tasks=cfg.data.num_tasks,
        expert_ema_beta=cfg.routing.prototype_ema_beta,
        task_ema_gamma=cfg.routing.task_prototype_ema_gamma,
    )
    # Built once and shared between the router (for q(x)) and the routing
    # loss (for q̃). The shared Parameter objects are registered on the
    # learner via the router, so a single optimizer covers them.
    projection = build_routing_projection(cfg, backbone.embed_dim)
    router = build_router(cfg, memory, projection=projection)
    classifier = nn.Linear(backbone.embed_dim, cfg.model.num_classes)
    learner = ContinualLearner(backbone, lora_bank, router, classifier)

    # --- buffer, optimizer, losses ---
    buffer = build_buffer(cfg)
    log.info(
        "Replay buffer: %s (capacity=%d)",
        type(buffer).__name__,
        cfg.replay.per_expert_capacity,
    )
    trainable = [p for p in learner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    routing_loss = (
        RoutingKLLoss(
            memory=memory,
            temperature=cfg.routing.temperature,
            projection=projection,
        )
        if cfg.train.lambda_route > 0
        else None
    )
    if cfg.train.lambda_route > 0 and projection is None:
        log.warning(
            "lambda_route=%s but routing.projection is disabled — q(x) has no "
            "gradient path so the routing loss will not train anything.",
            cfg.train.lambda_route,
        )
    metrics = ContinualMetrics(num_tasks=cfg.data.num_tasks)
    trainer_cfg = TrainerConfig(
        epochs_per_task=cfg.train.epochs_per_task,
        train_batch_size=cfg.data.batch_size,
        eval_batch_size=cfg.data.get("eval_batch_size", cfg.data.batch_size * 2),
        replay_batch_size=cfg.replay.replay_batch_size,
        num_workers=cfg.data.num_workers,
        lambda_replay=cfg.train.lambda_replay,
        lambda_route=cfg.train.lambda_route,
        log_every=cfg.train.log_every,
    )

    # --- run ---
    with WandBLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[str(cfg.method), f"k{int(cfg.model.num_experts)}"],
        mode=cfg.wandb.mode,
    ) as logger:
        trainer = ContinualTrainer(
            learner=learner,
            memory=memory,
            buffer=buffer,
            optimizer=optimizer,
            metrics=metrics,
            config=trainer_cfg,
            device_info=device_info,
            routing_loss=routing_loss,
            log_callback=logger.log,
        )
        final_metrics = trainer.run(stream)

        out_dir = Path("outputs/results") / (cfg.wandb.name or "run")
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "accuracy_matrix.npy", final_metrics.matrix)

        # Task-expert similarity (uninitialized rows/cols are NaN-masked so
        # downstream plotting does not show "perfectly dissimilar" by mistake).
        task_p = memory.task_prototypes.detach().cpu().numpy()
        expert_p = memory.expert_prototypes.detach().cpu().numpy()
        task_init = memory.task_initialized.detach().cpu().numpy()
        expert_init = memory.expert_initialized.detach().cpu().numpy()

        def _norm(x):
            n = np.linalg.norm(x, axis=-1, keepdims=True)
            return x / np.clip(n, 1e-8, None)

        sim = _norm(task_p) @ _norm(expert_p).T
        valid = task_init[:, None] & expert_init[None, :]
        sim = np.where(valid, sim, np.nan)
        np.save(out_dir / "task_expert_similarity.npy", sim)

        # Final summary as JSON for easy table generation.
        snap = final_metrics.snapshot(cfg.data.num_tasks - 1)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "method": cfg.method,
                    "global_replay": cfg.replay.global_replay,
                    "seed": cfg.seed,
                    "avg_accuracy": snap.average_accuracy,
                    "avg_forgetting": snap.average_forgetting,
                    "bwt": snap.backward_transfer,
                    "per_task_accuracy": snap.per_task_accuracy,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
