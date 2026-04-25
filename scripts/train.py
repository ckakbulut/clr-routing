"""Main training entrypoint, configured via Hydra.

Usage:
    python scripts/train.py                              # default config
    python scripts/train.py experiment=baseline_top1     # baseline
    python scripts/train.py model.num_experts=8 train.lambda_route=0.5  # overrides
"""

from __future__ import annotations

import hydra
import torch
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
    ViTBackbone,
)
from clr_routing.utils import WandBLogger, select_device, set_seed


def build_router(cfg: DictConfig, memory: PrototypeMemory):
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
    set_seed(cfg.seed)
    device_info = select_device(cfg.device)
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
    router = build_router(cfg, memory)
    classifier = nn.Linear(backbone.embed_dim, cfg.model.num_classes)
    learner = ContinualLearner(backbone, lora_bank, router, classifier)

    # --- buffer, optimizer, losses ---
    buffer = build_buffer(cfg)
    trainable = [p for p in learner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    routing_loss = (
        RoutingKLLoss(memory=memory, temperature=cfg.routing.temperature)
        if cfg.train.lambda_route > 0
        else None
    )
    metrics = ContinualMetrics(num_tasks=cfg.data.num_tasks)
    trainer_cfg = TrainerConfig(
        epochs_per_task=cfg.train.epochs_per_task,
        train_batch_size=cfg.data.batch_size,
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
        tags=[cfg.method, f"k{cfg.model.num_experts}"],
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
        trainer.run(stream)


if __name__ == "__main__":
    main()
