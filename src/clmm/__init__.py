"""CLMM: Uncertainty-Aware Prototype-Guided Routing with LoRA Experts for Task-Agnostic Continual Learning."""

from clmm.config import Config
from clmm.data import TaskSubset, build_transforms, make_task_split, indices_for_classes, prepare_cifar100
from clmm.lora import ExpertLoRALinear, replace_vit_mlp_with_lora, iter_lora_layers
from clmm.model import RouterMLP, PrototypeManager, CLLoRAViT, count_trainable
from clmm.routing import routing_entropy, normalize_entropy, select_experts, load_balance_loss, routing_kl_loss
from clmm.replay import ClassBalancedBuffer, ReplayManager
from clmm.trainer import (
    make_optimizer,
    get_seen_classes,
    train_one_task,
    evaluate_task,
    evaluate_seen_tasks,
    compute_metrics,
    reinitialize_experts_from_classes,
)
from clmm.experiment import method_config, run_experiment
from clmm.utils import set_seed, get_device

__all__ = [
    "Config",
    "TaskSubset", "build_transforms", "make_task_split", "indices_for_classes", "prepare_cifar100",
    "ExpertLoRALinear", "replace_vit_mlp_with_lora", "iter_lora_layers",
    "RouterMLP", "PrototypeManager", "CLLoRAViT", "count_trainable",
    "routing_entropy", "normalize_entropy", "select_experts", "load_balance_loss", "routing_kl_loss",
    "ClassBalancedBuffer", "ReplayManager",
    "make_optimizer", "get_seen_classes", "train_one_task", "evaluate_task",
    "evaluate_seen_tasks", "compute_metrics", "reinitialize_experts_from_classes",
    "method_config", "run_experiment",
    "set_seed", "get_device",
]
