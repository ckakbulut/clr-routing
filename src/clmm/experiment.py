import copy
from collections import defaultdict

import numpy as np

from clmm.config import Config
from clmm.model import CLLoRAViT, count_trainable
from clmm.replay import ReplayManager
from clmm.trainer import (
    train_one_task,
    evaluate_seen_tasks,
    compute_metrics,
    reinitialize_experts_from_classes,
)
from clmm.utils import set_seed


def method_config(base_cfg: Config, name: str):
    cfg = copy.deepcopy(base_cfg)
    if name == "full":
        cfg.routing_mode = "entropy_adaptive"
        cfg.replay_mode = "expert"
        cfg.use_routing_kl = True
    elif name == "static_top1":
        cfg.routing_mode = "top1"
        cfg.replay_mode = "expert"
        cfg.use_routing_kl = True
    elif name == "static_top2":
        cfg.routing_mode = "top2"
        cfg.replay_mode = "expert"
        cfg.use_routing_kl = True
    elif name == "static_top3":
        cfg.routing_mode = "top3"
        cfg.replay_mode = "expert"
        cfg.use_routing_kl = True
    elif name == "global_buffer":
        cfg.routing_mode = "entropy_adaptive"
        cfg.replay_mode = "global"
        cfg.use_routing_kl = True
    else:
        raise ValueError(name)
    return cfg


def run_experiment(method_name: str, base_cfg: Config, train_task_datasets, test_task_datasets, device):
    cfg = method_config(base_cfg, method_name)
    set_seed(cfg.seed)
    model = CLLoRAViT(cfg).to(device)
    total, trainable = count_trainable(model)
    print(f"\n=== Method: {method_name} ===")
    print(f"Total params: {total:,} | Trainable params: {trainable:,}")

    replay_manager = ReplayManager(cfg)
    n_tasks = len(train_task_datasets)
    acc_matrix = np.full((n_tasks, n_tasks), np.nan, dtype=np.float32)
    routing_soft_by_task = np.zeros((n_tasks, cfg.num_experts), dtype=np.float32)
    routing_hard_by_task = np.zeros((n_tasks, cfg.num_experts), dtype=np.float32)
    entropy_by_task = np.zeros(n_tasks, dtype=np.float32)

    logs = defaultdict(list)

    for task_id in range(n_tasks):
        print(f"Training task {task_id+1}/{n_tasks} ...")
        logs = train_one_task(model, replay_manager, train_task_datasets[task_id], task_id, cfg, device, logs)

        if task_id == 0:
            reinitialize_experts_from_classes(model, cfg)

        evals = evaluate_seen_tasks(model, test_task_datasets, task_id, cfg, device)
        for j, res in enumerate(evals):
            acc_matrix[task_id, j] = res["acc"]
            routing_soft_by_task[j] = res["soft_mean"]
            routing_hard_by_task[j] = res["hard_freq"]
            entropy_by_task[j] = res["entropy_mean"]
        avg_acc, forget, bwt, final_avg_acc, final_forgetting, final_bwt = compute_metrics(acc_matrix[:task_id+1, :task_id+1])
        print(
            f"After task {task_id}: "
            f"AA={avg_acc[-1]:.4f}, "
            f"Forgetting={forget[-1]:.4f}, "
            f"BWT={bwt[-1]:.4f}, "
            f"buffers={replay_manager.expert_sizes()}"
        )
    avg_acc, forgetting, bwt, final_avg_acc, final_forgetting, final_bwt = compute_metrics(acc_matrix)
    result = {
        "model": model,
        "method": method_name,
        "cfg": cfg,
        "acc_matrix": acc_matrix,

        "avg_acc": avg_acc,
        "forgetting": forgetting,
        "bwt": bwt,

        "final_avg_acc": final_avg_acc,
        "final_forgetting": final_forgetting,
        "final_bwt": final_bwt,

        "logs": dict(logs),

        "routing_soft_by_task": routing_soft_by_task,
        "routing_hard_by_task": routing_hard_by_task,
        "entropy_by_task": entropy_by_task,

        "buffer_class_counts": [b.class_counts() for b in replay_manager.expert_buffers],
    }
    return result
