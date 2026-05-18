from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clmm.config import Config
from clmm.model import CLLoRAViT
from clmm.replay import ReplayManager
from clmm.routing import select_experts, routing_kl_loss, load_balance_loss


def make_optimizer(model: CLLoRAViT, cfg: Config):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)


def get_seen_classes(task_id: int, tasks_run):
    return sorted({
        c
        for task_classes in tasks_run[:task_id + 1]
        for c in task_classes
    })


def train_one_task(model, replay_manager, train_dataset, task_id: int, cfg: Config, device, logs: Dict):
    model.train()
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    optimizer = make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    for epoch in range(cfg.epochs_per_task):
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                r, z, q = model.route_from_x(x)
                selected, weights, ent = select_experts(q, cfg.routing_mode, cfg)

                logits, _ = model.forward_with_routing(x, selected, weights)
                loss_task = F.cross_entropy(logits, y)

                loss_replay = torch.tensor(0.0, device=device)

                if task_id >= cfg.warmup_tasks and cfg.replay_mode != "none":
                    replay = replay_manager.sample_for_selected(
                        selected,
                        cfg.replay_batch_size,
                        cfg.replay_mode
                    )

                    if replay is not None:
                        xr, yr = replay
                        xr = xr.to(device, non_blocking=True)
                        yr = yr.to(device, non_blocking=True)

                        rr, zr, qr = model.route_from_x(xr)
                        sr, wr, _ = select_experts(qr, cfg.routing_mode, cfg)

                        replay_logits, _ = model.forward_with_routing(xr, sr, wr)
                        loss_replay = F.cross_entropy(replay_logits, yr)

                loss_route = torch.tensor(0.0, device=device)

                use_route_loss_now = cfg.use_routing_kl and (task_id >= cfg.warmup_tasks)

                if use_route_loss_now:
                    q_tilde, valid_mask = model.prototype_manager.routing_target_from_labels(
                        y,
                        cfg.tau
                    )

                    if q_tilde is not None:
                        loss_route = routing_kl_loss(
                            q[valid_mask],
                            q_tilde[valid_mask]
                        )

                loss_balance = (
                    load_balance_loss(q, cfg)
                    if cfg.use_load_balance
                    else torch.tensor(0.0, device=device)
                )

                loss = (
                    loss_task
                    + cfg.lambda_replay * loss_replay
                    + cfg.lambda_route * loss_route
                    + cfg.lambda_balance * loss_balance
                )

            scaler.scale(loss).backward()

            if cfg.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                model.prototype_manager.update_class(y.detach(), z.detach())
                top1_selected = selected[:, :1]
                model.prototype_manager.update_experts(z.detach(), top1_selected.detach())

                replay_manager.add_batch(
                    x.detach().cpu(),
                    y.detach().cpu(),
                    selected.detach().cpu()
                )

            logs["loss"].append(float(loss.detach().cpu()))
            logs["loss_task"].append(float(loss_task.detach().cpu()))
            logs["loss_replay"].append(float(loss_replay.detach().cpu()))
            logs["loss_route"].append(float(loss_route.detach().cpu()))
            logs["loss_balance"].append(float(loss_balance.detach().cpu()))
            logs["entropy"].extend(ent.detach().cpu().tolist())
            logs["num_selected"].extend((weights > 0).sum(dim=1).detach().cpu().tolist())
            logs["expert_soft_sum"].append(q.detach().mean(dim=0).cpu().numpy())

            hard = torch.zeros(cfg.num_experts)
            top = torch.argmax(q.detach().cpu(), dim=-1)

            for k in top.tolist():
                hard[k] += 1

            logs["expert_hard_counts"].append(hard.numpy())

    return logs


@torch.no_grad()
def evaluate_task(model, dataset, cfg: Config, device):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    correct, total = 0, 0
    entropy_vals = []
    hard_counts = torch.zeros(cfg.num_experts)
    soft_sum = torch.zeros(cfg.num_experts)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        r, z, q = model.route_from_x(x)
        selected, weights, ent = select_experts(q, cfg.routing_mode, cfg)

        logits, _ = model.forward_with_routing(x, selected, weights)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        entropy_vals.extend(ent.cpu().tolist())
        soft_sum += q.cpu().sum(dim=0)

        top = q.cpu().argmax(dim=1)

        for k in top.tolist():
            hard_counts[k] += 1

    acc = correct / max(total, 1)

    return {
        "acc": acc,
        "entropy_mean": float(np.mean(entropy_vals)) if entropy_vals else 0.0,
        "hard_freq": (hard_counts / max(total, 1)).numpy(),
        "soft_mean": (soft_sum / max(total, 1)).numpy(),
    }


def evaluate_seen_tasks(model, test_task_datasets, seen_until: int, cfg: Config, device):
    results = []

    for tid in range(seen_until + 1):
        results.append(
            evaluate_task(
                model,
                test_task_datasets[tid],
                cfg,
                device
            )
        )

    return results


def compute_metrics(acc_matrix: np.ndarray):
    """
    acc_matrix[k, j] = accuracy on task j after training task k
    """
    T = acc_matrix.shape[0]

    avg_acc = []
    forgetting = []
    backward_transfer = []

    for k in range(T):
        seen = acc_matrix[k, :k + 1]
        avg_acc.append(np.nanmean(seen))

        if k == 0:
            forgetting.append(0.0)
            backward_transfer.append(0.0)
        else:
            f_vals = []
            bwt_vals = []

            for j in range(k):
                prev = acc_matrix[:k, j]

                if np.all(np.isnan(prev)):
                    continue

                best_prev = np.nanmax(prev)
                current = acc_matrix[k, j]

                f = best_prev - current
                bwt = current - best_prev

                f_vals.append(max(0.0, f))
                bwt_vals.append(bwt)

            forgetting.append(float(np.mean(f_vals)) if f_vals else 0.0)
            backward_transfer.append(float(np.mean(bwt_vals)) if bwt_vals else 0.0)

    final_avg_acc = avg_acc[-1]
    final_forgetting = forgetting[-1]
    final_bwt = backward_transfer[-1]

    return (
        np.array(avg_acc),
        np.array(forgetting),
        np.array(backward_transfer),
        final_avg_acc,
        final_forgetting,
        final_bwt
    )


@torch.no_grad()
def reinitialize_experts_from_classes(model, cfg):
    proto_mgr = model.prototype_manager

    class_ids = sorted(proto_mgr.class_prototypes.keys())

    if len(class_ids) < cfg.num_experts:
        print("Not enough class prototypes to reinitialize experts.")
        return

    C = torch.stack([
        proto_mgr.class_prototypes[c].to(proto_mgr.expert_prototypes.device)
        for c in class_ids
    ])
    C = F.normalize(C, dim=-1)

    centroids = [C[0]]

    for _ in range(1, cfg.num_experts):
        sims = torch.stack([
            torch.matmul(C, c)
            for c in centroids
        ], dim=1)

        closest_sim = sims.max(dim=1).values
        next_idx = torch.argmin(closest_sim)

        centroids.append(C[next_idx])

    centroids = torch.stack(centroids)

    proto_mgr.expert_prototypes = F.normalize(centroids, dim=-1)
    proto_mgr.expert_initialized[:] = True

    proto_mgr.class_to_expert = {}

    for cls_id, c in zip(class_ids, C):
        sims = torch.matmul(proto_mgr.expert_prototypes, c)
        expert_id = int(torch.argmax(sims).item())
        proto_mgr.class_to_expert[cls_id] = expert_id

    print("Reinitialized expert prototypes from class prototype clustering.")
    print("Class → Expert:", proto_mgr.class_to_expert)
