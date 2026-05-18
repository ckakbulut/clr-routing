"""Run CLMM experiments from the command line.

Replicates the full notebook execution flow: data setup, one or more method
runs, figure generation, and a final printed summary.

Usage:
    python scripts/run_clmm.py
    python scripts/run_clmm.py --methods full static_top1 static_top2 static_top3
    python scripts/run_clmm.py --debug
    python scripts/run_clmm.py --out_dir ./outputs/clmm --data_root ./data
"""

import argparse
import os

import numpy as np
import torch

from clmm.config import Config
from clmm.data import prepare_cifar100
from clmm.experiment import run_experiment
from clmm.plotting import (
    plot_average_accuracy,
    plot_forgetting,
    plot_training_loss,
    plot_entropy_hist,
    plot_expert_utilization,
    plot_num_selected,
    plot_routing_heatmap,
    plot_per_task_bars,
    plot_entropy_by_task,
    plot_buffer_class_distribution,
    plot_class_expert_heatmap,
    plot_class_expert_usage_with_labels,
)
from clmm.routing import select_experts
from clmm.utils import set_seed, get_device
from torch.utils.data import DataLoader


def compute_class_expert_usage(model, dataset, cfg, device):
    """Returns class-expert usage matrix [num_classes, num_experts]."""
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    num_classes = cfg.num_classes
    num_experts = cfg.num_experts

    counts = torch.zeros(num_classes, num_experts)
    totals = torch.zeros(num_classes)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            _, _, q = model.route_from_x(x)
            selected, weights, _ = select_experts(q, cfg.routing_mode, cfg)

            selected = selected.cpu()
            weights = weights.cpu()
            y = y.cpu()

            B, S = selected.shape

            for i in range(B):
                c = int(y[i].item())
                totals[c] += 1

                for s in range(S):
                    e = int(selected[i, s].item())
                    w = float(weights[i, s].item())
                    counts[c, e] += w

    for c in range(num_classes):
        if totals[c] > 0:
            counts[c] /= totals[c]

    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLMM experiments")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full"],
        choices=["full", "static_top1", "static_top2", "static_top3", "global_buffer"],
        help="Methods to run",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for CIFAR-100 data")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer tasks/samples)")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    if args.out_dir is not None:
        config.out_dir = args.out_dir
    if args.data_root is not None:
        config.out_dir = args.data_root
    if args.debug:
        config.debug = True
    if args.seed is not None:
        config.seed = args.seed

    os.makedirs(config.out_dir, exist_ok=True)

    set_seed(config.seed)
    device = get_device()
    print("Using device:", device)

    tasks_run, train_task_datasets, test_task_datasets = prepare_cifar100(config)

    all_results = {}
    for method in args.methods:
        all_results[method] = run_experiment(
            method, config, train_task_datasets, test_task_datasets, device
        )

    print("\nDone. Generating figures...")

    plot_average_accuracy(all_results, config.out_dir)
    plot_forgetting(all_results, config.out_dir)

    for name, res in all_results.items():
        plot_training_loss(res, config.out_dir)
        plot_entropy_hist(res, config.out_dir)
        plot_expert_utilization(res, config.out_dir)
        plot_num_selected(res, config.out_dir)
        plot_routing_heatmap(res, config.out_dir, use_soft=True)
        plot_routing_heatmap(res, config.out_dir, use_soft=False)
        plot_per_task_bars(res, config.out_dir)
        plot_entropy_by_task(res, config.out_dir)
        plot_buffer_class_distribution(res, config.out_dir)
        plot_class_expert_heatmap(res["model"], config.out_dir, name)

    print("Plots saved to:", config.out_dir)

    print("\nFINAL SUMMARY")
    print("=" * 70)

    for name, res in all_results.items():
        print(f"Method: {name}")
        print("Accuracy matrix A[k, j]:")
        print(np.round(res["acc_matrix"], 4))
        print("Average accuracy by task:", np.round(res["avg_acc"], 4))
        print("Forgetting by task:", np.round(res["forgetting"], 4))
        print("Backward transfer by task:", np.round(res["bwt"], 4))
        print(f"Final average accuracy: {res['final_avg_acc']:.4f}")
        print(f"Final average forgetting: {res['final_forgetting']:.4f}")
        print(f"Final backward transfer: {res['final_bwt']:.4f}")
        print("-" * 70)

    for name, res in all_results.items():
        model = res["model"]
        cfg = res["cfg"]

        full_test = torch.utils.data.ConcatDataset(test_task_datasets)
        counts = compute_class_expert_usage(model, full_test, cfg, device)

        all_classes = train_task_datasets[0].base_dataset.classes
        selected_classes = sorted({c for task in tasks_run for c in task})
        class_names_subset = [all_classes[c] for c in selected_classes]

        plot_class_expert_usage_with_labels(
            counts[selected_classes],
            class_names_subset,
            config.out_dir,
            name
        )

    if "full" in all_results:
        full_avg_k = np.mean(all_results["full"]["logs"]["num_selected"])
        top3_avg_k = 3.0
        compute_ratio = full_avg_k / top3_avg_k
        compute_saved = 1.0 - compute_ratio
        print("Full avg selected experts:", full_avg_k)
        print("Static top3 avg selected experts:", top3_avg_k)
        print("Relative expert compute:", compute_ratio)
        print("Compute saved vs top3:", compute_saved)

    for name, res in all_results.items():
        avg_k = np.mean(res["logs"]["num_selected"])
        compute_ratio = avg_k / 3.0
        saved_vs_top3 = 1.0 - compute_ratio
        print(
            f"{name:12s} | avg experts/sample={avg_k:.3f} | "
            f"relative compute={compute_ratio:.3f} | "
            f"saved vs top3={100*saved_vs_top3:.1f}%"
        )


if __name__ == "__main__":
    main()
