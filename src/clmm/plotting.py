import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def _apply_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.4,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "#FCFCFC",
    })


def _style_axes(ax=None, grid_axis="y"):
    ax = ax or plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.tick_params(direction="out", length=4, width=1.0)

    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.25)
    else:
        ax.grid(False)


def _style_legend(ax=None):
    ax = ax or plt.gca()
    leg = ax.legend(
        frameon=True,
        fancybox=True,
        borderpad=0.5,
        handlelength=2.0,
    )
    leg.get_frame().set_facecolor("#F7F7F7")
    leg.get_frame().set_edgecolor("#C8C8C8")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.96)


def savefig(path):
    _apply_pub_style()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_average_accuracy(all_results, out_dir):
    _apply_pub_style()
    plt.figure(figsize=(5.4, 3.0))

    colors = ["#3B6FB6", "#5DAA68", "#8B6BB8", "#D8893C", "#6BAED6", "#A6A6A6"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (name, res) in enumerate(all_results.items()):
        x = np.arange(1, len(res["avg_acc"]) + 1)
        y = np.array(res["avg_acc"])
        plt.plot(
            x, y,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=name,
        )
        plt.fill_between(x, y, alpha=0.08, color=colors[i % len(colors)])

    plt.xlabel("Task Number")
    plt.ylabel("Average Accuracy")
    plt.title("Average Accuracy vs. Task", pad=10, weight="semibold")
    plt.xticks(x)

    _style_axes()
    _style_legend()
    savefig(os.path.join(out_dir, "average_accuracy_vs_task.png"))


def plot_forgetting(all_results, out_dir):
    _apply_pub_style()
    plt.figure(figsize=(5.4, 3.0))

    colors = ["#3B6FB6", "#5DAA68", "#8B6BB8", "#D8893C", "#6BAED6", "#A6A6A6"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (name, res) in enumerate(all_results.items()):
        x = np.arange(1, len(res["forgetting"]) + 1)
        y = np.array(res["forgetting"])
        plt.plot(
            x, y,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=name,
        )
        plt.fill_between(x, y, alpha=0.08, color=colors[i % len(colors)])

    plt.xlabel("Task Number")
    plt.ylabel("Average Forgetting")
    plt.title("Forgetting vs. Task", pad=10, weight="semibold")
    plt.xticks(x)

    _style_axes()
    _style_legend()
    savefig(os.path.join(out_dir, "forgetting_vs_task.png"))


def plot_training_loss(result, out_dir):
    _apply_pub_style()
    plt.figure(figsize=(5.4, 3.0))

    vals = result["logs"].get("loss", [])
    plt.plot(vals, color="#3B6FB6", linewidth=2.2)

    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title(f"Training Loss — {result['method']}", pad=10, weight="semibold")

    _style_axes()
    savefig(os.path.join(out_dir, f"loss_{result['method']}.png"))


def plot_entropy_hist(result, out_dir):
    _apply_pub_style()
    vals = result["logs"].get("entropy", [])

    plt.figure(figsize=(5.4, 3.0))
    plt.hist(vals, bins=30, color="#8B6BB8", edgecolor="white", linewidth=0.8, alpha=0.9)

    plt.xlabel("Routing Entropy")
    plt.ylabel("Count")
    plt.title(f"Entropy Histogram — {result['method']}", pad=10, weight="semibold")

    _style_axes()
    savefig(os.path.join(out_dir, f"entropy_hist_{result['method']}.png"))


def plot_expert_utilization(result, out_dir):
    _apply_pub_style()
    hard = np.array(result["logs"].get("expert_hard_counts", []))
    if hard.size == 0:
        return

    counts = hard.sum(axis=0)

    plt.figure(figsize=(5.0, 3.0))
    plt.bar(
        np.arange(len(counts)),
        counts,
        color="#5DAA68",
        edgecolor="#3E7F4C",
        linewidth=0.8,
        alpha=0.9,
    )

    plt.xlabel("Expert ID")
    plt.ylabel("Hard Assignment Count")
    plt.title(f"Expert Utilization — {result['method']}", pad=10, weight="semibold")

    _style_axes()
    savefig(os.path.join(out_dir, f"expert_utilization_{result['method']}.png"))


def plot_num_selected(result, out_dir):
    _apply_pub_style()
    vals = result["logs"].get("num_selected", [])

    plt.figure(figsize=(5.0, 3.0))
    plt.hist(
        vals,
        bins=np.arange(1, 5) - 0.5,
        rwidth=0.8,
        color="#D8893C",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    plt.xticks([1, 2, 3])
    plt.xlabel("Number of Selected Experts")
    plt.ylabel("Count")
    plt.title(f"Selected Experts Over Time — {result['method']}", pad=10, weight="semibold")

    _style_axes()
    savefig(os.path.join(out_dir, f"num_selected_{result['method']}.png"))


def plot_routing_heatmap(result, out_dir, use_soft=True):
    _apply_pub_style()
    mat = result["routing_soft_by_task"] if use_soft else result["routing_hard_by_task"]
    name = "soft_mean_q" if use_soft else "hard_frequency"

    plt.figure(figsize=(5.8, 3.4))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")

    cbar = plt.colorbar(im, label=name)
    cbar.outline.set_linewidth(0.8)

    plt.xlabel("Expert ID")
    plt.ylabel("Task ID")
    plt.title(f"Per-task Expert Routing Heatmap ({name}) — {result['method']}",
              pad=10, weight="semibold")

    plt.xticks(np.arange(mat.shape[1]))
    plt.yticks(np.arange(mat.shape[0]))

    _style_axes(grid_axis=None)
    savefig(os.path.join(out_dir, f"routing_heatmap_{name}_{result['method']}.png"))


def plot_per_task_bars(result, out_dir):
    _apply_pub_style()
    mat = result["routing_hard_by_task"]

    for t in range(mat.shape[0]):
        plt.figure(figsize=(4.8, 2.8))
        plt.bar(
            np.arange(mat.shape[1]),
            mat[t],
            color="#3B6FB6",
            edgecolor="#2D548A",
            linewidth=0.8,
            alpha=0.9,
        )

        plt.xlabel("Expert ID")
        plt.ylabel("Hard Selection Frequency")
        plt.title(f"Task {t} Expert Usage — {result['method']}", pad=10, weight="semibold")

        _style_axes()
        savefig(os.path.join(out_dir, f"task_{t}_expert_usage_{result['method']}.png"))


def plot_entropy_by_task(result, out_dir):
    _apply_pub_style()
    vals = result["entropy_by_task"]

    plt.figure(figsize=(5.0, 3.0))
    plt.plot(
        np.arange(len(vals)),
        vals,
        marker="o",
        color="#8B6BB8",
        linewidth=2.4,
    )

    plt.xlabel("Task ID")
    plt.ylabel("Average Routing Entropy")
    plt.title(f"Routing Entropy per Task — {result['method']}", pad=10, weight="semibold")

    _style_axes()
    savefig(os.path.join(out_dir, f"entropy_by_task_{result['method']}.png"))


def plot_buffer_class_distribution(result, out_dir):
    _apply_pub_style()
    counts_list = result["buffer_class_counts"]

    for k, counter in enumerate(counts_list):
        if not counter:
            continue

        labels = sorted(counter.keys())
        vals = [counter[c] for c in labels]

        plt.figure(figsize=(6.2, 2.8))
        plt.bar(
            [str(c) for c in labels],
            vals,
            color="#5DAA68",
            edgecolor="#3E7F4C",
            linewidth=0.7,
            alpha=0.9,
        )

        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.title(f"Buffer Class Distribution Expert {k} — {result['method']}",
                  pad=10, weight="semibold")
        plt.xticks(rotation=90)

        _style_axes()
        savefig(os.path.join(out_dir, f"buffer_class_dist_expert_{k}_{result['method']}.png"))


def plot_class_expert_heatmap(model, out_dir, method_name, class_names=None):
    """Visualize class prototype -> expert similarity."""
    _apply_pub_style()
    proto_mgr = model.prototype_manager

    if len(proto_mgr.class_prototypes) == 0:
        print("No class prototypes available.")
        return

    class_ids = sorted(proto_mgr.class_prototypes.keys())

    class_vecs = torch.stack([
        proto_mgr.class_prototypes[c]
        for c in class_ids
    ])

    expert_vecs = proto_mgr.expert_prototypes

    sims = torch.matmul(class_vecs, expert_vecs.T).cpu().numpy()

    plt.figure(figsize=(6.4, 5.0))
    im = plt.imshow(sims, aspect="auto", cmap="viridis")

    cbar = plt.colorbar(im, label="Cosine Similarity")
    cbar.outline.set_linewidth(0.8)

    plt.xlabel("Expert ID")

    if class_names is not None:
        labels = [class_names[c] for c in class_ids]
        plt.ylabel("Class")
        plt.yticks(np.arange(len(class_ids)), labels)
    else:
        plt.ylabel("Class ID")
        plt.yticks(np.arange(len(class_ids)), class_ids)

    plt.title(f"Class → Expert Similarity — {method_name}",
              pad=10, weight="semibold")

    plt.xticks(np.arange(sims.shape[1]))

    _style_axes(grid_axis=None)
    savefig(os.path.join(out_dir, f"class_expert_heatmap_{method_name}.png"))


def plot_class_expert_usage_with_labels(counts, class_names, out_dir, method_name):
    plt.figure(figsize=(10, 12))

    plt.imshow(counts.numpy(), aspect="auto")
    plt.colorbar(label="Average expert weight")

    plt.xlabel("Expert ID")
    plt.ylabel("Class")

    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xticks(np.arange(counts.shape[1]))

    plt.title(f"Class → Expert Usage — {method_name}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"class_expert_usage_labels_{method_name}.png"), dpi=200)
    plt.show()
