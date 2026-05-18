import math
import random
from collections import Counter
from typing import List, Optional, Tuple

import torch

from clmm.config import Config


class ClassBalancedBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: List[Tuple[torch.Tensor, int]] = []

    def __len__(self):
        return len(self.data)

    def add(self, x_cpu: torch.Tensor, y: int):
        x_cpu = x_cpu.detach().cpu()
        y = int(y)
        if len(self.data) < self.capacity:
            self.data.append((x_cpu, y))
            return
        labels = [yy for _, yy in self.data]
        counts = Counter(labels)
        majority_class = max(counts, key=counts.get)
        candidates = [i for i, (_, yy) in enumerate(self.data) if yy == majority_class]
        replace_idx = random.choice(candidates)
        self.data[replace_idx] = (x_cpu, y)

    def sample(self, n: int):
        if len(self.data) == 0 or n <= 0:
            return None
        batch = random.sample(self.data, min(n, len(self.data)))
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def class_counts(self):
        return Counter([y for _, y in self.data])


class ReplayManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.expert_buffers = [ClassBalancedBuffer(cfg.buffer_size_per_expert) for _ in range(cfg.num_experts)]
        self.global_buffer = ClassBalancedBuffer(cfg.global_buffer_size)

    @torch.no_grad()
    def add_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        selected_experts: torch.Tensor
    ):
        """
        Add each sample to the global buffer and only to its top-1 selected expert buffer.

        x: [B, C, H, W]
        y: [B]
        selected_experts: [B, S]
        """
        selected_experts = selected_experts.detach().cpu()

        for i in range(x.shape[0]):
            yi = int(y[i].detach().cpu().item())
            xi = x[i].detach().cpu()

            self.global_buffer.add(xi, yi)

            top_expert = int(selected_experts[i, 0].item())
            self.expert_buffers[top_expert].add(xi, yi)

    def sample_for_selected(self, selected_experts: torch.Tensor, total_n: int, mode: str):
        if mode == "none" or total_n <= 0:
            return None
        if mode == "global":
            return self.global_buffer.sample(total_n)

        unique_experts = torch.unique(selected_experts.detach().cpu()).tolist()
        unique_experts = [int(k) for k in unique_experts]
        if len(unique_experts) == 0:
            return None
        per = max(1, math.ceil(total_n / len(unique_experts)))
        xs, ys = [], []
        for k in unique_experts:
            sample = self.expert_buffers[k].sample(per)
            if sample is not None:
                xb, yb = sample
                xs.append(xb)
                ys.append(yb)
        if not xs:
            return None
        xr = torch.cat(xs, dim=0)[:total_n]
        yr = torch.cat(ys, dim=0)[:total_n]
        return xr, yr

    def expert_sizes(self):
        return [len(b) for b in self.expert_buffers]
