import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from clmm.config import Config


class TaskSubset(Dataset):
    def __init__(self, base_dataset, indices, class_map):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.class_map = class_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        y = int(y)
        y = self.class_map[y]
        return x, y


def build_transforms(input_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


def make_task_split(cfg: Config):
    classes = list(range(cfg.num_classes))
    if cfg.random_task_order:
        rng = random.Random(cfg.seed)
        rng.shuffle(classes)
    tasks = [classes[i:i+cfg.classes_per_task] for i in range(0, cfg.num_classes, cfg.classes_per_task)]
    tasks = tasks[:cfg.num_tasks]
    return tasks


def indices_for_classes(dataset, class_ids: List[int], max_samples: Optional[int] = None, seed: int = 0):
    class_set = set(class_ids)
    targets = np.array(dataset.targets)
    idx = np.where(np.isin(targets, list(class_set)))[0].tolist()
    if max_samples is not None and len(idx) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(idx)
        idx = idx[:max_samples]
    return idx


def prepare_cifar100(cfg: Config):
    train_tf, test_tf = build_transforms(cfg.input_size)
    train_base = torchvision.datasets.CIFAR100(root=cfg.out_dir, train=True, download=True, transform=train_tf)
    test_base = torchvision.datasets.CIFAR100(root=cfg.out_dir, train=False, download=True, transform=test_tf)
    tasks = make_task_split(cfg)

    n_tasks_run = cfg.debug_num_tasks if cfg.debug else cfg.num_tasks
    tasks_run = tasks[:n_tasks_run]
    selected_classes = sorted({c for task in tasks_run for c in task})

    class_map = {old: new for new, old in enumerate(selected_classes)}

    train_task_datasets, test_task_datasets = [], []
    for tid, cls in enumerate(tasks_run):
        train_cap = cfg.debug_train_samples_per_task if cfg.debug else None
        test_cap = cfg.debug_test_samples_per_task if cfg.debug else None
        tr_idx = indices_for_classes(train_base, cls, max_samples=train_cap, seed=cfg.seed + tid)
        te_idx = indices_for_classes(test_base, cls, max_samples=test_cap, seed=cfg.seed + 1000 + tid)
        train_task_datasets.append(TaskSubset(train_base, tr_idx, class_map))
        test_task_datasets.append(TaskSubset(test_base, te_idx, class_map))

    print("Task class split:")
    for i, cls in enumerate(tasks_run):
        print(f"Task {i}: {cls}")
    return tasks_run, train_task_datasets, test_task_datasets
