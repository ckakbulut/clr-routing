"""Split CIFAR-100 task stream for continual learning.

Partitions CIFAR-100 into `num_tasks` tasks of `classes_per_task` classes each.
Class assignment is determined by a permutation seed for reproducibility.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of a single continual-learning task."""

    task_id: int
    class_ids: tuple[int, ...]  # original CIFAR-100 class indices

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)


class TaskDataset(Dataset):
    """A single task's portion of CIFAR-100.

    Wraps a `Subset` of the underlying CIFAR-100 dataset and exposes a
    `TaskSpec` describing which classes belong to this task.

    Class labels are NOT remapped — the global label space (0..99) is preserved
    so a single classifier head can be used in the task-agnostic setting.
    """

    def __init__(self, base: CIFAR100, indices: list[int], spec: TaskSpec) -> None:
        self._subset = Subset(base, indices)
        self._spec = spec

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._subset[idx]

    @property
    def spec(self) -> TaskSpec:
        return self._spec


class SplitCIFAR100:
    """Builds a sequence of `TaskDataset`s from CIFAR-100.

    Example:
        stream = SplitCIFAR100(num_tasks=10, classes_per_task=10, seed=42)
        for task_train, task_test in stream:
            ...
    """

    # Standard CIFAR-100 normalization stats (computed over the train set).
    _MEAN = (0.5071, 0.4867, 0.4408)
    _STD = (0.2675, 0.2565, 0.2761)

    def __init__(
        self,
        num_tasks: int = 10,
        classes_per_task: int = 10,
        seed: int = 42,
        root: str = "./data",
        image_size: int = 224,
        download: bool = True,
    ) -> None:
        if num_tasks * classes_per_task != 100:
            raise ValueError(
                f"num_tasks * classes_per_task must equal 100, "
                f"got {num_tasks} * {classes_per_task}"
            )
        self._num_tasks = num_tasks
        self._classes_per_task = classes_per_task

        train_tf = self._build_transform(image_size, train=True)
        test_tf = self._build_transform(image_size, train=False)

        self._train = CIFAR100(root=root, train=True, download=download, transform=train_tf)
        self._test = CIFAR100(root=root, train=False, download=download, transform=test_tf)

        self._task_specs = self._build_task_specs(seed)
        self._train_indices = self._index_by_task(self._train, self._task_specs)
        self._test_indices = self._index_by_task(self._test, self._task_specs)

    def _build_transform(self, size: int, train: bool) -> transforms.Compose:
        ops = [transforms.Resize((size, size))]
        if train:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size, padding=size // 28),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )
        return transforms.Compose(ops)

    def _build_task_specs(self, seed: int) -> list[TaskSpec]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(100)
        specs = []
        for t in range(self._num_tasks):
            start = t * self._classes_per_task
            end = start + self._classes_per_task
            specs.append(TaskSpec(task_id=t, class_ids=tuple(int(c) for c in order[start:end])))
        return specs

    def _index_by_task(
        self, dataset: CIFAR100, specs: list[TaskSpec]
    ) -> dict[int, list[int]]:
        targets = np.asarray(dataset.targets)
        indices: dict[int, list[int]] = {}
        for spec in specs:
            mask = np.isin(targets, spec.class_ids)
            indices[spec.task_id] = np.where(mask)[0].tolist()
        return indices

    def __len__(self) -> int:
        return self._num_tasks

    def __iter__(self) -> Iterator[tuple[TaskDataset, TaskDataset]]:
        for spec in self._task_specs:
            yield self.get_task(spec.task_id)

    def get_task(self, task_id: int) -> tuple[TaskDataset, TaskDataset]:
        """Return (train_dataset, test_dataset) for the given task."""
        spec = self._task_specs[task_id]
        train_ds = TaskDataset(self._train, self._train_indices[task_id], spec)
        test_ds = TaskDataset(self._test, self._test_indices[task_id], spec)
        return train_ds, test_ds

    @property
    def task_specs(self) -> list[TaskSpec]:
        return list(self._task_specs)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    """Standard DataLoader factory.

    `pin_memory` defaults to False because pinning is only meaningful for CUDA
    and triggers a warning on MPS/CPU. Callers should pass `device_info.pin_memory`.
    Workers (when used) are re-seeded via `seed_worker` for reproducibility.
    """
    from clr_routing.utils.seed import seed_worker

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["worker_init_fn"] = seed_worker
    return DataLoader(**kwargs)
