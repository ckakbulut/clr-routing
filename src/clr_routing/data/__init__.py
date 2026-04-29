"""Data loaders for continual learning task streams."""

from clr_routing.data.cifar100_split import SplitCIFAR100, TaskDataset

__all__ = ["SplitCIFAR100", "TaskDataset"]
