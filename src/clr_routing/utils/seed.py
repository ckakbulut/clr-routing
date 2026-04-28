"""Deterministic seeding for reproducibility across torch, numpy, and Python."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds and configure deterministic algorithms.

    Args:
        seed: The seed value. Use the same seed across runs for reproducibility.
        deterministic: If True, configures cuDNN and torch for deterministic
            operations. May reduce throughput; disable for benchmarking only.

    Notes:
        - `torch.manual_seed` already covers CPU and (in recent versions) MPS,
          but we call backend-specific seeders explicitly for safety across
          PyTorch versions.
        - cuDNN flags are CUDA-specific but safe to set unconditionally.
        - PYTHONHASHSEED can only be set before interpreter startup; we do not
          touch it at runtime because that has no effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        # torch.mps.manual_seed exists from PyTorch 2.0+.
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Only relevant on CUDA; harmless on other backends.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for DataLoader: re-seeds Python's `random` and NumPy in
    each worker from the per-worker torch seed, so augmentation and any user
    code in a worker is deterministic given a fixed `set_seed` in the main
    process.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
