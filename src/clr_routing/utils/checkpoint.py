"""Checkpoint save/load. Stores model, optimizer, and arbitrary state dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Manages periodic checkpointing during continual learning.

    Each checkpoint is keyed by task index, so per-task ablations can resume
    from a specific point in the task sequence.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        task_id: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint after completing `task_id`.

        Returns the path written.
        """
        payload: dict[str, Any] = {
            "task_id": task_id,
            "model": model_state,
        }
        if optimizer_state is not None:
            payload["optimizer"] = optimizer_state
        if extras is not None:
            payload["extras"] = extras

        path = self._root / f"task_{task_id:03d}.pt"
        torch.save(payload, path)
        return path

    def load(self, task_id: int, map_location: str = "cpu") -> dict[str, Any]:
        """Load a checkpoint by task index."""
        path = self._root / f"task_{task_id:03d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    def latest(self) -> int | None:
        """Return the highest task_id with a checkpoint, or None."""
        ckpts = sorted(self._root.glob("task_*.pt"))
        if not ckpts:
            return None
        return int(ckpts[-1].stem.split("_")[-1])
