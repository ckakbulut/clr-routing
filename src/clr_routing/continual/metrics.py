"""Continual learning metrics: accuracy matrix, average accuracy, forgetting, BWT (Backward Transfer).

Standard definitions following Lopez-Paz & Ranzato (GEM, 2017):

    R[i, j] = test accuracy on task j after training through task i.

    Average accuracy after task T:    A_T = (1/T) * sum_{j<=T} R[T, j]
    Average forgetting after task T:  F_T = (1/(T-1)) * sum_{j<T} max_i<T R[i,j] - R[T,j]
    Backward transfer (BWT):          BWT_T = (1/(T-1)) * sum_{j<T} R[T, j] - R[j, j]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContinualSnapshot:
    """Summary metrics at a single point in the task sequence."""

    task_id: int
    average_accuracy: float
    average_forgetting: float
    backward_transfer: float
    per_task_accuracy: dict[int, float]


class ContinualMetrics:
    """Maintains R[i, j] and computes derived metrics on demand."""

    def __init__(self, num_tasks: int) -> None:
        self._num_tasks = num_tasks
        # NaN sentinel for "not yet evaluated".
        self._matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)

    def record(self, after_task: int, eval_task: int, accuracy: float) -> None:
        """Record R[after_task, eval_task] = accuracy."""
        self._matrix[after_task, eval_task] = accuracy

    def snapshot(self, after_task: int) -> ContinualSnapshot:
        """Compute summary metrics at the row `after_task`.

        Only tasks 0..after_task are considered seen.
        """
        seen = after_task + 1
        row = self._matrix[after_task, :seen]
        avg_acc = float(np.nanmean(row))

        if after_task == 0:
            return ContinualSnapshot(
                task_id=after_task,
                average_accuracy=avg_acc,
                average_forgetting=0.0,
                backward_transfer=0.0,
                per_task_accuracy={0: float(row[0])},
            )

        # Forgetting: for each prior task j<after_task, compare best historical
        # accuracy on j against current accuracy R[after_task, j].
        forgetting_terms = []
        bwt_terms = []
        for j in range(after_task):
            history = self._matrix[:after_task, j]  # rows 0..after_task-1
            best = float(np.nanmax(history))
            current = float(row[j])
            forgetting_terms.append(best - current)
            bwt_terms.append(current - float(self._matrix[j, j]))

        avg_forget = float(np.mean(forgetting_terms))
        bwt = float(np.mean(bwt_terms))

        per_task = {j: float(row[j]) for j in range(seen)}
        return ContinualSnapshot(
            task_id=after_task,
            average_accuracy=avg_acc,
            average_forgetting=avg_forget,
            backward_transfer=bwt,
            per_task_accuracy=per_task,
        )

    @property
    def matrix(self) -> np.ndarray:
        """Return a copy of the accuracy matrix R."""
        return self._matrix.copy()
