"""W&B logging wrapped as a context manager (RAII pattern).

The logger acquires a W&B run on entry and finalizes it on exit, even if an
exception is raised. Use as:

    with WandBLogger(project="clr-routing", config=cfg) as logger:
        logger.log({"loss": 0.5}, step=0)
"""

from __future__ import annotations

from typing import Any

import wandb


class WandBLogger:
    """Context-managed W&B run.

    Attributes:
        project: W&B project name.
        name: Optional run name.
        config: Configuration dict to log alongside the run.
        tags: Optional list of tags.
        entity: W&B team/user (None uses default).
        mode: 'online', 'offline', or 'disabled' for unit tests.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        entity: str | None = None,
        mode: str = "online",
    ) -> None:
        self._project = project
        self._name = name
        self._config = config or {}
        self._tags = tags or []
        self._entity = entity
        self._mode = mode
        self._run: Any = None

    def __enter__(self) -> WandBLogger:
        self._run = wandb.init(
            project=self._project,
            name=self._name,
            config=self._config,
            tags=self._tags,
            entity=self._entity,
            mode=self._mode,
            reinit=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._run is not None:
            wandb.finish(exit_code=1 if exc_type is not None else 0)
            self._run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log scalar metrics, histograms, or W&B media objects."""
        if self._run is None:
            return
        wandb.log(metrics, step=step)

    def log_table(self, key: str, columns: list[str], data: list[list[Any]]) -> None:
        """Log a table (e.g. expert utilization per task)."""
        if self._run is None:
            return
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run is not None else None
