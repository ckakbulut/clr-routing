"""Continual learning machinery: buffers, losses, metrics, trainer."""

from clr_routing.continual.buffer import (
    GlobalReplayBuffer,
    PerExpertReplayBuffer,
    ReplayBuffer,
    ReservoirBuffer,
)
from clr_routing.continual.losses import RoutingKLLoss
from clr_routing.continual.metrics import ContinualMetrics
from clr_routing.continual.trainer import ContinualTrainer

__all__ = [
    "ContinualMetrics",
    "ContinualTrainer",
    "GlobalReplayBuffer",
    "PerExpertReplayBuffer",
    "ReplayBuffer",
    "ReservoirBuffer",
    "RoutingKLLoss",
]
