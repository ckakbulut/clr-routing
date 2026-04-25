"""Device selection.

Picks the best available accelerator in order: CUDA > MPS (Apple Silicon) > CPU.

Returns a `DeviceInfo` value object that bundles the `torch.device` with
device-dependent settings — primarily `pin_memory`, which only works for CUDA
and emits warnings on MPS.

Usage:
    info = select_device("auto")  # or "cuda", "mps", "cpu"
    x = x.to(info.device, non_blocking=info.supports_non_blocking)
    loader = DataLoader(..., pin_memory=info.pin_memory)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable bundle of device-specific settings."""

    device: torch.device
    backend: str  # 'cuda', 'mps', or 'cpu'

    @property
    def pin_memory(self) -> bool:
        """pin_memory is only meaningful (and warning-free) for CUDA."""
        return self.backend == "cuda"

    @property
    def supports_non_blocking(self) -> bool:
        """Async host-to-device copies are only useful for CUDA + pinned memory."""
        return self.backend == "cuda"

    def __str__(self) -> str:
        return f"DeviceInfo(backend={self.backend}, device={self.device})"


def select_device(preference: str = "auto") -> DeviceInfo:
    """Select an accelerator.

    Args:
        preference: 'auto' picks the best available backend.
            Otherwise one of 'cuda', 'mps', 'cpu' to force a specific backend.

    Raises:
        RuntimeError: if a specific backend is requested but unavailable.
        ValueError: if `preference` is not a recognized value.
    """
    pref = preference.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            backend = "cuda"
        elif _mps_available():
            backend = "mps"
        else:
            backend = "cpu"
    elif pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        backend = "cuda"
    elif pref == "mps":
        if not _mps_available():
            raise RuntimeError(
                "MPS requested but not available. Requires Apple Silicon and "
                "PyTorch built with MPS support."
            )
        backend = "mps"
    elif pref == "cpu":
        backend = "cpu"
    else:
        raise ValueError(f"Unknown device preference: {preference!r}")

    if backend == "mps":
        # Some torch ops still lack native MPS kernels. Enabling fallback lets
        # them run on CPU transparently instead of raising NotImplementedError.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return DeviceInfo(device=torch.device(backend), backend=backend)


def _mps_available() -> bool:
    """True if running on Apple Silicon with an MPS-capable PyTorch build."""
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    return bool(mps.is_available()) and bool(mps.is_built())
