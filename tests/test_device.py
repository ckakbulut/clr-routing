"""Tests for device selection.

These tests don't require a real GPU/MPS — they verify the dispatch logic
and the `DeviceInfo` value object's properties.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from clr_routing.utils.device import DeviceInfo, select_device


def test_device_info_pin_memory_only_for_cuda():
    cuda_info = DeviceInfo(device=torch.device("cuda"), backend="cuda")
    mps_info = DeviceInfo(device=torch.device("cpu"), backend="mps")
    cpu_info = DeviceInfo(device=torch.device("cpu"), backend="cpu")
    assert cuda_info.pin_memory is True
    assert mps_info.pin_memory is False
    assert cpu_info.pin_memory is False


def test_device_info_non_blocking_only_for_cuda():
    assert DeviceInfo(torch.device("cuda"), "cuda").supports_non_blocking is True
    assert DeviceInfo(torch.device("cpu"), "mps").supports_non_blocking is False
    assert DeviceInfo(torch.device("cpu"), "cpu").supports_non_blocking is False


def test_select_device_cpu_always_works():
    info = select_device("cpu")
    assert info.backend == "cpu"
    assert info.device.type == "cpu"


def test_select_device_rejects_unknown():
    with pytest.raises(ValueError):
        select_device("tpu")


def test_select_device_auto_falls_through_to_cpu():
    """When neither CUDA nor MPS is available, auto picks cpu."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("clr_routing.utils.device._mps_available", return_value=False),
    ):
        info = select_device("auto")
        assert info.backend == "cpu"


def test_select_device_auto_prefers_cuda():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("clr_routing.utils.device._mps_available", return_value=True),
    ):
        info = select_device("auto")
        assert info.backend == "cuda"


def test_select_device_auto_falls_to_mps_without_cuda():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("clr_routing.utils.device._mps_available", return_value=True),
    ):
        info = select_device("auto")
        assert info.backend == "mps"


def test_select_device_explicit_cuda_unavailable_raises():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA"):
            select_device("cuda")


def test_select_device_explicit_mps_unavailable_raises():
    with patch("clr_routing.utils.device._mps_available", return_value=False):
        with pytest.raises(RuntimeError, match="MPS"):
            select_device("mps")
