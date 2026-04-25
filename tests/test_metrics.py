"""Tests for continual metrics: accuracy matrix, forgetting, BWT."""

from __future__ import annotations

import math

import pytest

from clr_routing.continual.metrics import ContinualMetrics


def test_first_task_snapshot_has_zero_forgetting():
    m = ContinualMetrics(num_tasks=3)
    m.record(after_task=0, eval_task=0, accuracy=0.8)
    snap = m.snapshot(after_task=0)
    assert snap.average_accuracy == pytest.approx(0.8)
    assert snap.average_forgetting == 0.0
    assert snap.backward_transfer == 0.0


def test_forgetting_formula_against_handworked():
    """
    R matrix (3 tasks):
        [0.9, nan, nan]   <- after task 0
        [0.7, 0.8, nan]   <- after task 1
        [0.5, 0.6, 0.85]  <- after task 2
    Best-historical for task 0: max(0.9, 0.7) = 0.9; current = 0.5; forget = 0.4
    Best-historical for task 1: max(0.8) = 0.8; current = 0.6; forget = 0.2
    Avg forgetting after task 2 = (0.4 + 0.2) / 2 = 0.3
    BWT = ((0.5 - 0.9) + (0.6 - 0.8)) / 2 = -0.3
    Avg accuracy = (0.5 + 0.6 + 0.85) / 3
    """
    m = ContinualMetrics(num_tasks=3)
    m.record(0, 0, 0.9)
    m.record(1, 0, 0.7)
    m.record(1, 1, 0.8)
    m.record(2, 0, 0.5)
    m.record(2, 1, 0.6)
    m.record(2, 2, 0.85)

    snap = m.snapshot(after_task=2)
    assert snap.average_forgetting == pytest.approx(0.3)
    assert snap.backward_transfer == pytest.approx(-0.3)
    assert snap.average_accuracy == pytest.approx((0.5 + 0.6 + 0.85) / 3)


def test_per_task_accuracy_dict_contains_all_seen():
    m = ContinualMetrics(num_tasks=3)
    m.record(0, 0, 0.9)
    m.record(1, 0, 0.7)
    m.record(1, 1, 0.85)
    snap = m.snapshot(after_task=1)
    assert set(snap.per_task_accuracy.keys()) == {0, 1}
    assert snap.per_task_accuracy[0] == pytest.approx(0.7)
    assert snap.per_task_accuracy[1] == pytest.approx(0.85)


def test_matrix_returns_copy():
    m = ContinualMetrics(num_tasks=2)
    m.record(0, 0, 0.5)
    mat = m.matrix
    mat[0, 0] = 999.0
    snap = m.snapshot(after_task=0)
    # Original should be unchanged.
    assert snap.average_accuracy == pytest.approx(0.5)
