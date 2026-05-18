"""Tests for compute_metrics."""

import numpy as np
import pytest

from clmm.trainer import compute_metrics


def test_single_task_no_forgetting():
    acc = np.array([[0.8]])
    avg_acc, forgetting, bwt, final_aa, final_f, final_bwt = compute_metrics(acc)
    assert avg_acc[0] == pytest.approx(0.8)
    assert forgetting[0] == pytest.approx(0.0)
    assert bwt[0] == pytest.approx(0.0)
    assert final_aa == pytest.approx(0.8)


def test_two_tasks_no_forgetting():
    # Task 0 stays at 0.9; task 1 introduced at 0.8
    acc = np.array([
        [0.9, np.nan],
        [0.9, 0.8],
    ])
    avg_acc, forgetting, bwt, final_aa, final_f, final_bwt = compute_metrics(acc)
    # After task 1: avg = (0.9 + 0.8) / 2
    assert avg_acc[1] == pytest.approx(0.85)
    # No forgetting: task 0 still at 0.9
    assert forgetting[1] == pytest.approx(0.0)
    assert bwt[1] == pytest.approx(0.0)


def test_two_tasks_with_forgetting():
    acc = np.array([
        [0.9, np.nan],
        [0.7, 0.8],
    ])
    avg_acc, forgetting, bwt, final_aa, final_f, final_bwt = compute_metrics(acc)
    # Forgetting on task 0: best was 0.9, now 0.7 -> forget = 0.2
    assert forgetting[1] == pytest.approx(0.2, abs=1e-5)
    # BWT: current - first_seen = 0.7 - 0.9 = -0.2
    assert bwt[1] == pytest.approx(-0.2, abs=1e-5)


def test_three_tasks_avg_accuracy():
    acc = np.array([
        [0.8, np.nan, np.nan],
        [0.7, 0.9, np.nan],
        [0.6, 0.8, 0.7],
    ])
    avg_acc, _, _, _, _, _ = compute_metrics(acc)
    assert avg_acc[0] == pytest.approx(0.8)
    assert avg_acc[1] == pytest.approx((0.7 + 0.9) / 2)
    assert avg_acc[2] == pytest.approx((0.6 + 0.8 + 0.7) / 3)


def test_forgetting_is_nonnegative():
    # Even if accuracy improves, forgetting is clipped at 0
    acc = np.array([
        [0.7, np.nan],
        [0.9, 0.8],
    ])
    _, forgetting, _, _, _, _ = compute_metrics(acc)
    assert forgetting[1] == pytest.approx(0.0)


def test_output_shapes():
    T = 4
    acc = np.full((T, T), np.nan)
    for i in range(T):
        for j in range(i + 1):
            acc[i, j] = 0.5 + 0.1 * (i - j)
    avg_acc, forgetting, bwt, _, _, _ = compute_metrics(acc)
    assert avg_acc.shape == (T,)
    assert forgetting.shape == (T,)
    assert bwt.shape == (T,)


def test_final_metrics_match_last_row():
    acc = np.array([
        [0.8, np.nan],
        [0.7, 0.85],
    ])
    avg_acc, forgetting, bwt, final_aa, final_f, final_bwt = compute_metrics(acc)
    assert final_aa == pytest.approx(avg_acc[-1])
    assert final_f == pytest.approx(forgetting[-1])
    assert final_bwt == pytest.approx(bwt[-1])
