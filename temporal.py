from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalFold:
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray


def expanding_window_folds(
    sample_count: int, folds: int, calibration_fraction: float
) -> list[TemporalFold]:
    """Create strict train-past/calibrate-next/test-future expanding folds."""
    if folds < 2:
        raise ValueError("At least two walk-forward folds are required")
    if sample_count < 100:
        raise ValueError("At least 100 chronological rows are required")
    first_test_start = max(int(sample_count * 0.60), 60)
    remaining = sample_count - first_test_start
    test_size = max(1, remaining // folds)
    result: list[TemporalFold] = []
    for fold_index in range(folds):
        test_start = first_test_start + fold_index * test_size
        test_end = sample_count if fold_index == folds - 1 else min(
            test_start + test_size, sample_count
        )
        if test_start >= test_end:
            continue
        calibration_size = max(20, int(test_start * calibration_fraction))
        train_end = test_start - calibration_size
        if train_end < 40:
            continue
        result.append(
            TemporalFold(
                train_indices=np.arange(0, train_end),
                calibration_indices=np.arange(train_end, test_start),
                test_indices=np.arange(test_start, test_end),
            )
        )
    if len(result) < 2:
        raise ValueError("Unable to construct at least two valid expanding-window folds")
    return result


def final_train_calibration_split(
    sample_count: int, calibration_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    calibration_size = max(30, int(sample_count * calibration_fraction))
    split = sample_count - calibration_size
    if split < 50:
        raise ValueError("Insufficient rows for chronological final training and calibration")
    return np.arange(0, split), np.arange(split, sample_count)

