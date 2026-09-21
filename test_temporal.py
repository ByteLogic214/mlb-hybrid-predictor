import numpy as np

from evaluation.temporal import expanding_window_folds


def test_expanding_windows_never_cross_the_future_boundary() -> None:
    folds = expanding_window_folds(300, folds=3, calibration_fraction=0.20)
    previous_test_end = 0
    for fold in folds:
        assert np.max(fold.train_indices) < np.min(fold.calibration_indices)
        assert np.max(fold.calibration_indices) < np.min(fold.test_indices)
        assert np.min(fold.train_indices) == 0
        assert np.max(fold.test_indices) >= previous_test_end
        previous_test_end = int(np.max(fold.test_indices))

