from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

EPSILON = 1e-6


def binary_probability_metrics(
    observed: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> dict[str, Any]:
    outcomes = np.asarray(observed, dtype=int)
    predicted = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    if outcomes.ndim != 1 or predicted.ndim != 1 or len(outcomes) != len(predicted):
        raise ValueError("Observed outcomes and probabilities must be equal-length vectors")
    if len(outcomes) == 0:
        raise ValueError("At least one observation is required")
    result: dict[str, Any] = {
        "count": int(len(outcomes)),
        "log_loss": float(log_loss(outcomes, predicted, labels=[0, 1])),
        "brier_score": float(brier_score_loss(outcomes, predicted)),
        "accuracy": float(accuracy_score(outcomes, predicted >= 0.5)),
        "reliability_bins": reliability_bins(outcomes, predicted, bins=bins),
    }
    if np.unique(outcomes).size == 2 and len(outcomes) >= 10:
        logits = np.log(predicted / (1 - predicted)).reshape(-1, 1)
        calibration = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        calibration.fit(logits, outcomes)
        result["calibration_intercept"] = float(calibration.intercept_[0])
        result["calibration_slope"] = float(calibration.coef_[0][0])
    else:
        result["calibration_intercept"] = None
        result["calibration_slope"] = None
        result["calibration_note"] = "At least 10 outcomes containing both classes are required"
    result["expected_calibration_error"] = float(
        sum(
            bucket["count"] / len(outcomes)
            * abs(bucket["mean_probability"] - bucket["observed_frequency"])
            for bucket in result["reliability_bins"]
        )
    )
    return result


def reliability_bins(
    observed: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        result.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_probability": float(np.mean(probabilities[mask])),
                "observed_frequency": float(np.mean(observed[mask])),
            }
        )
    return result

