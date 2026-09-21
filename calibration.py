from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from models.classifiers import BinaryEstimator

EPSILON = 1e-6


@dataclass
class ProbabilityCalibrator:
    method: str
    model: Any | None = None

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
        if self.method == "identity":
            return raw
        if self.model is None:
            raise RuntimeError(f"Calibration model is missing for method: {self.method}")
        if self.method == "logistic":
            logits = np.log(raw / (1 - raw)).reshape(-1, 1)
            return np.asarray(self.model.predict_proba(logits)[:, 1], dtype=float)
        if self.method == "isotonic":
            return np.asarray(self.model.predict(raw), dtype=float)
        raise ValueError(f"Unknown calibration method: {self.method}")


@dataclass
class CalibratedBinaryModel:
    estimator: BinaryEstimator
    calibrator: ProbabilityCalibrator
    calibration_scores: dict[str, float]

    @classmethod
    def fit(
        cls,
        estimator: BinaryEstimator,
        train_features: pd.DataFrame,
        train_target: np.ndarray,
        calibration_features: pd.DataFrame,
        calibration_target: np.ndarray,
    ) -> CalibratedBinaryModel:
        if np.unique(train_target).size != 2 or np.unique(calibration_target).size != 2:
            raise ValueError("Training and calibration windows must each contain both outcomes")
        estimator.fit(train_features, train_target)
        raw = np.asarray(estimator.predict_proba(calibration_features)[:, 1], dtype=float)
        candidates: dict[str, ProbabilityCalibrator] = {
            "identity": ProbabilityCalibrator("identity")
        }
        logits = np.log(np.clip(raw, EPSILON, 1 - EPSILON) / (1 - np.clip(raw, EPSILON, 1 - EPSILON)))
        logistic = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        logistic.fit(logits.reshape(-1, 1), calibration_target)
        candidates["logistic"] = ProbabilityCalibrator("logistic", logistic)
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=EPSILON, y_max=1 - EPSILON)
        isotonic.fit(raw, calibration_target)
        candidates["isotonic"] = ProbabilityCalibrator("isotonic", isotonic)
        scores = {
            name: float(brier_score_loss(calibration_target, candidate.transform(raw)))
            for name, candidate in candidates.items()
        }
        preference = {"logistic": 0, "isotonic": 1, "identity": 2}
        selected = min(scores, key=lambda name: (scores[name], preference[name]))
        return cls(estimator=estimator, calibrator=candidates[selected], calibration_scores=scores)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=float)
        return self.calibrator.transform(raw)
