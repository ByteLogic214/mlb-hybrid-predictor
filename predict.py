from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.mlb_stats import MLBStatsAPI
from data.dataset import HistoricalDatasetBuilder, PreparedData
from evaluation.metrics import binary_probability_metrics
from evaluation.temporal import expanding_window_folds, final_train_calibration_split
from models.calibration import CalibratedBinaryModel
from models.classifiers import classifier_factories
from models.run_model import RunDistributionModel
from utils.logging import log_event
from utils.real_data import reject_forbidden_markers, validate_source_urls
from utils.time import utc_now

LOGGER = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, client: MLBStatsAPI, settings: Mapping[str, Any]) -> None:
        self.client = client
        self.settings = settings

    def run(self, target_feed: Mapping[str, Any], artifact_directory: str | Path) -> dict[str, Any]:
        artifacts = Path(artifact_directory)
        artifacts.mkdir(parents=True, exist_ok=True)
        builder = HistoricalDatasetBuilder(
            self.client,
            self.settings["training"],
            self.settings["features"],
        )
        prepared = builder.build(target_feed)
        training = prepared.training
        frame = training.frame
        _assert_chronological(frame)
        features = frame[training.feature_columns]
        target_features = prepared.target[training.feature_columns]
        target = frame["home_win"].astype(int).to_numpy()
        seed = int(self.settings["training"]["random_seed"])
        factories = classifier_factories(
            training.feature_columns,
            training.categorical_columns,
            self.settings["models"],
            seed,
        )
        folds = expanding_window_folds(
            len(frame),
            int(self.settings["training"]["walk_forward_folds"]),
            float(self.settings["training"]["calibration_fraction"]),
        )
        evaluation: dict[str, Any] = {
            "protocol": "strict expanding-window train-past/calibrate-next/test-future",
            "folds": len(folds),
            "moneyline": {},
        }
        predictions: dict[str, float] = {}
        calibrations: dict[str, str] = {}
        for model_name, factory in factories.items():
            observed_parts: list[np.ndarray] = []
            probability_parts: list[np.ndarray] = []
            fold_methods: list[str] = []
            for fold in folds:
                model = CalibratedBinaryModel.fit(
                    factory(),
                    features.iloc[fold.train_indices],
                    target[fold.train_indices],
                    features.iloc[fold.calibration_indices],
                    target[fold.calibration_indices],
                )
                probability_parts.append(model.predict_proba(features.iloc[fold.test_indices]))
                observed_parts.append(target[fold.test_indices])
                fold_methods.append(model.calibrator.method)
            evaluation["moneyline"][model_name] = {
                **binary_probability_metrics(
                    np.concatenate(observed_parts), np.concatenate(probability_parts)
                ),
                "fold_calibration_methods": fold_methods,
            }
            train_indices, calibration_indices = final_train_calibration_split(
                len(frame), float(self.settings["training"]["calibration_fraction"])
            )
            final_model = CalibratedBinaryModel.fit(
                factory(),
                features.iloc[train_indices],
                target[train_indices],
                features.iloc[calibration_indices],
                target[calibration_indices],
            )
            predictions[model_name] = float(final_model.predict_proba(target_features)[0])
            calibrations[model_name] = final_model.calibrator.method
            joblib.dump(final_model, artifacts / f"moneyline_{model_name}.joblib")
            log_event(
                LOGGER,
                logging.INFO,
                "model_evaluated",
                model=model_name,
                metrics=evaluation["moneyline"][model_name],
            )

        run_evaluation = self._evaluate_runs(prepared, folds)
        evaluation["run_line"] = run_evaluation["run_line"]
        evaluation["total"] = run_evaluation["total"]
        final_run_model = RunDistributionModel(
            training.feature_columns, training.categorical_columns
        )
        final_run_model.fit(
            features,
            frame["home_runs"].to_numpy(),
            frame["away_runs"].to_numpy(),
        )
        markets = self.settings["markets"]
        run_probabilities = final_run_model.market_probabilities(
            target_features,
            home_run_line=float(markets["home_run_line"]),
            total_line=float(markets["total_line"]),
            maximum_runs=int(markets["maximum_runs_for_distribution"]),
        )
        joblib.dump(final_run_model, artifacts / "run_distribution.joblib")

        identity = prepared.identity
        prediction: dict[str, Any] = {
            "schema_version": "1.0",
            "game_pk": identity.game_pk,
            "generated_at": utc_now().isoformat(),
            "scheduled_start": identity.game_date.isoformat(),
            "source_cutoff": utc_now().isoformat(),
            "teams": {
                "home": {"id": identity.home_team_id, "name": identity.home_team_name},
                "away": {"id": identity.away_team_id, "name": identity.away_team_name},
            },
            "venue": {"id": identity.venue_id, "name": identity.venue_name},
            "promoted_model": "xgboost",
            "moneyline": {
                "home_win_probability": predictions["xgboost"],
                "away_win_probability": 1.0 - predictions["xgboost"],
                "calibration": calibrations["xgboost"],
                "benchmarks": {
                    "random_forest": predictions["random_forest"],
                    "catboost": predictions["catboost"],
                },
            },
            "runs": asdict(run_probabilities),
            "markets": {
                "home_run_line": float(markets["home_run_line"]),
                "total_line": float(markets["total_line"]),
            },
            "training": {
                "rows": len(frame),
                "first_game_time": str(frame["game_date"].iloc[0]),
                "last_game_time": str(frame["game_date"].iloc[-1]),
                "processed_final_games": prepared.processed_final_games,
                "skipped_rows": prepared.skipped_feature_rows,
                "features": training.feature_columns,
                "excluded_features": training.excluded_columns,
            },
            "evaluation_file": "evaluation.json",
            "provenance_file": "provenance.json",
        }
        reject_forbidden_markers(prediction)
        validate_source_urls(record.url for record in self.client.provenance)
        _write_json(artifacts / "prediction.json", prediction)
        _write_json(artifacts / "evaluation.json", evaluation)
        frame.to_csv(artifacts / "training_dataset.csv", index=False)
        self.client.write_provenance(artifacts / "provenance.json")
        _print_metrics(evaluation)
        return prediction

    def _evaluate_runs(self, prepared: PreparedData, folds: list[Any]) -> dict[str, Any]:
        training = prepared.training
        frame = training.frame
        features = frame[training.feature_columns]
        run_line_observed: list[int] = []
        run_line_predicted: list[float] = []
        total_observed: list[int] = []
        total_predicted: list[float] = []
        markets = self.settings["markets"]
        home_line = float(markets["home_run_line"])
        total_line = float(markets["total_line"])
        for fold in folds:
            past = np.concatenate([fold.train_indices, fold.calibration_indices])
            model = RunDistributionModel(training.feature_columns, training.categorical_columns)
            model.fit(
                features.iloc[past],
                frame["home_runs"].iloc[past].to_numpy(),
                frame["away_runs"].iloc[past].to_numpy(),
            )
            for index in fold.test_indices:
                market = model.market_probabilities(
                    features.iloc[[index]],
                    home_run_line=home_line,
                    total_line=total_line,
                    maximum_runs=int(markets["maximum_runs_for_distribution"]),
                )
                home_runs = float(frame["home_runs"].iloc[index])
                away_runs = float(frame["away_runs"].iloc[index])
                line_margin = home_runs + home_line - away_runs
                total_margin = home_runs + away_runs - total_line
                if line_margin != 0:
                    run_line_observed.append(int(line_margin > 0))
                    run_line_predicted.append(market.home_cover_probability)
                if total_margin != 0:
                    total_observed.append(int(total_margin > 0))
                    total_predicted.append(market.over_probability)
        return {
            "run_line": binary_probability_metrics(
                np.asarray(run_line_observed), np.asarray(run_line_predicted)
            ),
            "total": binary_probability_metrics(
                np.asarray(total_observed), np.asarray(total_predicted)
            ),
        }


def _assert_chronological(frame: pd.DataFrame) -> None:
    times = pd.to_datetime(frame["game_date"], utc=True)
    if not times.is_monotonic_increasing:
        raise ValueError("Training rows are not in chronological order")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _print_metrics(evaluation: Mapping[str, Any]) -> None:
    print("MLB HYBRID PREDICTOR — MATHEMATICAL EVALUATION")
    print(json.dumps(evaluation, indent=2, sort_keys=True, allow_nan=False))
