from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class BinaryEstimator(Protocol):
    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Any: ...

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


EstimatorFactory = Callable[[], BinaryEstimator]


def classifier_factories(
    feature_columns: list[str],
    categorical_columns: list[str],
    config: Mapping[str, Any],
    random_seed: int,
) -> dict[str, EstimatorFactory]:
    numeric = [column for column in feature_columns if column not in categorical_columns]

    def preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            [
                ("numeric", "passthrough", numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    xgb_config = config["xgboost"]
    rf_config = config["random_forest"]
    cat_config = config["catboost"]

    def make_xgboost() -> BinaryEstimator:
        return cast(
            BinaryEstimator,
            Pipeline(
                [
                    ("features", preprocessor()),
                    (
                        "classifier",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_estimators=int(xgb_config["n_estimators"]),
                            max_depth=int(xgb_config["max_depth"]),
                            learning_rate=float(xgb_config["learning_rate"]),
                            subsample=float(xgb_config["subsample"]),
                            colsample_bytree=float(xgb_config["colsample_bytree"]),
                            reg_lambda=float(xgb_config["reg_lambda"]),
                            random_state=random_seed,
                            n_jobs=2,
                        ),
                    ),
                ],
            ),
        )

    def make_random_forest() -> BinaryEstimator:
        return cast(
            BinaryEstimator,
            Pipeline(
                [
                    ("features", preprocessor()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=int(rf_config["n_estimators"]),
                            min_samples_leaf=int(rf_config["min_samples_leaf"]),
                            max_features=str(rf_config["max_features"]),
                            class_weight="balanced_subsample",
                            random_state=random_seed,
                            n_jobs=2,
                        ),
                    ),
                ],
            ),
        )

    def make_catboost() -> BinaryEstimator:
        return CatBoostAdapter(
            categorical_columns=categorical_columns,
            iterations=int(cat_config["iterations"]),
            depth=int(cat_config["depth"]),
            learning_rate=float(cat_config["learning_rate"]),
            l2_leaf_reg=float(cat_config["l2_leaf_reg"]),
            random_seed=random_seed,
        )

    return {
        "xgboost": make_xgboost,
        "random_forest": make_random_forest,
        "catboost": make_catboost,
    }


class CatBoostAdapter:
    def __init__(
        self,
        *,
        categorical_columns: list[str],
        iterations: int,
        depth: int,
        learning_rate: float,
        l2_leaf_reg: float,
        random_seed: int,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> CatBoostAdapter:
        prepared = self._prepare(features)
        self.model.fit(prepared, target, cat_features=self.categorical_columns)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._prepare(features)), dtype=float)

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        result = features.copy()
        for column in self.categorical_columns:
            result[column] = result[column].astype(str)
        return result
