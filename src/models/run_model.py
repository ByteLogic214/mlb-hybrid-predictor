from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom, poisson
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class RunMarketProbabilities:
    family: str
    dispersion_alpha: float
    expected_home_runs: float
    expected_away_runs: float
    expected_run_differential: float
    distribution_home_win_probability: float
    home_cover_probability: float
    run_line_push_probability: float
    over_probability: float
    total_push_probability: float


class RunDistributionModel:
    """Separate team-run means with Poisson or overdispersed Negative Binomial errors."""

    def __init__(self, feature_columns: list[str], categorical_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        numeric = [column for column in feature_columns if column not in categorical_columns]
        self.transformer = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                ),
            ],
            verbose_feature_names_out=False,
        )
        self.family_name = "poisson"
        self.alpha = 0.0
        self.result: Any | None = None

    def fit(self, features: pd.DataFrame, home_runs: np.ndarray, away_runs: np.ndarray) -> None:
        base = features[self.feature_columns].reset_index(drop=True)
        stacked = pd.concat([base.assign(team_is_home=1.0), base.assign(team_is_home=0.0)])
        target = np.concatenate([home_runs.astype(float), away_runs.astype(float)])
        matrix = self.transformer.fit_transform(stacked[self.feature_columns])
        home_flag = stacked["team_is_home"].to_numpy().reshape(-1, 1)
        design = sm.add_constant(np.column_stack([matrix, home_flag]), has_constant="add")
        mean = float(np.mean(target))
        variance = float(np.var(target, ddof=1))
        self.alpha = max((variance - mean) / max(mean**2, 1e-9), 1e-6)
        if variance > mean * 1.10:
            self.family_name = "negative_binomial"
            family = sm.families.NegativeBinomial(alpha=self.alpha)
        else:
            self.family_name = "poisson"
            self.alpha = 0.0
            family = sm.families.Poisson()
        self.result = sm.GLM(target, design, family=family).fit_regularized(
            alpha=0.01, L1_wt=0.0, maxiter=1000
        )

    def expected_runs(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.result is None:
            raise RuntimeError("Run model has not been fitted")
        base = features[self.feature_columns]
        matrix = self.transformer.transform(base)
        home_design = sm.add_constant(
            np.column_stack([matrix, np.ones(len(base))]), has_constant="add"
        )
        away_design = sm.add_constant(
            np.column_stack([matrix, np.zeros(len(base))]), has_constant="add"
        )
        home = np.asarray(self.result.predict(home_design), dtype=float)
        away = np.asarray(self.result.predict(away_design), dtype=float)
        return np.clip(home, 0.05, 20.0), np.clip(away, 0.05, 20.0)

    def market_probabilities(
        self,
        features: pd.DataFrame,
        *,
        home_run_line: float,
        total_line: float,
        maximum_runs: int,
    ) -> RunMarketProbabilities:
        home_expected, away_expected = self.expected_runs(features)
        if len(home_expected) != 1:
            raise ValueError("market_probabilities requires exactly one game")
        home_pmf = self._pmf(float(home_expected[0]), maximum_runs)
        away_pmf = self._pmf(float(away_expected[0]), maximum_runs)
        joint = np.outer(home_pmf, away_pmf)
        home_grid, away_grid = np.meshgrid(
            np.arange(maximum_runs + 1), np.arange(maximum_runs + 1), indexing="ij"
        )
        tie_probability = float(joint[home_grid == away_grid].sum())
        regulation_home_win = float(joint[home_grid > away_grid].sum())
        non_tie = 1.0 - tie_probability
        conditioned_home_win = regulation_home_win / non_tie if non_tie > 0 else 0.5
        adjusted_margin = home_grid + home_run_line - away_grid
        totals = home_grid + away_grid
        return RunMarketProbabilities(
            family=self.family_name,
            dispersion_alpha=self.alpha,
            expected_home_runs=float(home_expected[0]),
            expected_away_runs=float(away_expected[0]),
            expected_run_differential=float(home_expected[0] - away_expected[0]),
            distribution_home_win_probability=conditioned_home_win,
            home_cover_probability=float(joint[adjusted_margin > 0].sum()),
            run_line_push_probability=float(joint[np.isclose(adjusted_margin, 0)].sum()),
            over_probability=float(joint[totals > total_line].sum()),
            total_push_probability=float(joint[np.isclose(totals, total_line)].sum()),
        )

    def _pmf(self, expected: float, maximum_runs: int) -> np.ndarray:
        values = np.arange(maximum_runs + 1)
        if self.family_name == "negative_binomial":
            size = 1.0 / self.alpha
            probability = size / (size + expected)
            pmf = np.asarray(nbinom.pmf(values, size, probability), dtype=float)
            pmf[-1] = float(nbinom.sf(maximum_runs - 1, size, probability))
        else:
            pmf = np.asarray(poisson.pmf(values, expected), dtype=float)
            pmf[-1] = float(poisson.sf(maximum_runs - 1, expected))
        return np.asarray(pmf / pmf.sum(), dtype=float)
