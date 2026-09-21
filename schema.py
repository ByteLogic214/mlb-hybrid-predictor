from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GameIdentity:
    game_pk: int
    game_date: datetime
    official_date: str
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    venue_id: int
    venue_name: str
    home_starter_id: int
    away_starter_id: int


@dataclass(frozen=True)
class TrainingDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: dict[str, str]

    @property
    def target_columns(self) -> list[str]:
        return ["home_win", "home_runs", "away_runs", "run_diff", "total_runs"]


JSON = dict[str, Any]

