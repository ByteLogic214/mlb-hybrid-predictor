from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from api.mlb_stats import MLBStatsAPI
from data.schema import GameIdentity, TrainingDataset
from features.build import (
    add_official_result,
    build_feature_row,
    game_identity,
    update_state_from_final,
    venue_coordinates,
)
from features.state import FeatureDataError, LeagueState
from utils.logging import log_event
from utils.time import parse_mlb_datetime

LOGGER = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """The official source cannot support a leakage-safe dataset."""


@dataclass(frozen=True)
class PreparedData:
    training: TrainingDataset
    target: pd.DataFrame
    identity: GameIdentity
    processed_final_games: int
    skipped_feature_rows: int


class HistoricalDatasetBuilder:
    def __init__(
        self,
        client: MLBStatsAPI,
        training_config: Mapping[str, Any],
        feature_config: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.training_config = training_config
        self.feature_config = feature_config

    def build(self, target_feed: Mapping[str, Any]) -> PreparedData:
        target_identity = game_identity(target_feed)
        target_date = target_identity.game_date.date()
        start_date = target_date - timedelta(days=int(self.training_config["history_days"]))
        schedule = self.client.schedule(start_date, target_date)
        eligible = [
            game
            for game in schedule
            if int(game.get("gamePk", 0)) != target_identity.game_pk
            and _is_final(game)
            and _game_time(game) < target_identity.game_date
        ]
        eligible.sort(key=_game_time)
        maximum = int(self.training_config["max_training_games"])
        eligible = eligible[-maximum:]
        if not eligible:
            raise DatasetError("No final official MLB games exist before the target cutoff")

        state = LeagueState()
        venue_locations: dict[int, tuple[float, float]] = {}
        rows: list[dict[str, Any]] = []
        skipped = 0
        processed = 0
        for index, schedule_game in enumerate(eligible, start=1):
            game_pk = int(schedule_game["gamePk"])
            try:
                feed = self.client.live_feed(game_pk)
                identity = game_identity(feed, allow_final_starter_fallback=True)
                self._ensure_venue(identity.venue_id, venue_locations, feed)
                if self._state_ready(state, identity):
                    try:
                        row = build_feature_row(
                            identity=identity,
                            feed=feed,
                            state=state,
                            venue_locations=venue_locations,
                            config=self.feature_config,
                        )
                        add_official_result(row, feed)
                        rows.append(row)
                    except FeatureDataError as exc:
                        skipped += 1
                        log_event(
                            LOGGER,
                            logging.WARNING,
                            "historical_feature_row_skipped",
                            game_pk=game_pk,
                            reason=str(exc),
                        )
                update_state_from_final(state=state, identity=identity, feed=feed)
                processed += 1
            except FeatureDataError as exc:
                skipped += 1
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "historical_game_skipped",
                    game_pk=game_pk,
                    reason=str(exc),
                )
            if index % 50 == 0:
                log_event(
                    LOGGER,
                    logging.INFO,
                    "historical_progress",
                    fetched=index,
                    eligible=len(eligible),
                    training_rows=len(rows),
                )

        self._ensure_venue(target_identity.venue_id, venue_locations, target_feed)
        if not self._state_ready(state, target_identity):
            raise DatasetError(
                "Insufficient real prior history for both teams, probable starters, or venue"
            )
        try:
            target_row = build_feature_row(
                identity=target_identity,
                feed=target_feed,
                state=state,
                venue_locations=venue_locations,
                config=self.feature_config,
            )
        except FeatureDataError as exc:
            raise DatasetError(
                f"Cannot build target features from real public data: {exc}"
            ) from exc
        minimum_rows = int(self.training_config["minimum_training_rows"])
        if len(rows) < minimum_rows:
            raise DatasetError(
                f"Only {len(rows)} leakage-safe rows were available; {minimum_rows} are required"
            )
        training_frame = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
        target_frame = pd.DataFrame([target_row])
        training_dataset, target_selected = _select_complete_features(
            training_frame, target_frame
        )
        return PreparedData(
            training=training_dataset,
            target=target_selected,
            identity=target_identity,
            processed_final_games=processed,
            skipped_feature_rows=skipped,
        )

    def _state_ready(self, state: LeagueState, identity: GameIdentity) -> bool:
        return state.ready(
            home_team_id=identity.home_team_id,
            away_team_id=identity.away_team_id,
            home_starter_id=identity.home_starter_id,
            away_starter_id=identity.away_starter_id,
            venue_id=identity.venue_id,
            minimum_team_games=int(self.training_config["minimum_team_games"]),
            minimum_starter_outings=int(self.feature_config["minimum_starter_outings"]),
            minimum_venue_games=int(self.feature_config["minimum_venue_games"]),
        )

    def _ensure_venue(
        self,
        venue_id: int,
        venue_locations: dict[int, tuple[float, float]],
        feed: Mapping[str, Any],
    ) -> None:
        if venue_id not in venue_locations:
            game_data = feed.get("gameData")
            feed_venue = game_data.get("venue") if isinstance(game_data, Mapping) else None
            if isinstance(feed_venue, Mapping):
                try:
                    venue_locations[venue_id] = venue_coordinates(feed_venue)
                    return
                except FeatureDataError:
                    pass
            venue_locations[venue_id] = venue_coordinates(self.client.venue(venue_id))


def _select_complete_features(
    training: pd.DataFrame, target: pd.DataFrame
) -> tuple[TrainingDataset, pd.DataFrame]:
    reserved = {
        "game_pk",
        "game_date",
        "home_win",
        "home_runs",
        "away_runs",
        "run_diff",
        "total_runs",
    }
    categorical = ["home_team_id", "away_team_id", "venue_id"]
    candidates = [column for column in training.columns if column not in reserved]
    selected: list[str] = []
    excluded: dict[str, str] = {}
    for column in candidates:
        if column not in target.columns:
            excluded[column] = "absent from pregame target"
            continue
        if training[column].isna().any() or target[column].isna().any():
            excluded[column] = "missing in official source for at least one required observation"
            continue
        if column not in categorical:
            train_values = pd.to_numeric(training[column], errors="coerce")
            target_values = pd.to_numeric(target[column], errors="coerce")
            if not np.isfinite(train_values).all() or not np.isfinite(target_values).all():
                excluded[column] = "non-finite official or derived value"
                continue
        selected.append(column)
    selected_categorical = [column for column in categorical if column in selected]
    if len(selected) < 15:
        raise DatasetError(
            f"Only {len(selected)} complete real features remain; at least 15 required"
        )
    target_columns = ["home_win", "home_runs", "away_runs", "run_diff", "total_runs"]
    frame = training[["game_pk", "game_date", *selected, *target_columns]].copy()
    return (
        TrainingDataset(
            frame=frame,
            feature_columns=selected,
            categorical_columns=selected_categorical,
            excluded_columns=excluded,
        ),
        target[["game_pk", "game_date", *selected]].copy(),
    )


def _is_final(game: Mapping[str, Any]) -> bool:
    status = game.get("status")
    if not isinstance(status, Mapping):
        return False
    return status.get("abstractGameState") == "Final" or status.get("detailedState") in {
        "Final",
        "Game Over",
        "Completed Early",
    }


def _game_time(game: Mapping[str, Any]) -> datetime:
    raw = game.get("gameDate")
    if not isinstance(raw, str):
        raise DatasetError("Official schedule game has no gameDate")
    return parse_mlb_datetime(raw)
