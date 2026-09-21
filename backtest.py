from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.metrics import binary_probability_metrics


class BacktestError(RuntimeError):
    """Prediction and official final result cannot be scored safely."""


def official_score(feed: Mapping[str, Any]) -> tuple[int, int]:
    game_data = _mapping(feed, "gameData")
    status = _mapping(game_data, "status")
    if status.get("abstractGameState") != "Final" and status.get("detailedState") not in {
        "Final",
        "Game Over",
        "Completed Early",
    }:
        raise BacktestError(f"Official game state is not final: {status.get('detailedState')}")
    live_data = _mapping(feed, "liveData")
    teams = _mapping(_mapping(live_data, "linescore"), "teams")
    home = int(_mapping(teams, "home")["runs"])
    away = int(_mapping(teams, "away")["runs"])
    if home == away:
        raise BacktestError("Official final score is tied and cannot be scored")
    return home, away


def score_prediction(
    prediction: Mapping[str, Any], final_feed: Mapping[str, Any]
) -> dict[str, Any]:
    final_game_pk = int(_mapping(_mapping(final_feed, "gameData"), "game")["pk"])
    prediction_game_pk = int(prediction["game_pk"])
    if final_game_pk != prediction_game_pk:
        raise BacktestError("Prediction and final feed game_pk do not match")
    home_runs, away_runs = official_score(final_feed)
    markets = _mapping(prediction, "markets")
    home_line = float(markets["home_run_line"])
    total_line = float(markets["total_line"])
    run_line_margin = home_runs + home_line - away_runs
    total_margin = home_runs + away_runs - total_line
    teams = _mapping(prediction, "teams")
    home_team = _mapping(teams, "home")
    away_team = _mapping(teams, "away")
    return {
        "game_pk": prediction_game_pk,
        "generated_at": prediction["generated_at"],
        "scored_at": datetime.now(UTC).isoformat(),
        "scheduled_start": prediction["scheduled_start"],
        "home_team_id": int(home_team["id"]),
        "home_team_name": str(home_team["name"]),
        "away_team_id": int(away_team["id"]),
        "away_team_name": str(away_team["name"]),
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_win_probability": float(_mapping(prediction, "moneyline")["home_win_probability"]),
        "home_win_observed": int(home_runs > away_runs),
        "home_run_line": home_line,
        "home_cover_probability": float(_mapping(prediction, "runs")["home_cover_probability"]),
        "home_cover_observed": None if run_line_margin == 0 else int(run_line_margin > 0),
        "total_line": total_line,
        "over_probability": float(_mapping(prediction, "runs")["over_probability"]),
        "over_observed": None if total_margin == 0 else int(total_margin > 0),
    }


def append_and_evaluate(
    result: Mapping[str, Any], csv_path: str | Path
) -> dict[str, Any]:
    output = Path(csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result.keys())
    existing_rows: list[dict[str, Any]] = []
    if output.is_file():
        with output.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fieldnames:
                raise BacktestError(
                    "Existing backtest ledger schema does not match current version"
                )
            existing_rows = list(reader)
    deduplicated = [row for row in existing_rows if int(row["game_pk"]) != int(result["game_pk"])]
    deduplicated.append(dict(result))
    deduplicated.sort(key=lambda row: str(row["scheduled_start"]))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduplicated)
    frame = pd.DataFrame(deduplicated)
    report = {
        "games": len(frame),
        "moneyline": binary_probability_metrics(
            frame["home_win_observed"].astype(int).to_numpy(),
            frame["home_win_probability"].astype(float).to_numpy(),
        ),
        "run_line": _market_metrics(frame, "home_cover_observed", "home_cover_probability"),
        "total": _market_metrics(frame, "over_observed", "over_probability"),
    }
    serialized = json.dumps(report, allow_nan=False)
    parsed = json.loads(serialized)
    if not isinstance(parsed, dict):
        raise BacktestError("Cumulative report did not serialize to an object")
    return parsed


def _market_metrics(frame: pd.DataFrame, outcome: str, probability: str) -> dict[str, Any]:
    clean = frame.loc[frame[outcome].notna() & (frame[outcome].astype(str) != "")]
    if clean.empty:
        return {"count": 0, "note": "All recorded results were pushes"}
    return binary_probability_metrics(
        clean[outcome].astype(int).to_numpy(), clean[probability].astype(float).to_numpy()
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise BacktestError(f"Missing object: {key}")
    return child
