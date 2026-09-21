from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from api.mlb_stats import MLBStatsAPI
from evaluation.backtest import append_and_evaluate, score_prediction
from pipeline.predict import PredictionPipeline
from utils.config import Settings, load_settings
from utils.logging import configure_logging
from utils.real_data import (
    reject_forbidden_markers,
    require_positive_game_pk,
    validate_source_urls,
)


def validate_main() -> None:
    parser = _common_parser("Validate a real MLB game_pk before any model execution")
    parser.add_argument("--game-pk", required=True)
    parser.add_argument("--output", default="artifacts/validated_game.json")
    args = parser.parse_args()
    settings, client = _runtime(args.config)
    game_pk = require_positive_game_pk(args.game_pk)
    feed = client.validate_pregame(game_pk)
    game_data = feed["gameData"]
    output = {
        "game_pk": game_pk,
        "status": game_data["status"],
        "scheduled_start": game_data["datetime"]["dateTime"],
        "teams": game_data["teams"],
        "venue": game_data["venue"],
        "configuration": str(settings.path),
    }
    _write_json(args.output, output)
    client.write_provenance(Path(args.output).with_name("validation_provenance.json"))
    print(json.dumps(output, indent=2, sort_keys=True))


def predict_main() -> None:
    parser = _common_parser("Train, calibrate, evaluate, and predict a real pregame MLB game")
    parser.add_argument("--game-pk", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    settings, client = _runtime(args.config)
    game_pk = require_positive_game_pk(args.game_pk)
    feed = client.validate_pregame(game_pk)
    prediction = PredictionPipeline(client, settings.raw).run(feed, args.artifacts)
    print(json.dumps(prediction, indent=2, sort_keys=True))


def wait_main() -> None:
    parser = _common_parser("Wait for the official MLB final state")
    parser.add_argument("--game-pk", required=True)
    parser.add_argument("--output", default="artifacts/final_feed.json")
    args = parser.parse_args()
    settings, client = _runtime(args.config)
    game_pk = require_positive_game_pk(args.game_pk)
    workflow = settings.section("workflow")
    feed = client.wait_until_final(
        game_pk,
        poll_seconds=int(workflow["poll_seconds"]),
        maximum_wait_minutes=int(workflow["maximum_wait_minutes"]),
    )
    _write_json(args.output, feed)
    client.write_provenance(Path(args.output).with_name("final_provenance.json"))
    print(json.dumps({"game_pk": game_pk, "status": feed["gameData"]["status"]}, indent=2))


def backtest_main() -> None:
    parser = _common_parser("Score a saved prediction against an official final MLB feed")
    parser.add_argument("--prediction", default="artifacts/prediction.json")
    parser.add_argument("--final-feed", default="artifacts/final_feed.json")
    parser.add_argument("--ledger", default="data/backtests.csv")
    parser.add_argument("--output", default="artifacts/backtest_report.json")
    args = parser.parse_args()
    _settings, client = _runtime(args.config)
    prediction = _read_json(args.prediction)
    final_path = Path(args.final_feed)
    if final_path.is_file():
        final_feed = _read_json(final_path)
    else:
        final_feed = client.live_feed(int(prediction["game_pk"]), fresh=True)
    result = score_prediction(prediction, final_feed)
    report = append_and_evaluate(result, args.ledger)
    payload = {"result": result, "cumulative_metrics": report}
    reject_forbidden_markers(payload)
    _write_json(args.output, payload)
    print("OFFICIAL POSTGAME BACKTEST — CUMULATIVE MATHEMATICAL METRICS")
    print(json.dumps(payload, indent=2, sort_keys=True))


def enforce_main() -> None:
    parser = argparse.ArgumentParser(description="Enforce official-public-only artifact provenance")
    parser.add_argument("--prediction", default="artifacts/prediction.json")
    parser.add_argument("--provenance", default="artifacts/provenance.json")
    args = parser.parse_args()
    prediction = _read_json(args.prediction)
    provenance = _read_json(args.provenance)
    reject_forbidden_markers(prediction)
    records = provenance.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Provenance records are missing")
    validate_source_urls(str(record["url"]) for record in records)
    print(f"Verified {len(records)} official public source responses")


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/default.yml")
    return parser


def _runtime(config_path: str) -> tuple[Settings, MLBStatsAPI]:
    configure_logging()
    settings = load_settings(config_path)
    return settings, MLBStatsAPI.from_config(settings.section("http"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value
