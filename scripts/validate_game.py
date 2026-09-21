#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a real MLB game before prediction"
    )
    parser.add_argument("--game-pk", required=True, type=int)
    parser.add_argument("--config", default="config/default.yml")
    parser.add_argument("--output", default="artifacts/validated_game.json")
    args = parser.parse_args()

    if args.game_pk <= 0:
        parser.error("--game-pk must be a positive integer")

    url = (
        f"https://statsapi.mlb.com/api/v1.1/game/"
        f"{args.game_pk}/feed/live"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "mlb-hybrid-predictor/1.0",
        },
    )

    with urlopen(request, timeout=30) as response:
        raw_data = response.read()

    feed = json.loads(raw_data.decode("utf-8"))
    game_data = feed["gameData"]
    status = game_data["status"]
    official_state = status.get("abstractGameState", "")

    if official_state != "Preview":
        raise RuntimeError(
            f"game_pk={args.game_pk} is not pregame; "
            f"official state is {official_state or 'unknown'}"
        )

    scheduled_start = game_data["datetime"]["dateTime"]
    scheduled_datetime = datetime.fromisoformat(
        scheduled_start.replace("Z", "+00:00")
    )

    if scheduled_datetime.tzinfo is None:
        scheduled_datetime = scheduled_datetime.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) >= scheduled_datetime:
        raise RuntimeError(
            "Scheduled start has passed; pregame validation refused"
        )

    teams = game_data["teams"]
    for side in ("away", "home"):
        team = teams[side]
        if int(team.get("sport", {}).get("id", 0)) != 1:
            raise RuntimeError(
                f"game_pk={args.game_pk} is not an MLB game"
            )

    output = {
        "game_pk": args.game_pk,
        "status": status,
        "scheduled_start": scheduled_start,
        "teams": teams,
        "venue": game_data["venue"],
        "configuration": args.config,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "official-public-only",
        "records": [
            {
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(raw_data).hexdigest(),
                "cache_hit": False,
            }
        ],
    }

    provenance_path = output_path.with_name(
        "validation_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
