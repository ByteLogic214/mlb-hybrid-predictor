# Data sources and provenance

## Active source

Every runtime observation comes from the official public MLB StatsAPI at
`https://statsapi.mlb.com`. The client accepts no alternative host. It records the complete URL,
UTC retrieval time, SHA-256 response digest, and cache status in `artifacts/provenance.json`.

The pipeline uses:

- `/api/v1/schedule` for the historical game index, start times, and official status;
- `/api/v1.1/game/{game_pk}/feed/live` for official game metadata, boxscores, player appearances,
  and final scores;
- venue metadata embedded in each official live feed for public coordinates used by the
  travel-distance proxy; `/api/v1/venues/{venue_id}` is retained as a strict fallback.

The target `game_pk` is resolved against the live endpoint before training. Execution is rejected
unless it is an MLB game in official `Preview` state and its scheduled start is still in the future.
The pregame feature vector never reads `liveData.linescore`, the final boxscore, winner, or any
postgame event from the target game.

## Permitted future sources

Baseball Savant/Statcast, Retrosheet, and Lahman are acceptable only through stable, public,
reproducible downloads with their own provenance adapter. They are not silently scraped here.
Baseball Reference is not used because reproducible automated public access is not guaranteed.
Adding any source requires an allowlisted HTTPS host, content digest, cutoff timestamp, schema
validation, and a temporal-leakage test.

## Missing values

No game, player, score, pitch, or metric is fabricated. A feature absent for the target or any
required training observation is removed from that run and listed under `excluded_features`.
There is no statistical imputation. If fewer than 15 complete features or fewer than the configured
minimum training rows remain, execution fails with an explicit error.
