# Operations

## Local pregame run

Use Python 3.11. Choose a real future game from the public MLB schedule, then run:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/validate_game.py --game-pk REAL_GAME_PK
python scripts/predict.py --game-pk REAL_GAME_PK
```

The prediction command can make hundreds of rate-limited public requests on a cold cache. Repeated
runs reuse response bytes whose URLs and SHA-256 digests remain in the provenance manifest.

## GitHub Actions

Open **Actions → MLB pregame prediction and official backtest → Run workflow** and enter a real
`game_pk` shortly before first pitch. The job has a six-hour ceiling and the official-final polling
window is 340 minutes. A game scheduled too far in the future should be dispatched later. The job
uploads the immutable pregame prediction before polling, preventing postgame data from changing it.

Repository `contents: write` permission is required to persist the cumulative ledger. Branch
protection must permit the GitHub Actions bot or the final commit will fail visibly; prediction and
postgame artifacts still remain attached to the workflow run.

## Failure policy

HTTP 429 and transient 5xx responses use bounded exponential retry and honor `Retry-After`.
Requests are rate-limited to four per second. Schema drift, absent probable pitchers, inadequate
prior starts, missing venue coordinates, insufficient complete rows, source-host changes, or absent
official final state are hard failures. No fallback creates replacement records.

