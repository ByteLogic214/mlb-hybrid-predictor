# MLB Hybrid Predictor

Production-oriented, XGBoost-first MLB pregame probability modeling with strict temporal validation,
explicit calibration, count-based run markets, and an official postgame audit trail. It uses only
real public MLB StatsAPI observations and stops instead of inventing unavailable data.

## Repository tree

```text
mlb-hybrid-predictor/
├── .github/workflows/mlb_predict_and_backtest.yml
├── config/default.yml
├── data/README.md
├── docs/
│   ├── data_sources.md
│   ├── methodology.md
│   ├── operations.md
│   └── scientific_validation.md
├── scripts/
│   ├── backtest.py
│   ├── enforce_real_data.py
│   ├── predict.py
│   ├── validate_game.py
│   └── wait_for_final.py
├── src/
│   ├── api/mlb_stats.py
│   ├── data/{dataset.py,schema.py}
│   ├── evaluation/{backtest.py,metrics.py,temporal.py}
│   ├── features/{build.py,state.py}
│   ├── models/{calibration.py,classifiers.py,lstm.py,run_model.py}
│   ├── pipeline/{cli.py,predict.py}
│   └── utils/{config.py,logging.py,real_data.py,time.py}
├── tests/
├── Makefile
├── README.md
└── pyproject.toml
```

## What it produces

- calibrated home/away Moneyline probabilities from the promoted XGBoost model;
- Random Forest and CatBoost benchmark probabilities under the same temporal protocol;
- expected home/away runs and run differential from Poisson or Negative Binomial;
- configurable home Run Line cover and Over probabilities, including explicit push mass;
- walk-forward log loss, Brier score, accuracy, calibration intercept/slope, ECE, and reliability
  bins printed in GitHub Actions and stored as JSON;
- immutable pregame artifacts, official final feed, cumulative backtest CSV, and source provenance.

## Quick start

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
make test lint typecheck
python scripts/validate_game.py --game-pk REAL_FUTURE_MLB_GAME_PK
python scripts/predict.py --game-pk REAL_FUTURE_MLB_GAME_PK
```

See [operations](docs/operations.md), [methodology](docs/methodology.md), [data sources](docs/data_sources.md),
and [scientific validation](docs/scientific_validation.md). The code is MIT licensed.

