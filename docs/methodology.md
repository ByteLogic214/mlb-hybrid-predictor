# Methodology

## Moneyline

XGBoost is the promoted model because boosted trees repeatedly provide the strongest tabular
baseline in the comparative evidence summarized in the project brief. Random Forest is retained as
a stable benchmark. CatBoost is retained as a challenger because team and venue identities are
genuinely categorical, but the repository does not describe it as an established MLB consensus.

Each classifier is trained on an older window and calibrated only on the next chronological window.
Identity, logistic (Platt), and isotonic calibration are compared by Brier score; the best is stored
with the fitted estimator. The final public probability always comes from calibrated XGBoost.

The optional LSTM is isolated in `src/models/lstm.py`, requires the `lstm` dependency extra, and is
never called by the production workflow. It cannot silently replace the tabular baseline.

## Features

All rolling inputs are accumulated from already-final games before the row being generated:

- win rate, role-specific home/away win rate, Log5, and Pythagorean expectation (exponent 1.83);
- OBP, SLG, OPS, and a documented wOBA proxy using the coefficients in `config/default.yml`;
- team and starter FIP proxy: `(13 HR + 3 (BB + HBP) - 2 K) / IP + 3.10`;
- starter rolling innings, strikeout rate, walk rate, pitch count, and contact-out proxy;
- bullpen pitches, appearances, and consecutive-day usage over the configured recent window;
- park run factor based only on earlier games at that venue versus the earlier league mean;
- rest days and great-circle travel miles from public venue coordinates;
- official temperature and wind when consistently present. Missing weather is excluded, not filled.

The wOBA weights are a documented stable proxy, not a claim that a season-specific league constant
was available from MLB StatsAPI. xFIP and hard-hit rate are excluded because this implementation has
no stable official public component feed for every temporal row.

## Runs and markets

The run model stacks home-team and away-team outcomes and fits a log-link count GLM. It uses Poisson
when the observed training variance is close to its mean and Negative Binomial when variance exceeds
the mean by more than 10%. The Negative Binomial dispersion is estimated from the real training
outcomes by method of moments.

Independent team-run distributions yield expected runs, expected run differential, home run-line
cover probability, total over probability, and push probabilities. Distributional moneyline is
reported conditional on a non-tied regulation score; the promoted moneyline remains calibrated
XGBoost. Run line and total thresholds are configurable.

