# Scientific validation

Rows are sorted by official scheduled time. The evaluator creates expanding folds with three
strictly ordered segments:

1. train on the oldest observations;
2. select probability calibration on the immediately following observations;
3. evaluate once on the later, untouched observations.

No random split exists in the code. For Moneyline, every model reports log loss, Brier score,
accuracy, calibration intercept, calibration slope, expected calibration error, and equal-width
reliability bins. Run Line and Total report the same probability diagnostics; pushes are explicitly
excluded rather than relabeled.

Calibration intercept and slope come from an unpenalized logistic regression of the observed result
on the logit of the predicted probability. A perfect calibration map has intercept 0 and slope 1.
The Brier score is the mean squared probability error. Log loss clips only at `1e-6` for numerical
stability and strongly penalizes confident mistakes.

After prediction, GitHub Actions waits for the official final state, derives Moneyline, Run Line,
and Total outcomes from the official linescore, appends one deduplicated row to
`data/backtests.csv`, recomputes cumulative diagnostics, prints them, uploads JSON/CSV artifacts,
and commits the ledger. Cancelled, postponed, suspended, tied, mismatched, or non-final games fail
without manufacturing a result.

