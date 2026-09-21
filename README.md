# Runtime data

This directory intentionally contains no game records. The pipeline downloads every record from
the official public MLB StatsAPI at execution time and writes only source-provenanced artifacts.
`backtests.csv` is created after an official final result and may be committed by GitHub Actions to
maintain the cumulative audit trail.

