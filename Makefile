.PHONY: install test lint typecheck validate predict backtest clean

PYTHON ?= python3
GAME_PK ?=

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy

validate:
	@test -n "$(GAME_PK)" || (echo "GAME_PK is required" >&2; exit 2)
	$(PYTHON) scripts/validate_game.py --game-pk $(GAME_PK)

predict:
	@test -n "$(GAME_PK)" || (echo "GAME_PK is required" >&2; exit 2)
	$(PYTHON) scripts/predict.py --game-pk $(GAME_PK)

backtest:
	$(PYTHON) scripts/backtest.py --prediction artifacts/prediction.json

clean:
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache artifacts

