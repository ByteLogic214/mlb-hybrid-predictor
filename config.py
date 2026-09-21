from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class Settings:
    raw: Mapping[str, Any]
    path: Path

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"Missing mapping section: {name}")
        return value


def load_settings(path: str | Path = "config/default.yml") -> Settings:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a mapping")
    _validate(raw)
    return Settings(raw=raw, path=config_path)


def _validate(raw: Mapping[str, Any]) -> None:
    required = {"http", "training", "features", "models", "markets", "workflow"}
    missing = required.difference(raw)
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {sorted(missing)}")
    training = raw["training"]
    if not isinstance(training, Mapping):
        raise ConfigurationError("training must be a mapping")
    fraction = float(training.get("calibration_fraction", 0.0))
    if not 0.05 <= fraction <= 0.40:
        raise ConfigurationError("calibration_fraction must be between 0.05 and 0.40")
    markets = raw["markets"]
    if not isinstance(markets, Mapping):
        raise ConfigurationError("markets must be a mapping")
    if int(markets.get("maximum_runs_for_distribution", 0)) < 15:
        raise ConfigurationError("maximum_runs_for_distribution must be at least 15")
