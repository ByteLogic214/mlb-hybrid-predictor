from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse


class RealDataViolation(RuntimeError):
    """Raised when an artifact is not traceable to an approved public source."""


APPROVED_HOSTS = {"statsapi.mlb.com"}
FORBIDDEN_MARKERS = {"fake", "dummy", "mock", "placeholder", "simulated", "synthetic"}


def require_positive_game_pk(value: int | str) -> int:
    try:
        game_pk = int(value)
    except (TypeError, ValueError) as exc:
        raise RealDataViolation("game_pk must be an integer returned by MLB StatsAPI") from exc
    if game_pk <= 0:
        raise RealDataViolation("game_pk must be positive")
    return game_pk


def validate_source_urls(urls: Iterable[str]) -> None:
    found = False
    for raw_url in urls:
        found = True
        parsed = urlparse(raw_url)
        if parsed.scheme != "https" or parsed.hostname not in APPROVED_HOSTS:
            raise RealDataViolation(f"Unapproved data source: {raw_url}")
    if not found:
        raise RealDataViolation("No public-source provenance was recorded")


def reject_forbidden_markers(value: Any, path: str = "root") -> None:
    """Reject explicit attempts to label a runtime record as fabricated.

    The check intentionally examines keys and source/provenance strings, not human-facing
    documentation that explains the prohibition.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(marker in normalized for marker in FORBIDDEN_MARKERS):
                raise RealDataViolation(f"Forbidden fabricated-data marker at {path}.{key}")
            reject_forbidden_markers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_markers(child, f"{path}[{index}]")
    elif isinstance(value, str) and ("source" in path or "provenance" in path):
        lowered = value.lower()
        if any(marker in lowered for marker in FORBIDDEN_MARKERS):
            raise RealDataViolation(f"Forbidden fabricated-data value at {path}")

