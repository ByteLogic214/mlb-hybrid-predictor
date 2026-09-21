from __future__ import annotations

from datetime import UTC, date, datetime


def parse_mlb_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_mlb_date(value: str) -> date:
    return date.fromisoformat(value)


def utc_now() -> datetime:
    return datetime.now(UTC)

