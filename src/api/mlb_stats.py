from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.logging import log_event
from utils.real_data import require_positive_game_pk
from utils.time import parse_mlb_datetime, utc_now

LOGGER = logging.getLogger(__name__)


class MLBAPIError(RuntimeError):
    """Official MLB StatsAPI request or schema failure."""


@dataclass(frozen=True)
class ProvenanceRecord:
    url: str
    fetched_at: str
    sha256: str
    cache_hit: bool


class MLBStatsAPI:
    """Rate-limited, retrying client for the official public MLB StatsAPI."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int,
        retries: int,
        requests_per_second: float,
        cache_directory: str | Path,
    ) -> None:
        if base_url.rstrip("/") != "https://statsapi.mlb.com":
            raise MLBAPIError("Only the official https://statsapi.mlb.com source is permitted")
        if requests_per_second <= 0:
            raise MLBAPIError("requests_per_second must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.minimum_interval = 1.0 / requests_per_second
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self._lock = threading.Lock()
        self.provenance: list[ProvenanceRecord] = []
        self.session = requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.7,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "mlb-hybrid-predictor/1.0"}
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MLBStatsAPI:
        return cls(
            base_url=str(config["base_url"]),
            timeout_seconds=int(config["timeout_seconds"]),
            retries=int(config["retries"]),
            requests_per_second=float(config["requests_per_second"]),
            cache_directory=str(config["cache_directory"]),
        )

    def _get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        query = urlencode(sorted(clean_params.items()), doseq=True)
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = self.cache_directory / f"{cache_key}.json"
        if use_cache and cache_path.is_file():
            payload_bytes = cache_path.read_bytes()
            payload = json.loads(payload_bytes)
            self._record(url, payload_bytes, cache_hit=True)
            return _require_object(payload, url)

        with self._lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                self._last_request = time.monotonic()
                response.raise_for_status()
            except requests.RequestException as exc:
                raise MLBAPIError(f"Official MLB request failed: {url}: {exc}") from exc
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise MLBAPIError(f"Official MLB response was not JSON: {url}") from exc
        object_payload = _require_object(payload, url)
        encoded = json.dumps(object_payload, sort_keys=True, separators=(",", ":")).encode()
        cache_path.write_bytes(encoded)
        self._record(url, encoded, cache_hit=False)
        log_event(LOGGER, logging.INFO, "mlb_api_fetch", url=url, bytes=len(encoded))
        return object_payload

    def _record(self, url: str, payload: bytes, cache_hit: bool) -> None:
        self.provenance.append(
            ProvenanceRecord(
                url=url,
                fetched_at=utc_now().isoformat(),
                sha256=hashlib.sha256(payload).hexdigest(),
                cache_hit=cache_hit,
            )
        )

    def live_feed(self, game_pk: int, *, fresh: bool = False) -> dict[str, Any]:
        game_pk = require_positive_game_pk(game_pk)
        return self._get(f"/api/v1.1/game/{game_pk}/feed/live", use_cache=not fresh)

    def schedule(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        if end_date < start_date:
            raise MLBAPIError("Schedule end_date must not precede start_date")
        games_by_id: dict[int, dict[str, Any]] = {}
        cursor = start_date
        while cursor <= end_date:
            chunk_end = min(cursor + timedelta(days=29), end_date)
            payload = self._get(
                "/api/v1/schedule",
                {
                    "sportId": 1,
                    "startDate": cursor.isoformat(),
                    "endDate": chunk_end.isoformat(),
                },
            )
            for day in payload.get("dates", []):
                if isinstance(day, Mapping):
                    for game in day.get("games", []):
                        if isinstance(game, dict) and isinstance(game.get("gamePk"), int):
                            games_by_id[int(game["gamePk"])] = game
            cursor = chunk_end + timedelta(days=1)
        return list(games_by_id.values())

    def venue(self, venue_id: int) -> dict[str, Any]:
        payload = self._get(f"/api/v1/venues/{int(venue_id)}")
        venues = payload.get("venues")
        if not isinstance(venues, list) or not venues or not isinstance(venues[0], dict):
            raise MLBAPIError(f"No official venue record for venue_id={venue_id}")
        return venues[0]

    def validate_pregame(self, game_pk: int) -> dict[str, Any]:
        feed = self.live_feed(game_pk, fresh=True)
        game_data = _mapping(feed, "gameData")
        status = _mapping(game_data, "status")
        abstract = str(status.get("abstractGameState", ""))
        if abstract != "Preview":
            raise MLBAPIError(
                f"game_pk={game_pk} is not pregame; official state is {abstract or 'unknown'}"
            )
        scheduled = parse_mlb_datetime(str(game_data.get("datetime", {}).get("dateTime", "")))
        if utc_now() >= scheduled:
            raise MLBAPIError("Scheduled start has passed; a leakage-safe pregame run is refused")
        teams = _mapping(game_data, "teams")
        for side in ("home", "away"):
            team = _mapping(teams, side)
            if not isinstance(team.get("id"), int):
                raise MLBAPIError(f"Official feed has no {side} team id")
            sport = _mapping(team, "sport")
            if int(sport.get("id", 0)) != 1:
                raise MLBAPIError(f"game_pk={game_pk} is not a Major League Baseball game")
        return feed

    def wait_until_final(
        self, game_pk: int, poll_seconds: int, maximum_wait_minutes: int
    ) -> dict[str, Any]:
        deadline = time.monotonic() + maximum_wait_minutes * 60
        while True:
            feed = self.live_feed(game_pk, fresh=True)
            status = _mapping(_mapping(feed, "gameData"), "status")
            abstract = str(status.get("abstractGameState", ""))
            detailed = str(status.get("detailedState", ""))
            log_event(
                LOGGER,
                logging.INFO,
                "game_status",
                game_pk=game_pk,
                abstract=abstract,
                detailed=detailed,
            )
            if abstract == "Final" or detailed in {"Final", "Game Over", "Completed Early"}:
                return feed
            if detailed in {"Cancelled", "Postponed", "Suspended"}:
                raise MLBAPIError(f"Game cannot be backtested in state: {detailed}")
            if time.monotonic() >= deadline:
                raise MLBAPIError(
                    f"Official final state not reached within {maximum_wait_minutes} minutes"
                )
            time.sleep(poll_seconds)

    def write_provenance(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        unique: dict[tuple[str, str], ProvenanceRecord] = {}
        for record in self.provenance:
            unique[(record.url, record.sha256)] = record
        payload = {
            "generated_at": utc_now().isoformat(),
            "source_policy": "official-public-only",
            "records": [asdict(record) for record in unique.values()],
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise MLBAPIError(f"Official MLB response is missing object: {key}")
    return child


def _require_object(value: Any, url: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MLBAPIError(f"Official MLB response root is not an object: {url}")
    return value
