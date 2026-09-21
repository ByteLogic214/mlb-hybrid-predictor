from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any


class FeatureDataError(RuntimeError):
    """A required real observation is absent or malformed."""


def number(value: Any, *, default: float = 0.0) -> float:
    if value in (None, "", "-"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureDataError(f"Expected numeric MLB statistic, received {value!r}") from exc


def innings_to_outs(value: Any) -> int:
    text = str(value or "0.0")
    if "." not in text:
        return int(text) * 3
    whole_text, remainder_text = text.split(".", maxsplit=1)
    whole = int(whole_text)
    remainder = int(remainder_text)
    if remainder not in (0, 1, 2):
        raise FeatureDataError(f"Invalid official inningsPitched value: {value!r}")
    return whole * 3 + remainder


@dataclass(frozen=True)
class BattingLine:
    at_bats: float
    hits: float
    doubles: float
    triples: float
    home_runs: float
    walks: float
    intentional_walks: float
    hit_by_pitch: float
    sacrifice_flies: float
    strikeouts: float

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any]) -> BattingLine:
        return cls(
            at_bats=number(stats.get("atBats")),
            hits=number(stats.get("hits")),
            doubles=number(stats.get("doubles")),
            triples=number(stats.get("triples")),
            home_runs=number(stats.get("homeRuns")),
            walks=number(stats.get("baseOnBalls")),
            intentional_walks=number(stats.get("intentionalWalks")),
            hit_by_pitch=number(stats.get("hitByPitch")),
            sacrifice_flies=number(stats.get("sacFlies")),
            strikeouts=number(stats.get("strikeOuts")),
        )


@dataclass(frozen=True)
class PitchingLine:
    outs: int
    batters_faced: float
    strikeouts: float
    walks: float
    hit_by_pitch: float
    home_runs: float
    pitches: float

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any]) -> PitchingLine:
        return cls(
            outs=innings_to_outs(stats.get("inningsPitched")),
            batters_faced=number(stats.get("battersFaced")),
            strikeouts=number(stats.get("strikeOuts")),
            walks=number(stats.get("baseOnBalls")),
            hit_by_pitch=number(stats.get("hitByPitch")),
            home_runs=number(stats.get("homeRuns")),
            pitches=number(
                stats.get("numberOfPitches"), default=number(stats.get("pitchesThrown"))
            ),
        )


@dataclass(frozen=True)
class PitcherAppearance:
    game_date: date
    line: PitchingLine
    started: bool


@dataclass(frozen=True)
class TeamGame:
    game_date: date
    venue_id: int
    was_home: bool
    won: bool
    runs_for: int
    runs_against: int
    batting: BattingLine
    pitching: PitchingLine
    bullpen_pitches: float
    bullpen_pitcher_ids: tuple[int, ...]


@dataclass
class TeamHistory:
    games: deque[TeamGame] = field(default_factory=lambda: deque(maxlen=200))

    def add(self, game: TeamGame) -> None:
        self.games.append(game)

    def rolling(self, count: int) -> list[TeamGame]:
        return list(self.games)[-count:]


@dataclass
class LeagueState:
    team_histories: dict[int, TeamHistory] = field(default_factory=lambda: defaultdict(TeamHistory))
    pitcher_histories: dict[int, deque[PitcherAppearance]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=50))
    )
    venue_totals: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    league_totals: list[int] = field(default_factory=list)

    def ready(
        self,
        home_team_id: int,
        away_team_id: int,
        home_starter_id: int,
        away_starter_id: int,
        venue_id: int,
        minimum_team_games: int,
        minimum_starter_outings: int,
        minimum_venue_games: int,
    ) -> bool:
        return (
            len(self.team_histories[home_team_id].games) >= minimum_team_games
            and len(self.team_histories[away_team_id].games) >= minimum_team_games
            and len(self.pitcher_histories[home_starter_id]) >= minimum_starter_outings
            and len(self.pitcher_histories[away_starter_id]) >= minimum_starter_outings
            and len(self.venue_totals[venue_id]) >= minimum_venue_games
            and len(self.league_totals) >= minimum_venue_games * 10
        )

    def update_game(
        self,
        *,
        game_date: date,
        venue_id: int,
        home_team_id: int,
        away_team_id: int,
        home_runs: int,
        away_runs: int,
        home_batting: BattingLine,
        away_batting: BattingLine,
        home_pitching: PitchingLine,
        away_pitching: PitchingLine,
        home_pitchers: Iterable[tuple[int, PitchingLine, bool]],
        away_pitchers: Iterable[tuple[int, PitchingLine, bool]],
    ) -> None:
        home_pitcher_rows = list(home_pitchers)
        away_pitcher_rows = list(away_pitchers)
        home_bullpen = tuple(pid for pid, _line, started in home_pitcher_rows if not started)
        away_bullpen = tuple(pid for pid, _line, started in away_pitcher_rows if not started)
        home_bullpen_pitches = sum(
            line.pitches for _pid, line, started in home_pitcher_rows if not started
        )
        away_bullpen_pitches = sum(
            line.pitches for _pid, line, started in away_pitcher_rows if not started
        )
        self.team_histories[home_team_id].add(
            TeamGame(
                game_date=game_date,
                venue_id=venue_id,
                was_home=True,
                won=home_runs > away_runs,
                runs_for=home_runs,
                runs_against=away_runs,
                batting=home_batting,
                pitching=home_pitching,
                bullpen_pitches=home_bullpen_pitches,
                bullpen_pitcher_ids=home_bullpen,
            )
        )
        self.team_histories[away_team_id].add(
            TeamGame(
                game_date=game_date,
                venue_id=venue_id,
                was_home=False,
                won=away_runs > home_runs,
                runs_for=away_runs,
                runs_against=home_runs,
                batting=away_batting,
                pitching=away_pitching,
                bullpen_pitches=away_bullpen_pitches,
                bullpen_pitcher_ids=away_bullpen,
            )
        )
        for pitcher_id, line, started in home_pitcher_rows + away_pitcher_rows:
            self.pitcher_histories[pitcher_id].append(
                PitcherAppearance(game_date=game_date, line=line, started=started)
            )
        total = home_runs + away_runs
        self.venue_totals[venue_id].append(total)
        self.league_totals.append(total)


def aggregate_batting(games: Iterable[TeamGame]) -> BattingLine:
    rows = list(games)
    return BattingLine(
        at_bats=sum(row.batting.at_bats for row in rows),
        hits=sum(row.batting.hits for row in rows),
        doubles=sum(row.batting.doubles for row in rows),
        triples=sum(row.batting.triples for row in rows),
        home_runs=sum(row.batting.home_runs for row in rows),
        walks=sum(row.batting.walks for row in rows),
        intentional_walks=sum(row.batting.intentional_walks for row in rows),
        hit_by_pitch=sum(row.batting.hit_by_pitch for row in rows),
        sacrifice_flies=sum(row.batting.sacrifice_flies for row in rows),
        strikeouts=sum(row.batting.strikeouts for row in rows),
    )


def aggregate_pitching(games: Iterable[TeamGame]) -> PitchingLine:
    rows = list(games)
    return PitchingLine(
        outs=sum(row.pitching.outs for row in rows),
        batters_faced=sum(row.pitching.batters_faced for row in rows),
        strikeouts=sum(row.pitching.strikeouts for row in rows),
        walks=sum(row.pitching.walks for row in rows),
        hit_by_pitch=sum(row.pitching.hit_by_pitch for row in rows),
        home_runs=sum(row.pitching.home_runs for row in rows),
        pitches=sum(row.pitching.pitches for row in rows),
    )


def safe_divide(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise FeatureDataError(f"Cannot derive {label}: official denominator is zero")
    return numerator / denominator


def haversine_miles(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, origin)
    lat2, lon2 = map(math.radians, destination)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 3958.7613 * 2 * math.asin(math.sqrt(value))
