from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from data.schema import GameIdentity
from features.state import (
    BattingLine,
    FeatureDataError,
    LeagueState,
    PitchingLine,
    TeamGame,
    aggregate_batting,
    aggregate_pitching,
    haversine_miles,
    number,
    safe_divide,
)
from utils.time import parse_mlb_date, parse_mlb_datetime

WIND_SPEED = re.compile(r"(-?\d+(?:\.\d+)?)")


def game_identity(
    feed: Mapping[str, Any], *, allow_final_starter_fallback: bool = False
) -> GameIdentity:
    game_data = _mapping(feed, "gameData")
    datetime_data = _mapping(game_data, "datetime")
    teams = _mapping(game_data, "teams")
    home = _mapping(teams, "home")
    away = _mapping(teams, "away")
    venue = _mapping(game_data, "venue")
    game_pk = int(_mapping(game_data, "game")["pk"])
    return GameIdentity(
        game_pk=game_pk,
        game_date=parse_mlb_datetime(str(datetime_data["dateTime"])),
        official_date=str(datetime_data["officialDate"]),
        home_team_id=int(home["id"]),
        away_team_id=int(away["id"]),
        home_team_name=str(home["name"]),
        away_team_name=str(away["name"]),
        venue_id=int(venue["id"]),
        venue_name=str(venue["name"]),
        home_starter_id=_starter_id(feed, home, "home", allow_final_starter_fallback),
        away_starter_id=_starter_id(feed, away, "away", allow_final_starter_fallback),
    )


def venue_coordinates(venue: Mapping[str, Any]) -> tuple[float, float]:
    location = _mapping(venue, "location")
    coordinates = _mapping(location, "defaultCoordinates")
    try:
        return float(coordinates["latitude"]), float(coordinates["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureDataError("Official venue has no usable coordinates") from exc


def build_feature_row(
    *,
    identity: GameIdentity,
    feed: Mapping[str, Any],
    state: LeagueState,
    venue_locations: Mapping[int, tuple[float, float]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rolling_games = int(config["rolling_games"])
    starter_outings = int(config["starter_outings"])
    bullpen_days = int(config["bullpen_days"])
    pythagorean_exponent = float(config["pythagorean_exponent"])
    fip_constant = float(config["fip_constant"])
    woba_weights = _mapping(config, "woba_weights")
    target_date = identity.game_date.date()

    row: dict[str, Any] = {
        "game_pk": identity.game_pk,
        "game_date": identity.game_date.isoformat(),
        "home_team_id": str(identity.home_team_id),
        "away_team_id": str(identity.away_team_id),
        "venue_id": str(identity.venue_id),
    }
    home_games = state.team_histories[identity.home_team_id].rolling(rolling_games)
    away_games = state.team_histories[identity.away_team_id].rolling(rolling_games)
    row.update(
        _team_features(
            "home",
            home_games,
            target_date,
            identity.venue_id,
            venue_locations,
            bullpen_days,
            pythagorean_exponent,
            fip_constant,
            woba_weights,
        )
    )
    row.update(
        _team_features(
            "away",
            away_games,
            target_date,
            identity.venue_id,
            venue_locations,
            bullpen_days,
            pythagorean_exponent,
            fip_constant,
            woba_weights,
        )
    )
    row.update(
        _starter_features(
            "home_starter",
            identity.home_starter_id,
            state,
            starter_outings,
            fip_constant,
        )
    )
    row.update(
        _starter_features(
            "away_starter",
            identity.away_starter_id,
            state,
            starter_outings,
            fip_constant,
        )
    )
    home_win_rate = float(row["home_win_rate"])
    away_win_rate = float(row["away_win_rate"])
    denominator = home_win_rate * (1 - away_win_rate) + (1 - home_win_rate) * away_win_rate
    if denominator <= 0:
        raise FeatureDataError("Log5 is undefined for the observed rolling records")
    row["log5_home_win_probability"] = home_win_rate * (1 - away_win_rate) / denominator
    row["home_field_form_advantage"] = (
        float(row["home_home_role_win_rate"]) - float(row["away_away_role_win_rate"])
    )
    row["pythagorean_strength_difference"] = (
        float(row["home_pythagorean_expectation"])
        - float(row["away_pythagorean_expectation"])
    )
    row["ops_difference"] = float(row["home_ops"]) - float(row["away_ops"])
    row["woba_proxy_difference"] = float(row["home_woba_proxy"]) - float(
        row["away_woba_proxy"]
    )
    row["starter_fip_difference"] = float(row["away_starter_fip_proxy"]) - float(
        row["home_starter_fip_proxy"]
    )
    venue_runs = state.venue_totals[identity.venue_id]
    row["park_run_factor"] = safe_divide(
        sum(venue_runs) / len(venue_runs),
        sum(state.league_totals) / len(state.league_totals),
        "park run factor",
    )
    row.update(_weather_features(feed))
    return row


def add_official_result(
    row: dict[str, Any], feed: Mapping[str, Any]
) -> tuple[int, int]:
    linescore = _mapping(_mapping(feed, "liveData"), "linescore")
    teams = _mapping(linescore, "teams")
    home_runs = int(_mapping(teams, "home")["runs"])
    away_runs = int(_mapping(teams, "away")["runs"])
    if home_runs == away_runs:
        raise FeatureDataError("A final MLB game cannot have a tied score")
    row.update(
        {
            "home_win": int(home_runs > away_runs),
            "home_runs": home_runs,
            "away_runs": away_runs,
            "run_diff": home_runs - away_runs,
            "total_runs": home_runs + away_runs,
        }
    )
    return home_runs, away_runs


def update_state_from_final(
    *, state: LeagueState, identity: GameIdentity, feed: Mapping[str, Any]
) -> None:
    live_data = _mapping(feed, "liveData")
    linescore = _mapping(live_data, "linescore")
    score_teams = _mapping(linescore, "teams")
    home_runs = int(_mapping(score_teams, "home")["runs"])
    away_runs = int(_mapping(score_teams, "away")["runs"])
    boxscore = _mapping(live_data, "boxscore")
    box_teams = _mapping(boxscore, "teams")
    home = _mapping(box_teams, "home")
    away = _mapping(box_teams, "away")
    home_team_stats = _mapping(home, "teamStats")
    away_team_stats = _mapping(away, "teamStats")
    state.update_game(
        game_date=parse_mlb_date(identity.official_date),
        venue_id=identity.venue_id,
        home_team_id=identity.home_team_id,
        away_team_id=identity.away_team_id,
        home_runs=home_runs,
        away_runs=away_runs,
        home_batting=BattingLine.from_stats(_mapping(home_team_stats, "batting")),
        away_batting=BattingLine.from_stats(_mapping(away_team_stats, "batting")),
        home_pitching=PitchingLine.from_stats(_mapping(home_team_stats, "pitching")),
        away_pitching=PitchingLine.from_stats(_mapping(away_team_stats, "pitching")),
        home_pitchers=_pitcher_lines(home),
        away_pitchers=_pitcher_lines(away),
    )


def _team_features(
    prefix: str,
    games: list[TeamGame],
    target_date: date,
    target_venue_id: int,
    venue_locations: Mapping[int, tuple[float, float]],
    bullpen_days: int,
    pythagorean_exponent: float,
    fip_constant: float,
    woba_weights: Mapping[str, Any],
) -> dict[str, float]:
    if not games:
        raise FeatureDataError(f"No official history for {prefix} team")
    batting = aggregate_batting(games)
    pitching = aggregate_pitching(games)
    runs_for = float(sum(game.runs_for for game in games))
    runs_against = float(sum(game.runs_against for game in games))
    wins = float(sum(game.won for game in games))
    home_role = [game for game in games if game.was_home]
    away_role = [game for game in games if not game.was_home]
    if not home_role or not away_role:
        raise FeatureDataError(f"Insufficient home/away role history for {prefix} team")
    singles = batting.hits - batting.doubles - batting.triples - batting.home_runs
    if singles < 0:
        raise FeatureDataError("Official batting components produced negative singles")
    obp_denominator = (
        batting.at_bats + batting.walks + batting.hit_by_pitch + batting.sacrifice_flies
    )
    total_bases = (
        singles + 2 * batting.doubles + 3 * batting.triples + 4 * batting.home_runs
    )
    woba_denominator = (
        batting.at_bats
        + batting.walks
        - batting.intentional_walks
        + batting.sacrifice_flies
        + batting.hit_by_pitch
    )
    woba_numerator = (
        float(woba_weights["walk"]) * (batting.walks - batting.intentional_walks)
        + float(woba_weights["hit_by_pitch"]) * batting.hit_by_pitch
        + float(woba_weights["single"]) * singles
        + float(woba_weights["double"]) * batting.doubles
        + float(woba_weights["triple"]) * batting.triples
        + float(woba_weights["home_run"]) * batting.home_runs
    )
    run_denominator = runs_for**pythagorean_exponent + runs_against**pythagorean_exponent
    if run_denominator <= 0:
        raise FeatureDataError("Pythagorean expectation has no observed runs")
    innings = pitching.outs / 3
    fip = safe_divide(
        13 * pitching.home_runs
        + 3 * (pitching.walks + pitching.hit_by_pitch)
        - 2 * pitching.strikeouts,
        innings,
        "team FIP proxy",
    ) + fip_constant
    cutoff_ordinal = target_date.toordinal() - bullpen_days
    recent = [game for game in games if game.game_date.toordinal() >= cutoff_ordinal]
    usage_dates: dict[int, list[date]] = {}
    for game in recent:
        for pitcher_id in game.bullpen_pitcher_ids:
            usage_dates.setdefault(pitcher_id, []).append(game.game_date)
    back_to_back = 0
    for dates in usage_dates.values():
        ordinals = sorted({item.toordinal() for item in dates})
        back_to_back += sum(
            right - left == 1 for left, right in zip(ordinals, ordinals[1:], strict=False)
        )
    previous = games[-1]
    rest_days = max((target_date - previous.game_date).days - 1, 0)
    origin = venue_locations.get(previous.venue_id)
    destination = venue_locations.get(target_venue_id)
    if origin is None or destination is None:
        raise FeatureDataError("Official venue coordinates required for travel proxy are absent")
    travel = haversine_miles(origin, destination)
    obp = safe_divide(
        batting.hits + batting.walks + batting.hit_by_pitch,
        obp_denominator,
        "OBP",
    )
    slg = safe_divide(total_bases, batting.at_bats, "SLG")
    return {
        f"{prefix}_win_rate": wins / len(games),
        f"{prefix}_home_role_win_rate": sum(game.won for game in home_role) / len(home_role),
        f"{prefix}_away_role_win_rate": sum(game.won for game in away_role) / len(away_role),
        f"{prefix}_runs_per_game": runs_for / len(games),
        f"{prefix}_runs_allowed_per_game": runs_against / len(games),
        f"{prefix}_pythagorean_expectation": runs_for**pythagorean_exponent / run_denominator,
        f"{prefix}_obp": obp,
        f"{prefix}_slg": slg,
        f"{prefix}_ops": obp + slg,
        f"{prefix}_woba_proxy": safe_divide(woba_numerator, woba_denominator, "wOBA proxy"),
        f"{prefix}_strikeout_rate": safe_divide(
            batting.strikeouts, obp_denominator, "offensive strikeout rate"
        ),
        f"{prefix}_pitching_fip_proxy": fip,
        f"{prefix}_bullpen_pitches_recent": sum(game.bullpen_pitches for game in recent),
        f"{prefix}_bullpen_appearances_recent": float(
            sum(len(game.bullpen_pitcher_ids) for game in recent)
        ),
        f"{prefix}_bullpen_back_to_back_recent": float(back_to_back),
        f"{prefix}_rest_days": float(rest_days),
        f"{prefix}_travel_miles": travel,
    }


def _starter_features(
    prefix: str,
    pitcher_id: int,
    state: LeagueState,
    count: int,
    fip_constant: float,
) -> dict[str, float]:
    appearances = [row for row in state.pitcher_histories[pitcher_id] if row.started][-count:]
    if not appearances:
        raise FeatureDataError(f"No prior official starts for pitcher_id={pitcher_id}")
    outs = sum(row.line.outs for row in appearances)
    innings = outs / 3
    batters = sum(row.line.batters_faced for row in appearances)
    strikeouts = sum(row.line.strikeouts for row in appearances)
    walks = sum(row.line.walks for row in appearances)
    hit_by_pitch = sum(row.line.hit_by_pitch for row in appearances)
    home_runs = sum(row.line.home_runs for row in appearances)
    pitches = sum(row.line.pitches for row in appearances)
    balls_in_play_outs = max(outs - strikeouts, 0)
    return {
        f"{prefix}_innings_per_start": safe_divide(innings, len(appearances), "starter innings"),
        f"{prefix}_strikeout_rate": safe_divide(strikeouts, batters, "starter K rate"),
        f"{prefix}_walk_rate": safe_divide(walks, batters, "starter BB rate"),
        f"{prefix}_contact_out_rate_proxy": safe_divide(
            balls_in_play_outs, batters, "starter contact-out proxy"
        ),
        f"{prefix}_pitches_per_start": safe_divide(pitches, len(appearances), "starter pitches"),
        f"{prefix}_fip_proxy": safe_divide(
            13 * home_runs + 3 * (walks + hit_by_pitch) - 2 * strikeouts,
            innings,
            "starter FIP proxy",
        )
        + fip_constant,
    }


def _weather_features(feed: Mapping[str, Any]) -> dict[str, float]:
    game_data = _mapping(feed, "gameData")
    weather = game_data.get("weather")
    if not isinstance(weather, Mapping):
        return {"weather_temperature_f": math.nan, "weather_wind_mph": math.nan}
    temperature = weather.get("temp")
    wind = weather.get("wind")
    wind_match = WIND_SPEED.search(str(wind)) if wind is not None else None
    return {
        "weather_temperature_f": float(temperature) if temperature is not None else math.nan,
        "weather_wind_mph": float(wind_match.group(1)) if wind_match else math.nan,
    }


def _pitcher_lines(team_boxscore: Mapping[str, Any]) -> list[tuple[int, PitchingLine, bool]]:
    players = _mapping(team_boxscore, "players")
    result: list[tuple[int, PitchingLine, bool]] = []
    for player in players.values():
        if not isinstance(player, Mapping):
            continue
        person = player.get("person")
        stats = player.get("stats")
        if not isinstance(person, Mapping) or not isinstance(stats, Mapping):
            continue
        pitching = stats.get("pitching")
        if not isinstance(pitching, Mapping) or not pitching:
            continue
        pitcher_id = int(person["id"])
        started = int(number(pitching.get("gamesStarted"))) == 1
        result.append((pitcher_id, PitchingLine.from_stats(pitching), started))
    if not result or not any(started for _pitcher_id, _line, started in result):
        raise FeatureDataError("Official boxscore contains no starting pitcher")
    return result


def _probable_pitcher_id(team: Mapping[str, Any], side: str) -> int:
    pitcher = team.get("probablePitcher")
    if not isinstance(pitcher, Mapping) or not isinstance(pitcher.get("id"), int):
        raise FeatureDataError(f"Official feed has no announced {side} probable pitcher")
    return int(pitcher["id"])


def _starter_id(
    feed: Mapping[str, Any],
    team: Mapping[str, Any],
    side: str,
    allow_final_starter_fallback: bool,
) -> int:
    try:
        return _probable_pitcher_id(team, side)
    except FeatureDataError as exc:
        game_data = _mapping(feed, "gameData")
        probable_pitchers = game_data.get("probablePitchers")
        if isinstance(probable_pitchers, Mapping):
            pitcher = probable_pitchers.get(side)
            if isinstance(pitcher, Mapping) and isinstance(pitcher.get("id"), int):
                return int(pitcher["id"])
        if not allow_final_starter_fallback:
            raise FeatureDataError(
                f"Official feed has no announced {side} probable pitcher"
            ) from exc
    live_data = _mapping(feed, "liveData")
    boxscore = _mapping(live_data, "boxscore")
    box_team = _mapping(_mapping(boxscore, "teams"), side)
    for pitcher_id, _line, started in _pitcher_lines(box_team):
        if started:
            return pitcher_id
    raise FeatureDataError(f"Official final feed has no {side} starter")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise FeatureDataError(f"Official MLB response is missing object: {key}")
    return child
