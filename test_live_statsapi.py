import os

import pytest

from api.mlb_stats import MLBStatsAPI
from utils.config import load_settings


@pytest.mark.live
def test_user_selected_real_game_contract() -> None:
    game_pk = os.getenv("MLB_TEST_GAME_PK")
    if not game_pk:
        pytest.skip("Set MLB_TEST_GAME_PK to opt into the official live integration test")
    settings = load_settings()
    client = MLBStatsAPI.from_config(settings.section("http"))
    feed = client.live_feed(int(game_pk), fresh=True)
    assert feed["gameData"]["game"]["pk"] == int(game_pk)
    assert feed["gameData"]["teams"]["home"]["sport"]["id"] == 1
    assert feed["gameData"]["teams"]["away"]["sport"]["id"] == 1
