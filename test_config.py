from pathlib import Path

from utils.config import load_settings


def test_default_configuration_is_valid() -> None:
    settings = load_settings(Path("config/default.yml"))
    assert settings.section("http")["base_url"] == "https://statsapi.mlb.com"
    assert settings.section("training")["walk_forward_folds"] >= 2
    assert settings.section("markets")["maximum_runs_for_distribution"] >= 15

