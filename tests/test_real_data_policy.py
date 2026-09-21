import pytest

from utils.real_data import RealDataViolation, validate_source_urls


def test_official_statsapi_https_source_is_accepted() -> None:
    validate_source_urls(["https://statsapi.mlb.com/api/v1/schedule?sportId=1"])


def test_nonofficial_host_is_rejected() -> None:
    with pytest.raises(RealDataViolation):
        validate_source_urls(["https://example.invalid/baseball"])

