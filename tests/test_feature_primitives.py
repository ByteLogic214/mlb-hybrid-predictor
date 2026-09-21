import pytest

from features.state import FeatureDataError, innings_to_outs


@pytest.mark.parametrize(
    ("official_value", "expected_outs"), [("0.0", 0), ("5.1", 16), ("7.2", 23)]
)
def test_official_innings_notation_to_outs(official_value: str, expected_outs: int) -> None:
    assert innings_to_outs(official_value) == expected_outs


def test_impossible_official_innings_notation_is_rejected() -> None:
    with pytest.raises(FeatureDataError):
        innings_to_outs("4.3")
