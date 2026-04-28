"""classify_period: clasifica un payload BIE por formato TIME_PERIOD."""
from __future__ import annotations

import pytest

from inflacion.inegi.client import classify_period


def _payload(time_period: str) -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": time_period, "OBS_VALUE": "1.0"}]}]}


@pytest.mark.parametrize(
    "tp,expected",
    [
        ("2024/03", "mensual"),
        ("2026/04", "mensual"),
        ("2024/03/Q1", "quincenal"),
        ("2024/03/Q2", "quincenal"),
        ("2024/03/1", "quincenal"),
        ("2024/03/2", "quincenal"),
        ("", "desconocido"),
        ("2024", "desconocido"),
        ("2024/03/Q3", "desconocido"),
    ],
)
def test_classify_period(tp, expected):
    assert classify_period(_payload(tp)) == expected


def test_classify_period_empty_observations_is_desconocido():
    assert classify_period({"Series": [{"OBSERVATIONS": []}]}) == "desconocido"


def test_classify_period_malformed_payload_is_desconocido():
    assert classify_period({}) == "desconocido"
    assert classify_period({"Series": []}) == "desconocido"
