"""Pruebas del parser de TIME_PERIOD (mensual + quincenal)."""
from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from inflacion.inegi.client import BIEClient, BIEError, _parse_time_period

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"


def test_parse_time_period_monthly():
    assert _parse_time_period("2024/03") == date(2024, 3, 31)
    assert _parse_time_period("2020/02") == date(2020, 2, 29)  # bisiesto


def test_parse_time_period_quincenal_q1():
    assert _parse_time_period("2024/03/Q1") == date(2024, 3, 15)
    assert _parse_time_period("2024/03/1") == date(2024, 3, 15)
    assert _parse_time_period("2024/03/01") == date(2024, 3, 15)


def test_parse_time_period_quincenal_q2():
    assert _parse_time_period("2024/03/Q2") == date(2024, 3, 31)
    assert _parse_time_period("2024/04/Q2") == date(2024, 4, 30)
    assert _parse_time_period("2024/03/2") == date(2024, 3, 31)


def test_parse_time_period_unknown_format_raises():
    with pytest.raises(BIEError):
        _parse_time_period("2024/03/X9")
    with pytest.raises(BIEError):
        _parse_time_period("2024")
    with pytest.raises(BIEError):
        _parse_time_period("2024/03/Q3")


@respx.mock
def test_fetch_series_handles_quincenal_payload():
    """Una respuesta del BIE con quincenales debe regresar 2 filas por mes."""
    payload = {
        "Series": [
            {
                "OBSERVATIONS": [
                    {"TIME_PERIOD": "2024/03/Q1", "OBS_VALUE": "133.500"},
                    {"TIME_PERIOD": "2024/03/Q2", "OBS_VALUE": "134.000"},
                    {"TIME_PERIOD": "2024/04/Q1", "OBS_VALUE": "134.250"},
                    {"TIME_PERIOD": "2024/04/Q2", "OBS_VALUE": "134.500"},
                ]
            }
        ]
    }
    url = f"{BASE}/INDICATOR/999/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(200, json=payload))

    with BIEClient(token=TOKEN) as client:
        df = client.fetch_series("999")

    assert df.shape == (4, 1)
    # Fechas Q1=día 15, Q2=fin de mes — y NO se colapsan en drop_duplicates
    assert df.index[0].day == 15  # 2024/03/Q1
    assert df.index[1].day == 31  # 2024/03/Q2
    assert df.index[2].day == 15  # 2024/04/Q1
    assert df.index[3].day == 30  # 2024/04/Q2 (abril)
