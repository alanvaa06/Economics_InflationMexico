"""Pruebas del cliente BIE con respx (mock httpx)."""
from __future__ import annotations

import httpx
import pytest
import respx

from inflacion.inegi.client import BIEClient, BIEError, MissingTokenError, _redact

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"


def _ok_payload() -> dict:
    return {
        "Series": [
            {
                "OBSERVATIONS": [
                    {"TIME_PERIOD": "2024/01", "OBS_VALUE": "133.555"},
                    {"TIME_PERIOD": "2024/02", "OBS_VALUE": "133.681"},
                    {"TIME_PERIOD": "2024/03", "OBS_VALUE": ""},
                ]
            }
        ]
    }


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("INEGI_API_TOKEN", raising=False)
    with pytest.raises(MissingTokenError):
        BIEClient(token="")


@respx.mock
def test_fetch_series_parses_observations():
    url = f"{BASE}/INDICATOR/583766/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(200, json=_ok_payload()))

    with BIEClient(token=TOKEN) as client:
        df = client.fetch_series("583766")

    assert df.shape == (3, 1)
    assert df.columns.tolist() == ["valor"]
    assert df.index.is_monotonic_increasing
    # último día del mes
    assert df.index[0].day == 31
    # NaN preserved
    assert df["valor"].isna().sum() == 1


@respx.mock
def test_fetch_many_keeps_only_successful_series():
    ok_url = f"{BASE}/INDICATOR/1/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    bad_url = f"{BASE}/INDICATOR/2/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(ok_url).mock(return_value=httpx.Response(200, json=_ok_payload()))
    respx.get(bad_url).mock(return_value=httpx.Response(403, text="Host not in allowlist"))

    with BIEClient(token=TOKEN) as client:
        df = client.fetch_many([1, 2])

    # solo la serie OK debe sobrevivir
    assert df.columns.tolist() == ["1"]


@respx.mock
def test_403_does_not_retry():
    url = f"{BASE}/INDICATOR/x/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    route = respx.get(url).mock(return_value=httpx.Response(403, text="Host not in allowlist"))

    with BIEClient(token=TOKEN) as client, pytest.raises(BIEError):
        client.fetch_series("x")

    assert route.call_count == 1  # sin reintentos


@respx.mock
def test_500_retries_then_succeeds():
    url = f"{BASE}/INDICATOR/x/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json=_ok_payload()),
        ]
    )
    with BIEClient(token=TOKEN) as client:
        df = client.fetch_series("x")
    assert route.call_count == 3
    assert not df.empty


def test_redact_uuid_in_logs():
    msg = "fallo al pedir https://example/INDICATOR/1/.../BIE/2.0/REDACTED-INEGI-TOKEN-ROTATED?type=json"
    redacted = _redact(msg)
    assert "4f988b8a" not in redacted
    assert "REDACTED" in redacted
