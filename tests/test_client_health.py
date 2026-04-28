"""BIEClient.health_check: valida token contra ID maestro conocido."""
from __future__ import annotations

import httpx
import pytest
import respx

from inflacion.inegi.client import BIEClient, BIEError, MissingTokenError

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"
HEALTH_ID = "583766"  # IndiceGeneral mensual; mismo valor usado en client.py


def _ok_payload() -> dict:
    return {
        "Series": [
            {"OBSERVATIONS": [{"TIME_PERIOD": "2024/01", "OBS_VALUE": "133.555"}]}
        ]
    }


@respx.mock
def test_health_check_ok_when_master_id_responds_200():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(200, json=_ok_payload()))
    with BIEClient(token=TOKEN) as client:
        client.health_check()  # no debe levantar


@respx.mock
def test_health_check_400_with_no_se_encontraron_raises_missing_token():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    body = '["ErrorInfo:No se encontraron resultados","ErrorDetails:No se encontraron resultados","ErrorCode:100"]'
    respx.get(url).mock(return_value=httpx.Response(400, text=body))
    with BIEClient(token=TOKEN) as client, pytest.raises(MissingTokenError, match="rechazó el token"):
        client.health_check()


@respx.mock
def test_health_check_500_raises_bie_error():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(500, text="server error"))
    with BIEClient(token=TOKEN) as client, pytest.raises(BIEError):
        client.health_check()


@respx.mock
def test_health_check_400_with_other_body_raises_bie_error():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(400, text="Some other 400"))
    with BIEClient(token=TOKEN) as client, pytest.raises(BIEError):
        client.health_check()
