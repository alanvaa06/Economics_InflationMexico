"""Pruebas del pipeline quincenal."""
from __future__ import annotations

import httpx
import pytest
import respx

from inflacion.data import pipeline
from inflacion.inegi.client import BIEClient

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"


def _q_payload() -> dict:
    return {
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


def test_refresh_quincenal_empty_catalog_raises(monkeypatch):
    """Si todos los IDs son placeholders vacíos, refresh debe fallar con un mensaje claro."""
    monkeypatch.setattr(
        "inflacion.data.pipeline.load_quincenal_catalog",
        lambda: __import__("pandas").Series(dtype=object),
    )
    with pytest.raises(RuntimeError, match="Catálogo quincenal vacío"):
        pipeline.refresh_inpc_quincenal(client=BIEClient(token=TOKEN))


@respx.mock
def test_refresh_quincenal_partial_success(monkeypatch, tmp_path):
    """Si una serie del catálogo responde 404, el resto debe sobrevivir."""
    fake_catalog = __import__("pandas").Series(
        {"IndiceGeneral": "111", "Subyacente": "222"}, name="Serie"
    )
    monkeypatch.setattr("inflacion.data.pipeline.load_quincenal_catalog", lambda: fake_catalog)

    respx.get(f"{BASE}/INDICATOR/111/es/0700/true/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    respx.get(f"{BASE}/INDICATOR/222/es/0700/true/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(403, text="Host not in allowlist")
    )

    out = tmp_path / "q.parquet"
    df = pipeline.refresh_inpc_quincenal(
        client=BIEClient(token=TOKEN), historic=True, out_path=out
    )
    # solo IndiceGeneral debe sobrevivir, renombrado al nombre amigable
    assert list(df.columns) == ["IndiceGeneral"]
    assert df.shape[0] == 4  # 4 quincenas
    assert out.exists()


def test_quincenal_candidates_has_expected_keys():
    """Los nombres canónicos de cabecera están en CANDIDATES (los IDs concretos los resuelve el resolver)."""
    from inflacion.inegi.series_quincenal import QUINCENAL_HEADLINE_CANDIDATES

    expected = {
        "IndiceGeneral",
        "Subyacente",
        "NoSubyacente",
        "Mercancías",
        "Servicios",
    }
    assert expected.issubset(QUINCENAL_HEADLINE_CANDIDATES.keys())
    # Cada candidato debe ser una lista no-vacía de strings
    for name, ids in QUINCENAL_HEADLINE_CANDIDATES.items():
        assert isinstance(ids, list) and ids, f"{name} sin candidatos"
        assert all(isinstance(i, str) and i for i in ids), f"{name} con candidatos vacíos"


@respx.mock
def test_refresh_quincenal_runs_resolver_when_catalog_empty(monkeypatch, tmp_path):
    """Si load_quincenal_catalog devuelve Series vacía, refresh corre el resolver."""
    import pandas as pd

    from inflacion.data.pipeline import refresh_inpc_quincenal_with_discovery
    from inflacion.inegi.client import BIEClient

    monkeypatch.setattr(
        "inflacion.data.pipeline.load_quincenal_catalog",
        lambda: pd.Series(dtype=object, name="Serie"),
    )
    monkeypatch.setattr(
        "inflacion.inegi.series_quincenal.QUINCENAL_HEADLINE_CANDIDATES",
        {"IndiceGeneral": ["XYZ"]},
    )

    # XYZ responde quincenal
    respx.get(f"{BASE}/INDICATOR/XYZ/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    # Y luego XYZ se descarga histórico
    respx.get(f"{BASE}/INDICATOR/XYZ/es/0700/true/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    out = tmp_path / "Q.parquet"
    sidecar = tmp_path / "ids.json"
    df = refresh_inpc_quincenal_with_discovery(
        client=BIEClient(token=TOKEN),
        out_path=out,
        sidecar_path=sidecar,
        historic=True,
    )
    assert "IndiceGeneral" in df.columns
    assert sidecar.exists()
    assert out.exists()
