"""resolve_quincenal_ids: sondea candidatos y persiste sidecar."""
from __future__ import annotations

import json

import httpx
import respx

from inflacion.data.quincenal_resolver import resolve_quincenal_ids
from inflacion.inegi.client import BIEClient

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"


def _q_payload(tp: str = "2024/03/Q1") -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": tp, "OBS_VALUE": "133.0"}]}]}


def _m_payload(tp: str = "2024/03") -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": tp, "OBS_VALUE": "133.0"}]}]}


@respx.mock
def test_resolver_picks_first_quincenal_candidate(tmp_path):
    candidates = {
        "IndiceGeneral": ["AAA", "BBB"],
        "Subyacente":    ["CCC"],
    }
    # AAA responde mensual → no
    respx.get(f"{BASE}/INDICATOR/AAA/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_m_payload())
    )
    # BBB responde quincenal → sí
    respx.get(f"{BASE}/INDICATOR/BBB/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    # CCC responde quincenal → sí
    respx.get(f"{BASE}/INDICATOR/CCC/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    sidecar = tmp_path / "ids.json"
    with BIEClient(token=TOKEN) as client:
        resolved = resolve_quincenal_ids(client, candidates, sidecar_path=sidecar)

    assert resolved == {"IndiceGeneral": "BBB", "Subyacente": "CCC"}
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "verified_at" in saved
    assert saved["ids"]["IndiceGeneral"]["id"] == "BBB"
    assert saved["ids"]["Subyacente"]["id"] == "CCC"


@respx.mock
def test_resolver_skips_concept_when_all_candidates_fail(tmp_path):
    candidates = {"IndiceGeneral": ["AAA"], "Subyacente": ["BBB"]}
    respx.get(f"{BASE}/INDICATOR/AAA/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(403, text="bad")
    )
    respx.get(f"{BASE}/INDICATOR/BBB/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    sidecar = tmp_path / "ids.json"
    with BIEClient(token=TOKEN) as client:
        resolved = resolve_quincenal_ids(client, candidates, sidecar_path=sidecar)

    assert resolved == {"Subyacente": "BBB"}


@respx.mock
def test_resolver_invokes_progress_cb(tmp_path):
    candidates = {"IndiceGeneral": ["AAA"], "Subyacente": ["BBB"]}
    respx.get(f"{BASE}/INDICATOR/AAA/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    respx.get(f"{BASE}/INDICATOR/BBB/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    progress: list[tuple[int, int, str]] = []
    with BIEClient(token=TOKEN) as client:
        resolve_quincenal_ids(
            client, candidates, sidecar_path=tmp_path / "ids.json",
            progress_cb=lambda d, t, n: progress.append((d, t, n)),
        )

    assert [c[0] for c in progress] == [1, 2]
    assert progress[0][1] == 2
