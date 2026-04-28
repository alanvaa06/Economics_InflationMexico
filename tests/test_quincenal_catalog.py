"""load_quincenal_catalog: fallback chain del catálogo quincenal."""
from __future__ import annotations

import json

from inflacion.inegi import series_quincenal


def test_load_catalog_returns_empty_when_no_sidecar_no_overrides(monkeypatch, tmp_path):
    """Sin sidecar y con QUINCENAL_HEADLINE_IDS vacío → Series vacía."""
    monkeypatch.setattr(series_quincenal, "SIDECAR_PATH", tmp_path / "no_existe.json")
    monkeypatch.setattr(series_quincenal, "QUINCENAL_HEADLINE_IDS", {})
    catalog = series_quincenal.load_quincenal_catalog()
    assert catalog.empty


def test_load_catalog_uses_manual_overrides_when_sidecar_missing(monkeypatch, tmp_path):
    """Sin sidecar pero con QUINCENAL_HEADLINE_IDS poblado → usa los overrides."""
    monkeypatch.setattr(series_quincenal, "SIDECAR_PATH", tmp_path / "no_existe.json")
    monkeypatch.setattr(
        series_quincenal,
        "QUINCENAL_HEADLINE_IDS",
        {"IndiceGeneral": "MANUAL_ID"},
    )
    catalog = series_quincenal.load_quincenal_catalog()
    assert catalog.to_dict() == {"IndiceGeneral": "MANUAL_ID"}


def test_load_catalog_prefers_sidecar_over_manual_overrides(monkeypatch, tmp_path):
    """Si el sidecar existe y tiene IDs, gana sobre QUINCENAL_HEADLINE_IDS."""
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "verified_at": "2026-04-27T00:00:00+00:00",
                "ids": {
                    "IndiceGeneral": {"id": "FROM_SIDECAR", "period_format": "YYYY/MM/Qn"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(series_quincenal, "SIDECAR_PATH", sidecar)
    monkeypatch.setattr(
        series_quincenal,
        "QUINCENAL_HEADLINE_IDS",
        {"IndiceGeneral": "MANUAL_ID"},
    )
    catalog = series_quincenal.load_quincenal_catalog()
    assert catalog.to_dict() == {"IndiceGeneral": "FROM_SIDECAR"}


def test_load_catalog_falls_back_when_sidecar_is_corrupt(monkeypatch, tmp_path, caplog):
    """Sidecar corrupto → log warning + cae a QUINCENAL_HEADLINE_IDS."""
    sidecar = tmp_path / "corrupt.json"
    sidecar.write_text("{ this is not json", encoding="utf-8")

    monkeypatch.setattr(series_quincenal, "SIDECAR_PATH", sidecar)
    monkeypatch.setattr(
        series_quincenal,
        "QUINCENAL_HEADLINE_IDS",
        {"IndiceGeneral": "MANUAL_FALLBACK"},
    )

    with caplog.at_level("WARNING", logger="inflacion.inegi.series_quincenal"):
        catalog = series_quincenal.load_quincenal_catalog()

    assert catalog.to_dict() == {"IndiceGeneral": "MANUAL_FALLBACK"}
    assert any("Sidecar quincenal corrupto" in rec.message for rec in caplog.records)
