"""Pruebas del catálogo mensual INPC sin dependencia a Excel."""
from __future__ import annotations

import pandas as pd

from inflacion.inegi.series import load_series_catalog


def test_series_catalog_contains_key_headline_series():
    catalog = load_series_catalog()
    assert "IndiceGeneral" in catalog.index
    assert "Alimentos" in set(catalog.index)
    assert len(catalog) >= 470
    assert catalog["IndiceGeneral"].isdigit()


def test_series_catalog_does_not_call_pandas_read_excel(monkeypatch):
    def _raise_read_excel(*args, **kwargs):
        raise AssertionError("read_excel should not be used")

    monkeypatch.setattr(pd, "read_excel", _raise_read_excel)
    catalog = load_series_catalog()
    assert "IndiceGeneral" in catalog.index
