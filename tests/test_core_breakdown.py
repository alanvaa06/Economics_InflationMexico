"""Verifica el módulo de descomposición y específicamente que el bug del
notebook (cell 56) ya no exista.

Construimos un Ponderadores sintético con la misma forma que el XLSX real:

* En la columna ``"INPC"`` viven los pesos por concepto.
* En cada columna de rubro: la celda del renglón ``IndiceGeneral`` guarda la
  participación del rubro en el INPC; las celdas de los conceptos hijos llevan
  la marca ``"X"``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from inflacion.analytics.core_breakdown import (
    breakdown_core_food_nonfood,
    breakdown_core_goods_services,
    breakdown_core_services,
)
from inflacion.data.ponderadores import Ponderadores


def _build_ponderadores(raw: pd.DataFrame) -> Ponderadores:
    weights_inpc: dict[str, pd.Series] = {}
    weights_100: dict[str, pd.Series] = {}
    for col in raw.columns[1:]:
        mask = raw[col] == "X"
        if not mask.any():
            continue
        pesos = raw.loc[mask, "INPC"].astype(float)
        pesos = pesos[pesos > 0]
        if pesos.empty:
            continue
        weights_inpc[col] = pesos
        weights_100[col] = pesos / pesos.sum() * 100.0
    return Ponderadores(raw=raw, weights_100=weights_100, weights_inpc=weights_inpc)


@pytest.fixture()
def synth_ponderadores() -> Ponderadores:
    """Construye un universo simplificado con shares conocidos en INPC."""
    rubros = [
        "INPC",
        "IndiceGeneral",
        "Total_Subyacente",
        "Total_Servicios",
        "Total_Mercancias_Subyacente",
        "Educacion_Servicios_Subyacente",
        "Vivienda_Servicios_Subyacente",
        "Otros_Servicios_Subyacente",
        "AlimentosBebidasTabaco_Mercancias_Subyacente",
        "NoAlimenticias_Mercacias_Subyacente",
    ]
    rows = ["IndiceGeneral", "Edu", "Viv", "Otros", "Alim", "NoAlim"]
    raw = pd.DataFrame("", index=rows, columns=rubros, dtype=object)

    raw["INPC"] = [100.0, 10.0, 20.0, 30.0, 25.0, 15.0]

    # shares en INPC (renglón IndiceGeneral)
    raw.loc["IndiceGeneral", "IndiceGeneral"] = 100.0
    raw.loc["IndiceGeneral", "Total_Subyacente"] = 100.0
    raw.loc["IndiceGeneral", "Total_Servicios"] = 60.0
    raw.loc["IndiceGeneral", "Total_Mercancias_Subyacente"] = 40.0
    raw.loc["IndiceGeneral", "Educacion_Servicios_Subyacente"] = 10.0
    raw.loc["IndiceGeneral", "Vivienda_Servicios_Subyacente"] = 20.0
    raw.loc["IndiceGeneral", "Otros_Servicios_Subyacente"] = 30.0
    raw.loc["IndiceGeneral", "AlimentosBebidasTabaco_Mercancias_Subyacente"] = 25.0
    raw.loc["IndiceGeneral", "NoAlimenticias_Mercacias_Subyacente"] = 15.0

    # marcas X (qué conceptos hijos componen cada rubro)
    raw.loc[["Edu", "Viv", "Otros", "Alim", "NoAlim"], "Total_Subyacente"] = "X"
    raw.loc[["Edu", "Viv", "Otros"], "Total_Servicios"] = "X"
    raw.loc[["Alim", "NoAlim"], "Total_Mercancias_Subyacente"] = "X"
    raw.loc["Edu", "Educacion_Servicios_Subyacente"] = "X"
    raw.loc["Viv", "Vivienda_Servicios_Subyacente"] = "X"
    raw.loc["Otros", "Otros_Servicios_Subyacente"] = "X"
    raw.loc["Alim", "AlimentosBebidasTabaco_Mercancias_Subyacente"] = "X"
    raw.loc["NoAlim", "NoAlimenticias_Mercacias_Subyacente"] = "X"

    return _build_ponderadores(raw)


def _items_with_known_yoy(rates: dict[str, float], n_months: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    data = {col: [100 * ((1 + r) ** (i / 12)) for i in range(n_months)] for col, r in rates.items()}
    return pd.DataFrame(data, index=idx)


def test_goods_services_contribution_signs_and_magnitudes(synth_ponderadores):
    items = _items_with_known_yoy(
        {"Edu": 0.10, "Viv": 0.10, "Otros": 0.10, "Alim": 0.10, "NoAlim": 0.10}
    )
    out = breakdown_core_goods_services(items, synth_ponderadores, since=None)
    last = out.iloc[-1]
    # Subyacente=100, Servicios=60 → 0.10 × 0.6 = 0.06; Mercancías 0.4 → 0.04
    assert abs(last["Servicios"] - 0.06) < 1e-2
    assert abs(last["Mercancías"] - 0.04) < 1e-2


def test_food_nonfood_uses_inpc_share(synth_ponderadores):
    items = _items_with_known_yoy(
        {"Alim": 0.10, "NoAlim": 0.10, "Edu": 0.0, "Viv": 0.0, "Otros": 0.0}
    )
    out = breakdown_core_food_nonfood(items, synth_ponderadores, since=None)
    last = out.iloc[-1]
    # Alim share INPC=25 → 0.10 * 0.25 = 0.025; NoAlim 15 → 0.015
    assert abs(last["Alimentos"] - 0.025) < 1e-3
    assert abs(last["No alimentos"] - 0.015) < 1e-3


def test_services_contribution_otros_uses_correct_weights(synth_ponderadores):
    """Bug original (cell 56): ``Otros`` reusaba la lista de Alimentos.

    Con la fórmula corregida, la contribución de ``Otros`` debe coincidir con
    su peso INPC × YoY (independiente de Alimentos).
    """
    items = _items_with_known_yoy(
        {"Edu": 0.0, "Viv": 0.0, "Otros": 0.20, "Alim": 0.0, "NoAlim": 0.0}
    )
    out = breakdown_core_services(items, synth_ponderadores, since=None)
    last = out.iloc[-1]
    # Otros share=30 → 0.20 * 0.30 = 0.06
    assert abs(last["Otros"] - 0.06) < 1e-3
    assert abs(last["Educación"]) < 1e-6
    assert abs(last["Vivienda"]) < 1e-6
