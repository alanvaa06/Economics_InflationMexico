"""Pruebas de núcleo alternativo (media truncada + mediana ponderada)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inflacion.analytics.alt_core import (
    _trimmed_mean_xs,
    _weighted_median_xs,
    trimmed_mean_yoy,
    weighted_median_yoy,
)


def _items_with_yoy(rates: dict[str, float], n: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2022-01-31", periods=n, freq="ME")
    data = {col: [100 * ((1 + r) ** (i / 12)) for i in range(n)] for col, r in rates.items()}
    return pd.DataFrame(data, index=idx)


def test_trimmed_mean_drops_outliers():
    """Una observación extrema en la cola debe quedar fuera con trim>0."""
    # 5 componentes con peso uniforme; cuatro al 4% YoY y una al 50%.
    rates = np.array([0.04, 0.04, 0.04, 0.04, 0.50])
    weights = np.full(5, 0.2)
    # trim=0.20 quita la cola extrema completa
    out = _trimmed_mean_xs(rates, weights, trim=0.20)
    assert out == pytest.approx(0.04)


def test_trimmed_mean_zero_equals_weighted_mean():
    """trim=0 debe coincidir con la media ponderada simple."""
    items = _items_with_yoy({"A": 0.05, "B": 0.10, "C": 0.02})
    weights = pd.Series({"A": 50.0, "B": 30.0, "C": 20.0})
    tm0 = trimmed_mean_yoy(items, weights, trim=0.0, since=None)
    last = tm0.iloc[-1]
    expected = 0.05 * 0.5 + 0.10 * 0.3 + 0.02 * 0.2
    assert last == pytest.approx(expected, abs=5e-3)


def test_trimmed_mean_invalid_trim():
    items = _items_with_yoy({"A": 0.04})
    with pytest.raises(ValueError):
        trimmed_mean_yoy(items, pd.Series({"A": 1.0}), trim=0.6, since=None)
    with pytest.raises(ValueError):
        trimmed_mean_yoy(items, pd.Series({"A": 1.0}), trim=-0.1, since=None)


def test_weighted_median_with_dominant_weight():
    """Un componente con > 50% del peso debe ser SIEMPRE la mediana."""
    items = _items_with_yoy({"A": 0.20, "B": 0.05, "C": 0.10})
    weights = pd.Series({"A": 60.0, "B": 25.0, "C": 15.0})
    wm = weighted_median_yoy(items, weights, since=None).dropna()
    # Para todas las fechas con YoY válido, la mediana debe ser ~A
    assert (np.abs(wm - 0.20) < 5e-3).all()


def test_alt_core_handles_partial_nan_row():
    """Filas con < 3 componentes no-NaN devuelven NaN, no excepción."""
    idx = pd.date_range("2022-01-31", periods=24, freq="ME")
    items = pd.DataFrame(
        {
            "A": [100 * (1.05 ** (i / 12)) for i in range(24)],
            "B": [np.nan] * 23 + [110.0],  # solo 1 valor → no hay YoY
            "C": [np.nan] * 24,
        },
        index=idx,
    )
    weights = pd.Series({"A": 60.0, "B": 25.0, "C": 15.0})
    tm = trimmed_mean_yoy(items, weights, since=None)
    wm = weighted_median_yoy(items, weights, since=None)
    # No revienta y la mayoría de filas son NaN (porque sólo A tiene YoY)
    assert tm.isna().sum() > 12
    assert wm.isna().sum() > 12


def test_weighted_median_xs_simple():
    """Caso explícito: 3 valores con pesos iguales → mediana = valor del medio."""
    rates = np.array([0.01, 0.05, 0.09])
    weights = np.array([1.0, 1.0, 1.0])
    assert _weighted_median_xs(rates, weights) == pytest.approx(0.05)
