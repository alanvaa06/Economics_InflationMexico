"""Núcleo alternativo: media truncada y mediana ponderada del INPC.

Se calculan **directamente sobre la sección transversal de YoY** por componente:
para cada fecha, se ordenan las tasas YoY, se ponderan por el peso de cada
componente y se descarta una fracción ``trim`` de cada cola (truncada) o se
toma el valor cuya masa acumulada cruza 0.5 (mediana).

Trabajar sobre YoY (en vez de MoM y luego anualizar) tiene tres ventajas:

- comparte unidades con :func:`contributions_yoy` y con la banda Banxico,
- evita componer 12 cambios MoM con NaN intermedios,
- coincide con la frecuencia que reportan Banxico y los analistas privados.

Estos indicadores no excluyen una lista fija (como la subyacente oficial), sino
que filtran los movimientos atípicos por sección transversal — útiles cuando
un solo subgrupo "ensucia" la subyacente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_NON_NAN = 3


def _normalize_weights(weights: pd.Series, columns: pd.Index) -> pd.Series:
    cols = [c for c in columns if c in weights.index]
    if not cols:
        raise ValueError("Ningún componente tiene peso definido en el rubro.")
    w = weights.loc[cols].astype(float)
    w = w[w > 0]
    if w.empty or w.sum() <= 0:
        raise ValueError("Suma de pesos <= 0; revisa los ponderadores.")
    return w / w.sum()


def trimmed_mean_yoy(
    items: pd.DataFrame,
    weights: pd.Series,
    *,
    trim: float = 0.10,
    since: str | None = "2018-01",
) -> pd.Series:
    """Media truncada por peso de la sección transversal de tasas YoY.

    Args:
        items: niveles del INPC por componente (columnas) y fecha (índice).
        weights: pesos por componente (cualquier escala; se renormalizan a 1
            sobre los componentes presentes en cada fecha).
        trim: fracción a cortar de **cada** cola (default 10%; ``trim=0``
            equivale a la media ponderada simple).
        since: fecha mínima a conservar (``"YYYY-MM"`` o ``None``).
    """
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim debe estar en [0, 0.5)")

    w = _normalize_weights(weights, items.columns)
    yoy = items[w.index].pct_change(12, fill_method=None)

    out = pd.Series(index=yoy.index, dtype="float64", name="trimmed_mean_yoy")
    for ts, row in yoy.iterrows():
        mask = row.notna()
        if mask.sum() < _MIN_NON_NAN:
            continue
        rates = row[mask].astype(float)
        ws = (w[mask] / w[mask].sum()).astype(float)
        out.loc[ts] = _trimmed_mean_xs(rates.values, ws.values, trim)

    return out.loc[since:] if since else out


def weighted_median_yoy(
    items: pd.DataFrame,
    weights: pd.Series,
    *,
    since: str | None = "2018-01",
) -> pd.Series:
    """Mediana ponderada de la sección transversal de tasas YoY (CPI median)."""
    w = _normalize_weights(weights, items.columns)
    yoy = items[w.index].pct_change(12, fill_method=None)

    out = pd.Series(index=yoy.index, dtype="float64", name="weighted_median_yoy")
    for ts, row in yoy.iterrows():
        mask = row.notna()
        if mask.sum() < _MIN_NON_NAN:
            continue
        rates = row[mask].astype(float)
        ws = (w[mask] / w[mask].sum()).astype(float)
        out.loc[ts] = _weighted_median_xs(rates.values, ws.values)

    return out.loc[since:] if since else out


def _trimmed_mean_xs(rates: np.ndarray, weights: np.ndarray, trim: float) -> float:
    """Media truncada con interpolación parcial en el peso del límite."""
    order = np.argsort(rates, kind="mergesort")
    r, w = rates[order], weights[order]
    cum = np.cumsum(w)
    total = cum[-1]
    lo = trim * total
    hi = (1.0 - trim) * total

    # Construye los pesos efectivos: cero por fuera de [lo, hi], parcial en
    # los componentes que cruzan los límites.
    left = np.maximum(0.0, np.minimum(cum, hi) - np.maximum(cum - w, lo))
    eff = np.where(left > 0, left, 0.0)
    s = eff.sum()
    if s <= 0:
        return float("nan")
    return float(np.dot(eff, r) / s)


def _weighted_median_xs(rates: np.ndarray, weights: np.ndarray) -> float:
    """Mediana ponderada: rate cuya masa acumulada cruza 0.5.

    Cuando el peso acumulado cae *exactamente* sobre un límite (caso simétrico
    poco frecuente), promediamos los dos rates colindantes — convención
    estándar para mediana ponderada.
    """
    order = np.argsort(rates, kind="mergesort")
    r, w = rates[order], weights[order]
    cum = np.cumsum(w)
    target = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, target, side="left"))
    if idx >= len(r):
        return float(r[-1])
    if cum[idx] == target and idx + 1 < len(r):
        return float(0.5 * (r[idx] + r[idx + 1]))
    return float(r[idx])
