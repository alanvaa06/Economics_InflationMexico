"""Contribuciones a la inflación (MoM y YoY) por concepto, dado un set de pesos."""
from __future__ import annotations

import pandas as pd


def _align_weights(items: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Devuelve pesos alineados a las columnas presentes, normalizando a 1."""
    cols = [c for c in items.columns if c in weights.index]
    if not cols:
        raise ValueError("Ninguna de las columnas tiene peso definido en el rubro.")
    w = weights.loc[cols].astype(float)
    total = w.sum()
    if total <= 0:
        raise ValueError("Suma de pesos <= 0; revisa los ponderadores.")
    return w / total


def contributions_yoy(
    items: pd.DataFrame, weights: pd.Series, *, since: str | None = "2018-01"
) -> pd.DataFrame:
    """Contribución YoY por concepto: ``pct_change(12) * peso_normalizado``.

    Args:
        items: niveles del INPC por concepto (columnas) y fecha (índice).
        weights: pesos del rubro (cualquier escala; se normalizan a 1).
        since: fecha mínima a conservar en el resultado (``"YYYY-MM"`` o ``None``).
    """
    w = _align_weights(items, weights)
    yoy = items[w.index].pct_change(12)
    contrib = yoy.mul(w, axis=1)
    return contrib.loc[since:] if since else contrib


def contributions_mom(
    items: pd.DataFrame, weights: pd.Series, *, since: str | None = "2018-01"
) -> pd.DataFrame:
    w = _align_weights(items, weights)
    mom = items[w.index].pct_change(1)
    contrib = mom.mul(w, axis=1)
    return contrib.loc[since:] if since else contrib


def incidencias(contrib: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-n contribuciones positivas y negativas del último período.

    Devuelve un DataFrame con columnas ``contribucion`` y ``pct_total`` (% del total
    contribuido), ordenado de mayor a menor contribución.
    """
    if contrib.empty:
        return pd.DataFrame(columns=["contribucion", "pct_total"])
    last = contrib.iloc[-1].dropna()
    last = last[last != 0].sort_values(ascending=False)
    if last.empty:
        return pd.DataFrame(columns=["contribucion", "pct_total"])
    head = last.head(n)
    tail = last.tail(n)
    sel = pd.concat([head, tail])
    sel = sel[~sel.index.duplicated(keep="first")]
    total = last.sum()
    out = pd.DataFrame(
        {
            "contribucion": sel,
            "pct_total": sel / total if total != 0 else 0.0,
        }
    )
    return out.sort_values("contribucion", ascending=False)
