"""Descomposición de la inflación en componentes (mercancías/servicios, etc.).

Reproduce los gráficos de barras apiladas del notebook original:

- Mercancías vs Servicios → contribución a la **subyacente**.
- Alimentos vs No-alimentos → contribución al **INPC general**.
- Educación / Vivienda / Otros → contribución al **INPC general**.

Bug histórico corregido
-----------------------
La versión original (cell 56) tenía una errata por copy-paste::

    others_list_100 = food_list / others_list.sum()  # ← debería ser others_list

Aquí se calcula correctamente como ``others_list / others_list.sum()``.
"""
from __future__ import annotations

import pandas as pd

from inflacion.data.ponderadores import Ponderadores

INPC_INDEX_LABEL = "IndiceGeneral"


def _share_in_inpc(p: Ponderadores, rubro_col: str) -> float:
    """Peso del rubro en el INPC general (en %, 0-100)."""
    raw = p.raw
    if rubro_col not in raw.columns:
        raise KeyError(f"Rubro {rubro_col!r} no es columna en ponderadores.")
    if INPC_INDEX_LABEL not in raw.index:
        raise KeyError(f"Falta el renglón {INPC_INDEX_LABEL!r} en ponderadores.")
    val = raw.loc[INPC_INDEX_LABEL, rubro_col]
    return float(val)


def _contribution_to_parent(
    items: pd.DataFrame,
    p: Ponderadores,
    *,
    child: str,
    parent_share: float,
    since: str | None,
) -> pd.Series:
    """Contribución YoY del rubro ``child`` a un padre con peso ``parent_share``.

    Replica la fórmula del notebook original:
        child_w_contribution = (items_INPC / items_INPC.sum()) * (child_share / parent_share)
        contribución         = Σ YoY_i * child_w_contribution_i
    """
    if child not in p.weights_inpc:
        raise KeyError(f"Rubro {child!r} no tiene pesos en ponderadores.")
    items_inpc = p.weights_inpc[child]
    rebased = items_inpc / items_inpc.sum()
    child_share = _share_in_inpc(p, child)
    weight = rebased * (child_share / parent_share)

    cols = [c for c in items.columns if c in weight.index]
    if not cols:
        return pd.Series(dtype="float64")
    yoy = items[cols].pct_change(12)
    contrib = yoy.mul(weight.loc[cols], axis=1).sum(axis=1)
    return contrib.loc[since:] if since else contrib


def breakdown_core_goods_services(
    items: pd.DataFrame, p: Ponderadores, *, since: str | None = "2018-01"
) -> pd.DataFrame:
    """Contribución de Mercancías y Servicios a la inflación **subyacente**."""
    parent = _share_in_inpc(p, "Total_Subyacente")
    goods = _contribution_to_parent(
        items, p, child="Total_Mercancias_Subyacente", parent_share=parent, since=since
    )
    services = _contribution_to_parent(
        items, p, child="Total_Servicios", parent_share=parent, since=since
    )
    return pd.concat([goods.rename("Mercancías"), services.rename("Servicios")], axis=1)


def breakdown_core_food_nonfood(
    items: pd.DataFrame, p: Ponderadores, *, since: str | None = "2018-01"
) -> pd.DataFrame:
    """Contribución de Alimentos vs No-alimentos al **INPC general**."""
    food = _contribution_to_parent(
        items,
        p,
        child="AlimentosBebidasTabaco_Mercancias_Subyacente",
        parent_share=100.0,
        since=since,
    )
    nonfood = _contribution_to_parent(
        items,
        p,
        child="NoAlimenticias_Mercacias_Subyacente",
        parent_share=100.0,
        since=since,
    )
    return pd.concat([food.rename("Alimentos"), nonfood.rename("No alimentos")], axis=1)


def breakdown_core_services(
    items: pd.DataFrame, p: Ponderadores, *, since: str | None = "2018-01"
) -> pd.DataFrame:
    """Contribución de Educación / Vivienda / Otros al **INPC general**."""
    edu = _contribution_to_parent(
        items, p, child="Educacion_Servicios_Subyacente", parent_share=100.0, since=since
    )
    housing = _contribution_to_parent(
        items, p, child="Vivienda_Servicios_Subyacente", parent_share=100.0, since=since
    )
    others = _contribution_to_parent(
        items, p, child="Otros_Servicios_Subyacente", parent_share=100.0, since=since
    )
    return pd.concat(
        [edu.rename("Educación"), housing.rename("Vivienda"), others.rename("Otros")], axis=1
    )
