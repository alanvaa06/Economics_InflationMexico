"""Figuras Plotly reutilizables (etiquetas en español)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BBVA_PALETTE = ["#004481", "#2DCCCD", "#1973B8", "#A5A5A5", "#D8BE75"]


def contributions_bar(df: pd.DataFrame, title: str) -> go.Figure:
    """Barras apiladas (cada columna = un componente)."""
    long = (df * 100).reset_index().melt(id_vars=df.index.name or "index", var_name="Componente", value_name="Contribución (%)")
    fig = px.bar(
        long,
        x=long.columns[0],
        y="Contribución (%)",
        color="Componente",
        color_discrete_sequence=BBVA_PALETTE,
        title=title,
    )
    fig.update_layout(barmode="stack", xaxis_title="Fecha", template="plotly_white")
    return fig


def distribution_area(df: pd.DataFrame, title: str) -> go.Figure:
    """Áreas apiladas en porcentaje (rangos de cambios)."""
    long = (df * 100).reset_index().melt(id_vars=df.index.name or "index", var_name="Rango", value_name="% componentes")
    fig = px.area(
        long,
        x=long.columns[0],
        y="% componentes",
        color="Rango",
        color_discrete_sequence=BBVA_PALETTE,
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_title="Fecha")
    return fig


def percentiles_heatmap(percentiles: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        percentiles.T,
        aspect="auto",
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Componente")
    return fig


def incidencias_bar(df: pd.DataFrame, title: str) -> go.Figure:
    """Barras horizontales: top contribuciones positivas y negativas."""
    df = df.copy().sort_values("contribucion")
    fig = px.bar(
        df,
        x="contribucion",
        y=df.index,
        orientation="h",
        title=title,
        color="contribucion",
        color_continuous_scale=["#004481", "#A5A5A5", "#D8BE75"],
    )
    fig.update_layout(template="plotly_white", xaxis_title="Contribución", yaxis_title="")
    return fig
