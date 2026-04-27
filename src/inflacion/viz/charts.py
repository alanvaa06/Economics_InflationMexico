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


def yoy_line_with_band(
    series: pd.Series | pd.DataFrame,
    title: str,
    *,
    target: float = 0.03,
    tolerance: float = 0.01,
    band_label: str = "Objetivo Banxico (3% ± 1pp)",
) -> go.Figure:
    """Líneas YoY con la banda objetivo de Banxico sombreada.

    Args:
        series: tasa(s) YoY en escala decimal (e.g. 0.0432 para 4.32%).
            Si es ``Series`` se traza una línea; si es ``DataFrame``,
            una línea por columna.
        title: título de la figura.
        target: nivel central del objetivo (default 3%).
        tolerance: amplitud de la banda en cada dirección (default ±1pp).
        band_label: texto que aparece junto a la banda.
    """
    df = series.to_frame() if isinstance(series, pd.Series) else series.copy()
    fig = go.Figure()

    # Banda + línea central
    band_lo = (target - tolerance) * 100
    band_hi = (target + tolerance) * 100
    fig.add_hrect(
        y0=band_lo,
        y1=band_hi,
        fillcolor="rgba(45,204,205,0.12)",
        line_width=0,
        annotation_text=band_label,
        annotation_position="top right",
        annotation_font_size=11,
    )
    fig.add_hline(y=target * 100, line_dash="dash", line_color="#1973B8", line_width=1)

    drawn = 0
    for i, col in enumerate(df.columns):
        s = (df[col] * 100).dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=str(col),
                line={"color": BBVA_PALETTE[i % len(BBVA_PALETTE)], "width": 2},
            )
        )
        drawn += 1

    if drawn == 0:
        fig.add_annotation(
            text="Sin datos suficientes",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#666"},
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Fecha",
        yaxis_title="Variación YoY (%)",
        legend_title="Serie",
        hovermode="x unified",
    )
    return fig
