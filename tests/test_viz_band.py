"""Pruebas del gráfico YoY con la banda objetivo de Banxico."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from inflacion.viz import yoy_line_with_band


def _yoy_series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2022-01-31", periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name="IndiceGeneral")


def test_yoy_line_with_band_renders_target_band():
    s = _yoy_series([0.045, 0.043, 0.042, 0.040, 0.038, 0.036])
    fig = yoy_line_with_band(s, "Test")

    assert isinstance(fig, go.Figure)
    # Una línea
    line_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines"]
    assert len(line_traces) == 1
    # Banda (hrect) + línea central (hline) → 2 shapes
    assert len(fig.layout.shapes) >= 2
    # La banda cubre 2%–4% (tras escalar a porcentaje)
    rect = next(s for s in fig.layout.shapes if s.type == "rect")
    assert rect.y0 == pytest.approx(2.0)
    assert rect.y1 == pytest.approx(4.0)


def test_yoy_line_with_band_skips_all_nan_columns():
    df = pd.DataFrame(
        {
            "ok": [0.04, 0.045, 0.05],
            "vacio": [np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    fig = yoy_line_with_band(df, "Test")
    line_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines"]
    # Sólo "ok" debe dibujarse
    assert len(line_traces) == 1
    assert line_traces[0].name == "ok"


def test_yoy_line_with_band_empty_dataframe_renders_message():
    df = pd.DataFrame({"x": []}, index=pd.DatetimeIndex([], name="fecha"))
    fig = yoy_line_with_band(df, "Test")
    line_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines"]
    assert len(line_traces) == 0
    # Debe haber al menos una anotación con el mensaje
    annotations = [a for a in fig.layout.annotations if "Sin datos" in (a.text or "")]
    assert annotations
