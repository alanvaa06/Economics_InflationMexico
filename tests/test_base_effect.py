"""Pruebas del detector de outliers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from inflacion.analytics.base_effect import identify_outliers_expanding_window


def test_detects_injected_outlier():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=120)
    series = pd.Series(base, index=pd.date_range("2015-01-31", periods=120, freq="ME"))
    series.iloc[-3] += 10  # outlier evidente
    df = series.to_frame("x")

    flags = identify_outliers_expanding_window(df, window=12, threshold=2.0)

    assert flags["x"].iloc[-3]
    # promedio de outliers en el resto de la ventana debe ser bajo
    assert flags["x"].iloc[-12:].sum() <= 2


def test_no_false_positive_on_constant_series():
    df = pd.DataFrame({"x": [1.0] * 60}, index=pd.date_range("2015-01-31", periods=60, freq="ME"))
    flags = identify_outliers_expanding_window(df, window=12, threshold=2.0)
    assert not flags["x"].any()


def test_handles_empty_dataframe():
    df = pd.DataFrame()
    out = identify_outliers_expanding_window(df)
    assert out.empty
