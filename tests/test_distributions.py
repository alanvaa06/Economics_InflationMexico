"""Pruebas de la distribución por rangos de cambio."""
from __future__ import annotations

import pandas as pd

from inflacion.analytics.distributions import bucket_distribution


def test_buckets_sum_to_one():
    yoy = pd.DataFrame(
        {
            "a": [0.01, 0.05, 0.09],
            "b": [0.03, 0.05, 0.07],
            "c": [-0.02, 0.04, 0.10],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    out = bucket_distribution(yoy)
    # cada fila suma a 1 (todas las columnas son no-nulas)
    assert (out.sum(axis=1).round(6) == 1.0).all()


def test_handles_all_nan_row():
    yoy = pd.DataFrame({"a": [None, 0.05]}, index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    out = bucket_distribution(yoy)
    assert (out.iloc[0] == 0).all()
    assert out.iloc[1].sum() == 1.0
