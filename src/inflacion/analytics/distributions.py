"""Distribución de los componentes según rangos de cambio porcentual."""
from __future__ import annotations

import pandas as pd

DEFAULT_BUCKETS: list[tuple[str, float, float]] = [
    ("<2%", float("-inf"), 0.02),
    ("2-4%", 0.02, 0.04),
    ("4-6%", 0.04, 0.06),
    ("6-8%", 0.06, 0.08),
    (">8%", 0.08, float("inf")),
]


def bucket_distribution(yoy: pd.DataFrame, buckets: list[tuple[str, float, float]] | None = None) -> pd.DataFrame:
    """Para cada fecha, calcula la fracción de columnas que caen en cada rango YoY."""
    rules = buckets or DEFAULT_BUCKETS
    rows = []
    for _ts, line in yoy.iterrows():
        clean = line.dropna()
        n = len(clean)
        if n == 0:
            rows.append({label: 0.0 for label, _, _ in rules})
            continue
        rows.append(
            {
                label: float(((clean >= lo) & (clean < hi)).sum() / n)
                for label, lo, hi in rules
            }
        )
    return pd.DataFrame(rows, index=yoy.index)
