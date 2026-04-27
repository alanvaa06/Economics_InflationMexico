"""Z-score con ventana móvil (no expansiva)."""
from __future__ import annotations

import pandas as pd


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """``(x - rolling_mean) / rolling_std`` usando estadísticos pasados (shift=1)."""
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std().shift(1)
    return (x - m) / s
