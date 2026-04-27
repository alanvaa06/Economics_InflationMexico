"""Detección de outliers / efectos base en componentes del INPC.

Estrategia: para cada serie, en cada punto, calcula el z-score con una ventana
expansiva (todo el histórico hasta `t-1`). Marca un punto como outlier si su
|z-score| supera el umbral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def identify_outliers_expanding_window(
    df: pd.DataFrame, window: int = 36, threshold: float = 2.0
) -> pd.DataFrame:
    """Devuelve un DataFrame booleano del mismo shape, ``True`` en outliers de la ventana final.

    Args:
        df: serie(s) del INPC con índice temporal.
        window: cuántos puntos finales evaluar.
        threshold: umbral absoluto de z-score (mean/std expansivos).
    """
    if df.empty:
        return df.copy().astype(bool)

    means = df.expanding(min_periods=2).mean().shift(1)
    stds = df.expanding(min_periods=2).std().shift(1)
    z = (df - means) / stds.replace(0, np.nan)
    flags = z.abs() > threshold

    if window > 0 and len(df) > window:
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        mask.iloc[-window:] = True
        flags = flags & mask

    return flags.fillna(False)
