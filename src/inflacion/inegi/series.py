"""Catálogo de series del INPC (nombre amigable -> ID BIE)."""
from __future__ import annotations

import pandas as pd

from inflacion.inegi.series_catalog_data import INPC_SERIES_PAIRS


def load_series_catalog() -> pd.Series:
    """Devuelve el catálogo mensual INPC como ``Series`` (nombre -> ID BIE)."""
    if not INPC_SERIES_PAIRS:
        raise ValueError("Catálogo INPC vacío.")
    ids = [sid for _, sid in INPC_SERIES_PAIRS]
    labels = [name for name, _ in INPC_SERIES_PAIRS]
    return pd.Series(ids, index=labels, name="Serie")
