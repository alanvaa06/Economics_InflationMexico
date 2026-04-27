"""Catálogo de series de INPC (carga del XLSX provisto en el repo)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from inflacion.config import PROJECT_ROOT


def load_series_catalog(path: Path | None = None) -> pd.Series:
    """Devuelve un ``Series`` con índice = nombre del concepto y valor = ID BIE."""
    src = Path(path) if path else PROJECT_ROOT / "SeriesInflation_ids.xlsx"
    df = pd.read_excel(src, index_col=0)
    if "Serie" not in df.columns:
        raise ValueError(f"Catálogo {src} no contiene columna 'Serie'")
    serie = df["Serie"].astype(str).str.strip()
    serie.index = serie.index.astype(str).str.strip()
    return serie
