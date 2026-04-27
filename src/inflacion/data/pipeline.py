"""Orquestación: descarga + persistencia local del INPC."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from inflacion.config import settings
from inflacion.inegi.client import BIEClient
from inflacion.inegi.series import load_series_catalog
from inflacion.inegi.series_quincenal import load_quincenal_catalog

logger = logging.getLogger(__name__)


def refresh_inpc(
    *,
    historic: bool = True,
    out_path: Path | None = None,
    client: BIEClient | None = None,
) -> pd.DataFrame:
    """Descarga todas las series del catálogo y devuelve un DataFrame ancho.

    Args:
        historic: ``True`` trae la serie completa; ``False`` solo lo más reciente.
        out_path: si se indica, guarda en parquet (recomendado) o xlsx.
        client: cliente a reutilizar (útil para tests o múltiples llamadas).
    """
    catalog = load_series_catalog()
    owns_client = client is None
    bie = client or BIEClient()
    try:
        wide = bie.fetch_many(catalog.tolist(), historic=historic)
    finally:
        if owns_client:
            bie.close()

    # renombramos columnas: id -> nombre del concepto
    id_to_name = {str(v): str(k) for k, v in catalog.items()}
    wide = wide.rename(columns=id_to_name)

    if out_path:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".parquet":
            wide.to_parquet(target)
        else:
            wide.to_excel(target)
        logger.info("INPC guardado en %s (%d filas, %d columnas)", target, *wide.shape)
    return wide


def load_local_inpc(path: Path | None = None) -> pd.DataFrame:
    """Carga el INPC ya descargado (parquet preferido, xlsx como fallback)."""
    candidates = (
        [Path(path)]
        if path
        else [
            settings.data_dir / "RelevantInflation.parquet",
            settings.data_dir.parent / "RelevantInflation.xlsx",
        ]
    )
    for c in candidates:
        if c.exists():
            df = pd.read_parquet(c) if c.suffix == ".parquet" else pd.read_excel(c, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
    raise FileNotFoundError(
        "No se encontró un archivo INPC local. Ejecuta `inflacion refresh` primero."
    )


def refresh_inpc_quincenal(
    *,
    historic: bool = True,
    out_path: Path | None = None,
    client: BIEClient | None = None,
) -> pd.DataFrame:
    """Descarga las series quincenales (encabezados) del INPC desde INEGI BIE.

    Args:
        historic: ``True`` trae la serie completa; ``False`` solo lo más reciente.
        out_path: si se indica, guarda en parquet (recomendado) o xlsx. Por
            defecto: ``data/RelevantInflation_Q.parquet``.
        client: cliente a reutilizar (útil para tests o múltiples llamadas).
    """
    catalog = load_quincenal_catalog()
    if catalog.empty:
        raise RuntimeError(
            "Catálogo quincenal vacío. Edita "
            "src/inflacion/inegi/series_quincenal.py con los IDs reales del BIE."
        )
    owns_client = client is None
    bie = client or BIEClient()
    try:
        wide = bie.fetch_many(catalog.tolist(), historic=historic)
    finally:
        if owns_client:
            bie.close()

    id_to_name = {str(v): str(k) for k, v in catalog.items()}
    wide = wide.rename(columns=id_to_name)

    expected = len(catalog)
    obtained = len(wide.columns)
    if obtained < expected:
        logger.warning(
            "Quincenal incompleto: se esperaban %d series, llegaron %d.", expected, obtained
        )

    if out_path:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".parquet":
            wide.to_parquet(target)
        else:
            wide.to_excel(target)
        logger.info(
            "INPC quincenal guardado en %s (%d filas, %d columnas)", target, *wide.shape
        )
    return wide


def load_local_inpc_quincenal(path: Path | None = None) -> pd.DataFrame:
    """Carga el INPC quincenal ya descargado (parquet preferido)."""
    candidates = (
        [Path(path)]
        if path
        else [
            settings.data_dir / "RelevantInflation_Q.parquet",
            settings.data_dir.parent / "RelevantInflation_Q.xlsx",
        ]
    )
    for c in candidates:
        if c.exists():
            df = pd.read_parquet(c) if c.suffix == ".parquet" else pd.read_excel(c, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
    raise FileNotFoundError(
        "No se encontró un archivo INPC quincenal local. "
        "Ejecuta `inflacion refresh --frequency quincenal` primero."
    )
