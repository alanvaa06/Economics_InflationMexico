"""Orquestación: descarga + persistencia local del INPC."""
from __future__ import annotations

import logging
from collections.abc import Callable
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
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Descarga todas las series del catálogo y devuelve un DataFrame ancho.

    Args:
        historic: ``True`` trae la serie completa; ``False`` solo lo más reciente.
        out_path: si se indica, guarda en parquet (recomendado) o xlsx.
        client: cliente a reutilizar (útil para tests o múltiples llamadas).
        progress_cb: función opcional ``(done, total, name) -> None`` llamada
            por cada serie descargada; útil para barras de progreso en UI.
    """
    catalog = load_series_catalog()
    owns_client = client is None
    bie = client or BIEClient()
    try:
        wide = bie.fetch_many(catalog.tolist(), historic=historic, progress_cb=progress_cb)
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
    """Carga el INPC local (parquet) o lo descarga de INEGI si falta o está vacío."""
    target = Path(path) if path else settings.data_dir / "RelevantInflation.parquet"
    if target.exists():
        df = pd.read_parquet(target)
        if df.empty or df.shape[1] == 0:
            logger.warning("Parquet vacío en %s — descartando y re-descargando.", target)
            target.unlink()
        else:
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

    try:
        settings.require_token()
    except RuntimeError as exc:
        raise FileNotFoundError(
            "No se encontró cache local del INPC y falta `INEGI_API_TOKEN`. "
            "Configura `.env` o ejecuta `inflacion refresh` en un entorno con token."
        ) from exc

    logger.info("INPC local no encontrado; descargando desde INEGI BIE a %s", target)
    # Sondeo de salud antes de bombardear con ~471 series: si el token está
    # rechazado, MissingTokenError sube y la app puede mostrar onboarding sin
    # esperar a que cada serie individual falle con HTTP 400.
    bie = BIEClient()
    try:
        bie.health_check()
        refresh_inpc(historic=True, out_path=target, client=bie)
    finally:
        bie.close()
    if target.exists():
        df = pd.read_parquet(target)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    raise FileNotFoundError("No se pudo generar `RelevantInflation.parquet` desde INEGI.")


def refresh_inpc_quincenal(
    *,
    historic: bool = True,
    out_path: Path | None = None,
    client: BIEClient | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Descarga las series quincenales (encabezados) del INPC desde INEGI BIE.

    Args:
        historic: ``True`` trae la serie completa; ``False`` solo lo más reciente.
        out_path: si se indica, guarda en parquet (recomendado) o xlsx. Por
            defecto: ``data/RelevantInflation_Q.parquet``.
        client: cliente a reutilizar (útil para tests o múltiples llamadas).
        progress_cb: función opcional ``(done, total, name) -> None`` llamada
            por cada serie descargada; útil para barras de progreso en UI.
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
        wide = bie.fetch_many(catalog.tolist(), historic=historic, progress_cb=progress_cb)
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


def refresh_inpc_quincenal_with_discovery(
    *,
    client: BIEClient | None = None,
    out_path: Path | None = None,
    sidecar_path: Path | None = None,
    historic: bool = True,
    discovery_progress_cb: Callable[[int, int, str], None] | None = None,
    fetch_progress_cb: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Resuelve IDs quincenales (si hace falta) y descarga el histórico.

    Si ``load_quincenal_catalog`` ya devuelve un mapping no-vacío (sidecar previo
    o overrides manuales), salta el resolver. Si no, sondea
    ``QUINCENAL_HEADLINE_CANDIDATES`` y persiste el sidecar antes del fetch.
    """
    from inflacion.data.quincenal_resolver import resolve_quincenal_ids
    from inflacion.inegi.series_quincenal import (
        QUINCENAL_HEADLINE_CANDIDATES,
        SIDECAR_PATH,
    )

    sidecar = sidecar_path or SIDECAR_PATH
    catalog = load_quincenal_catalog()
    owns_client = client is None
    bie = client or BIEClient()
    try:
        if catalog.empty:
            resolved = resolve_quincenal_ids(
                bie,
                QUINCENAL_HEADLINE_CANDIDATES,
                sidecar_path=sidecar,
                progress_cb=discovery_progress_cb,
            )
            if not resolved:
                raise RuntimeError(
                    "Ningún candidato quincenal respondió. Revisa token o agrega IDs en "
                    "QUINCENAL_HEADLINE_IDS / QUINCENAL_HEADLINE_CANDIDATES."
                )
            catalog = pd.Series(resolved, name="Serie")

        wide = bie.fetch_many(
            catalog.tolist(), historic=historic, progress_cb=fetch_progress_cb
        )
    finally:
        if owns_client:
            bie.close()

    id_to_name = {str(v): str(k) for k, v in catalog.items()}
    wide = wide.rename(columns=id_to_name)

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
    """Carga el INPC quincenal local (parquet) o lo descarga de INEGI si falta o está vacío."""
    target = Path(path) if path else settings.data_dir / "RelevantInflation_Q.parquet"
    if target.exists():
        df = pd.read_parquet(target)
        if df.empty or df.shape[1] == 0:
            logger.warning("Parquet quincenal vacío en %s — descartando.", target)
            target.unlink()
        else:
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

    try:
        settings.require_token()
    except RuntimeError as exc:
        raise FileNotFoundError(
            "No se encontró cache local quincenal y falta `INEGI_API_TOKEN`. "
            "Configura `.env` o ejecuta `inflacion refresh --frequency quincenal`."
        ) from exc

    logger.info("INPC quincenal local no encontrado; descargando desde INEGI BIE a %s", target)
    # Mismo sondeo de salud que load_local_inpc: para que un token caducado
    # no dispare 10+ probes inútiles antes de fallar.
    bie = BIEClient()
    try:
        bie.health_check()
        refresh_inpc_quincenal(historic=True, out_path=target, client=bie)
    finally:
        bie.close()
    if target.exists():
        df = pd.read_parquet(target)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    raise FileNotFoundError("No se pudo generar `RelevantInflation_Q.parquet` desde INEGI.")
