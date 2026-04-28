"""Carga y normaliza ponderadores INPC desde fuente oficial de INEGI.

Fuente oficial:
- RNM INEGI: ``/catalog/1015/download/32034`` (canasta y ponderadores 2024).

El módulo mantiene el contrato histórico ``Ponderadores`` y la misma lógica de
pesos rebasados a 100 por rubro para no romper la capa analítica.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from inflacion.config import settings
from inflacion.data.ponderadores_groups_data import AGGREGATE_MEMBERS

OFFICIAL_WEIGHTS_URL = "https://www.inegi.org.mx/rnm/index.php/catalog/1015/download/32034"
OFFICIAL_WEIGHTS_XLSX = settings.cache_dir / "inpc_canasta_ponderadores_32034.xlsx"
OFFICIAL_WEIGHTS_PARQUET = settings.data_dir / "ponderadores_inpc_official.parquet"


@dataclass(frozen=True)
class Ponderadores:
    """Resultado de leer la hoja de ponderadores."""

    raw: pd.DataFrame
    """Hoja completa tal como viene del XLSX."""

    weights_100: dict[str, pd.Series]
    """Por rubro: pesos rebasados a 100 (suman 100±ε), index = concepto."""

    weights_inpc: dict[str, pd.Series]
    """Por rubro: pesos en escala INPC (suman al peso del rubro en el índice general)."""

    def aggregates(self) -> list[str]:
        return list(self.weights_100.keys())

    def items(self, aggregate: str) -> list[str]:
        return self.weights_100[aggregate].index.tolist()


def load_ponderadores(path: Path | None = None) -> Ponderadores:
    """Carga ponderadores oficiales de INEGI y arma diccionarios de pesos.

    Args:
        path: ruta opcional a un workbook oficial INEGI para parseo directo.
    """
    raw = _build_raw_from_workbook(Path(path)) if path else _load_or_build_cached_raw()

    if "INPC" not in raw.columns:
        raise ValueError("Ponderadores oficiales: falta columna 'INPC'")

    weights_inpc: dict[str, pd.Series] = {}
    weights_100: dict[str, pd.Series] = {}

    # Cada columna a partir de la 1ª (después de INPC) es un rubro/agregado.
    for rubro in raw.columns[1:]:
        marca = raw[rubro] == "X"
        if not marca.any():
            continue
        pesos = raw.loc[marca, "INPC"].astype(float)
        pesos = pesos[pesos > 0]
        if pesos.empty:
            continue
        weights_inpc[rubro] = pesos
        weights_100[rubro] = (pesos / pesos.sum() * 100.0).rename(rubro)

    return Ponderadores(raw=raw, weights_100=weights_100, weights_inpc=weights_inpc)


def _load_or_build_cached_raw() -> pd.DataFrame:
    cache = Path(OFFICIAL_WEIGHTS_PARQUET)
    if cache.exists():
        return pd.read_parquet(cache)

    src = Path(OFFICIAL_WEIGHTS_XLSX)
    if not src.exists():
        _download_official_workbook(src)

    raw = _build_raw_from_workbook(src)
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache)
    return raw


def _download_official_workbook(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(OFFICIAL_WEIGHTS_URL)
        response.raise_for_status()
        destination.write_bytes(response.content)


def _build_raw_from_workbook(workbook_path: Path) -> pd.DataFrame:
    table = _load_official_table(workbook_path)
    weights = _build_weight_series(table)
    raw = _compose_raw(weights)
    return raw


def _load_official_table(workbook_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(workbook_path)
    if not xls.sheet_names:
        raise ValueError("Workbook oficial INEGI sin hojas.")
    sheet = xls.sheet_names[0]
    table = pd.read_excel(workbook_path, sheet_name=sheet)
    if table.shape[1] < 3:
        raise ValueError("Workbook oficial INEGI con estructura inesperada.")
    return table.iloc[:, :3].copy()


def _build_weight_series(table: pd.DataFrame) -> pd.Series:
    table.columns = ["concepto_raw", "durabilidad", "ponderador"]
    table = table.dropna(subset=["ponderador"]).copy()
    table["ponderador"] = pd.to_numeric(table["ponderador"], errors="coerce")
    table = table.dropna(subset=["ponderador"])
    table["concepto"] = table["concepto_raw"].map(_clean_official_label)
    table = table[table["concepto"].astype(str).str.len() > 0]
    # Si hay duplicados de etiqueta, conservamos la fila de mayor ponderador.
    table = table.sort_values("ponderador", ascending=False).drop_duplicates("concepto")
    return table.set_index("concepto")["ponderador"].astype(float)


def _compose_raw(weights: pd.Series) -> pd.DataFrame:
    all_members = {member for members in AGGREGATE_MEMBERS.values() for member in members}
    labels = sorted(set(weights.index.tolist()) | all_members)
    raw = pd.DataFrame(index=labels)
    raw["INPC"] = weights.reindex(raw.index)

    normalized_lookup = {_normalize_text(k): float(v) for k, v in weights.items()}
    for member in all_members:
        if pd.notna(raw.at[member, "INPC"]):
            continue
        maybe = normalized_lookup.get(_normalize_text(member))
        if maybe is not None:
            raw.at[member, "INPC"] = maybe

    for aggregate, members in AGGREGATE_MEMBERS.items():
        raw[aggregate] = pd.NA
        present_members = [m for m in members if m in raw.index]
        raw.loc[present_members, aggregate] = "X"

    # Si el rubro general quedó vacío por cambios de nomenclatura, usar universo.
    if "IndiceGeneral" in raw.columns:
        mask = raw["IndiceGeneral"] == "X"
        if float(raw.loc[mask, "INPC"].fillna(0).sum()) <= 0:
            raw["IndiceGeneral"] = pd.NA
            raw.loc[raw["INPC"].fillna(0) > 0, "IndiceGeneral"] = "X"

    return raw


def _clean_official_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    # El XLSX oficial incluye códigos tipo "01.1.1" o "001" al inicio.
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
    return text.strip()


def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    clean = re.sub(r"[^a-z0-9]+", " ", no_accents)
    return re.sub(r"\s+", " ", clean).strip()
