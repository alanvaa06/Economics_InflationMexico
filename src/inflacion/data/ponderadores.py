"""Carga y normaliza la hoja `ObjetoGasto` de `ponderadores.xlsx`.

Reglas de negocio
-----------------
- Cada concepto pertenece a uno o más rubros marcados con ``"X"``.
- Los pesos se rebasan a 100 dentro de cada rubro para calcular contribuciones
  ponderadas (mismo criterio que la metodología INEGI/Banxico).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from inflacion.config import PROJECT_ROOT


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
    """Lee `ponderadores.xlsx` (hoja ObjetoGasto) y arma los diccionarios de pesos.

    Args:
        path: ruta opcional al archivo. Por defecto se busca en la raíz del repo.
    """
    src = Path(path) if path else PROJECT_ROOT / "ponderadores.xlsx"
    raw = pd.read_excel(src, index_col=0, skiprows=9, sheet_name="ObjetoGasto")

    if "INPC" not in raw.columns:
        raise ValueError("ponderadores.xlsx: falta columna 'INPC'")

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
