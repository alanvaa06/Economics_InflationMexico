"""Catálogo (mínimo) de series quincenales del INPC publicadas por INEGI.

INEGI sólo publica quincenalmente un puñado de agregados de cabecera. Como BIE
no expone un endpoint público estable de "lista de series", aquí mantenemos un
**seed de candidatos** por nombre amigable. El resolver
(``inflacion.data.quincenal_resolver``) prueba cada candidato en orden contra el
BIE en vivo, clasifica por formato de ``TIME_PERIOD``, y persiste los resueltos
a un sidecar JSON. La función ``load_quincenal_catalog`` lee primero el sidecar;
si no existe, devuelve los IDs explícitos de ``QUINCENAL_HEADLINE_IDS`` (vacío
por defecto: el resolver lo poblará al primer refresh).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from inflacion.config import settings

# Candidatos publicados/conocidos para cada concepto quincenal de cabecera.
# Cada lista se prueba en orden hasta encontrar el primero que responda 200 +
# clasifique como "quincenal". Si todos fallan, ese nombre queda fuera.
QUINCENAL_HEADLINE_CANDIDATES: dict[str, list[str]] = {
    "IndiceGeneral":   ["628194", "910392"],
    "Subyacente":      ["628195", "910393"],
    "NoSubyacente":    ["628196", "910394"],
    "Mercancías":      ["628197", "910395"],
    "Servicios":       ["628198", "910396"],
    "Alimentos":       ["628199", "910397"],
    "NoAlimentos":     ["628200", "910398"],
    "Energéticos":     ["628201", "910399"],
    "TarifasGobierno": ["628202", "910400"],
    "Agropecuarios":   ["628203", "910401"],
}

# IDs explícitos verificados manualmente. Cuando un equipo encuentra el ID real
# y quiere fijarlo, lo agrega aquí. El resolver respetará estas entradas como
# candidatos prioritarios.
QUINCENAL_HEADLINE_IDS: dict[str, str] = {}

SIDECAR_PATH = settings.data_dir / "quincenal_ids_resolved.json"


def load_quincenal_catalog() -> pd.Series:
    """Devuelve el catálogo quincenal como ``Series`` (nombre → id).

    Prioridad de fuentes (la primera no-vacía gana):
    1. Sidecar JSON resuelto (``quincenal_ids_resolved.json``).
    2. ``QUINCENAL_HEADLINE_IDS`` (overrides manuales).
    3. Vacío → el llamador debe correr el resolver.
    """
    sidecar = _read_sidecar()
    if sidecar:
        return pd.Series(sidecar, name="Serie")
    if QUINCENAL_HEADLINE_IDS:
        return pd.Series(QUINCENAL_HEADLINE_IDS, name="Serie")
    return pd.Series(dtype=object, name="Serie")


def _read_sidecar(path: Path | None = None) -> dict[str, str]:
    target = path or SIDECAR_PATH
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        ids = data.get("ids", {})
        return {name: meta["id"] for name, meta in ids.items() if meta.get("id")}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}
