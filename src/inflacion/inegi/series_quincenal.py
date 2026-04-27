"""Catálogo (mínimo) de series quincenales del INPC publicadas por INEGI.

A diferencia del INPC mensual, INEGI sólo publica quincenalmente un puñado de
agregados de cabecera. Los IDs aquí son **placeholders** que el usuario debe
confirmar en https://www.inegi.org.mx/app/indicadores/ — el cliente
:class:`~inflacion.inegi.client.BIEClient` ignora silenciosamente los IDs que
respondan 404, así que un ID equivocado degrada la serie correspondiente sin
romper el resto del pipeline.

Para llenarlo:
1. Abrir el BIE con filtro "Quincenal".
2. Copiar el ID numérico de cada serie (e.g. ``910420``).
3. Sustituir el placeholder ``""`` correspondiente.
4. Ejecutar ``inflacion refresh --frequency quincenal`` y verificar que el
   parquet resultante tenga ≥ 6 fechas en el último año.
"""
from __future__ import annotations

import pandas as pd

# IDs placeholder; se sobreescriben con los reales antes del primer refresh.
QUINCENAL_HEADLINE_IDS: dict[str, str] = {
    "IndiceGeneral": "",
    "Subyacente": "",
    "NoSubyacente": "",
    "Mercancías": "",
    "Servicios": "",
    "Alimentos": "",
    "NoAlimentos": "",
    "Energéticos": "",
    "TarifasGobierno": "",
    "Agropecuarios": "",
}


def load_quincenal_catalog() -> pd.Series:
    """Devuelve el catálogo quincenal como ``Series`` (nombre → id), sólo IDs no vacíos."""
    pairs = {name: sid for name, sid in QUINCENAL_HEADLINE_IDS.items() if sid}
    return pd.Series(pairs, name="Serie")
