"""Pruebas de carga de ponderadores oficiales (formato RNM INEGI)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from inflacion.data.ponderadores import load_ponderadores


@pytest.fixture(scope="module")
def official_weights_workbook(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workbook = tmp_path_factory.mktemp("weights") / "inegi_weights.xlsx"
    sheet = pd.DataFrame(
        {
            "concepto": [
                "Canasta INPC",
                "001 Aceites y grasas vegetales comestibles",
                "002 Arroz",
                "003 Agua embotellada",
                "004 Servicio de telefonía móvil",
                "005 Electricidad",
                "006 Vivienda propia",
                "007 Cuotas de autopista",
            ],
            "durabilidad": [
                pd.NA,
                "No duradero",
                "No duradero",
                "No duradero",
                "Servicio",
                "Servicio",
                "Servicio",
                "Servicio",
            ],
            "ponderador": [pd.NA, 0.42, 0.20, 0.73, 1.92, 2.30, 6.80, 0.11],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        sheet.to_excel(writer, sheet_name="Table 1", index=False)
    return workbook


@pytest.fixture(scope="module")
def p(official_weights_workbook: Path):
    return load_ponderadores(path=official_weights_workbook)


def test_index_general_present(p):
    assert "IndiceGeneral" in p.weights_100


def test_each_aggregate_sums_to_100(p):
    for rubro, pesos in p.weights_100.items():
        total = pesos.sum()
        assert abs(total - 100.0) < 1e-6, f"{rubro} suma {total}"


def test_inpc_weights_are_subset_of_general(p):
    inpc_total = p.weights_inpc["IndiceGeneral"].sum()
    # los rubros hijos deben sumar a una fracción del INPC general
    for rubro, pesos in p.weights_inpc.items():
        if rubro == "IndiceGeneral":
            continue
        assert pesos.sum() <= inpc_total + 1e-3


def test_uses_official_labels_without_numeric_codes(p):
    assert "Arroz" in p.raw.index
    assert "001 Arroz" not in p.raw.index
