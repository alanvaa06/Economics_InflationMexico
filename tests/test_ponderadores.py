"""Pruebas de carga de ponderadores con el archivo real del repo."""
from __future__ import annotations

import pytest

from inflacion.data.ponderadores import load_ponderadores


@pytest.fixture(scope="module")
def p():
    return load_ponderadores()


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
