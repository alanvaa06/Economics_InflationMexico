"""Pruebas de las funciones de contribución."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inflacion.analytics.contributions import (
    contributions_mom,
    contributions_yoy,
    incidencias,
)


def test_yoy_contribution_matches_closed_form(synthetic_inpc, synthetic_weights):
    contrib = contributions_yoy(synthetic_inpc, synthetic_weights, since=None)

    last = contrib.iloc[-1]
    # tasas YoY teóricas
    expected = pd.Series({"A": 0.10, "B": 0.0, "C": -0.05})
    weights_norm = synthetic_weights / synthetic_weights.sum()
    expected_contrib = expected * weights_norm
    np.testing.assert_allclose(last.sort_index(), expected_contrib.sort_index(), atol=5e-3)


def test_yoy_contribution_first_year_is_nan(synthetic_inpc, synthetic_weights):
    contrib = contributions_yoy(synthetic_inpc, synthetic_weights, since=None)
    # los primeros 12 meses no tienen YoY válido
    assert contrib.iloc[:12].isna().all().all()


def test_mom_contribution_signs(synthetic_inpc, synthetic_weights):
    contrib = contributions_mom(synthetic_inpc, synthetic_weights, since=None)
    # A siempre sube → contribución positiva en MoM (después del 1er punto)
    assert (contrib["A"].dropna() > 0).all()
    # B constante → MoM ≈ 0
    assert contrib["B"].dropna().abs().max() < 1e-9
    # C siempre baja → contribución negativa
    assert (contrib["C"].dropna() < 0).all()


def test_incidencias_returns_top_and_bottom(synthetic_inpc, synthetic_weights):
    contrib = contributions_yoy(synthetic_inpc, synthetic_weights, since=None)
    out = incidencias(contrib, n=2)
    assert "contribucion" in out.columns
    assert "pct_total" in out.columns
    assert out["contribucion"].is_monotonic_decreasing


def test_incidencias_handles_empty():
    out = incidencias(pd.DataFrame(), n=5)
    assert out.empty


def test_yoy_raises_when_no_overlapping_columns(synthetic_inpc):
    with pytest.raises(ValueError):
        contributions_yoy(synthetic_inpc, pd.Series({"Z": 1.0}), since=None)
