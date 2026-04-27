"""Fixtures de pytest comunes a toda la suite."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

# Aseguramos que ningún test toque la red real con un token "real".
os.environ.setdefault("INEGI_API_TOKEN", "test-token-0000")


@pytest.fixture()
def synthetic_inpc() -> pd.DataFrame:
    """3 conceptos × 36 meses con tasas YoY conocidas:

    - Concepto A: ~10% anual
    - Concepto B: ~0%
    - Concepto C: ~−5%
    """
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    a = 100 * (1.10 ** (pd.Series(range(36)) / 12))
    b = pd.Series([100.0] * 36)
    c = 100 * (0.95 ** (pd.Series(range(36)) / 12))
    return pd.DataFrame({"A": a.values, "B": b.values, "C": c.values}, index=idx)


@pytest.fixture()
def synthetic_weights() -> pd.Series:
    return pd.Series({"A": 50.0, "B": 30.0, "C": 20.0})


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
