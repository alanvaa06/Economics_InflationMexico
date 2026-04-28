"""Pruebas del flujo mensual sin fallback a Excel."""
from __future__ import annotations

import pandas as pd
import pytest

from inflacion.data import pipeline


def _sample_monthly_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    return pd.DataFrame({"IndiceGeneral": [132.1, 132.8, 133.4]}, index=idx)


def test_load_local_inpc_reads_parquet_when_available(tmp_path):
    target = tmp_path / "RelevantInflation.parquet"
    expected = _sample_monthly_df()
    expected.to_parquet(target)

    loaded = pipeline.load_local_inpc(path=target)

    pd.testing.assert_frame_equal(loaded, expected, check_freq=False)


def test_load_local_inpc_refreshes_when_cache_missing(monkeypatch, tmp_path):
    target = tmp_path / "RelevantInflation.parquet"
    expected = _sample_monthly_df()
    called: dict[str, bool] = {"refresh": False}

    monkeypatch.setattr(type(pipeline.settings), "require_token", lambda self: "fake-token")

    def _fake_refresh(*, historic: bool, out_path, client=None):
        called["refresh"] = True
        expected.to_parquet(out_path)
        return expected

    monkeypatch.setattr(pipeline, "refresh_inpc", _fake_refresh)

    loaded = pipeline.load_local_inpc(path=target)

    assert called["refresh"] is True
    pd.testing.assert_frame_equal(loaded, expected, check_freq=False)


def test_load_local_inpc_without_token_raises_file_not_found(monkeypatch, tmp_path):
    target = tmp_path / "RelevantInflation.parquet"

    def _raise_token_error():
        raise RuntimeError("token missing")

    monkeypatch.setattr(type(pipeline.settings), "require_token", lambda self: _raise_token_error())

    with pytest.raises(FileNotFoundError, match="INEGI_API_TOKEN"):
        pipeline.load_local_inpc(path=target)
