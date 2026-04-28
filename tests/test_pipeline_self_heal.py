"""Pipeline auto-heal: parquet vacío/corrupto se elimina y se re-descarga."""
from __future__ import annotations

import pandas as pd
import pytest

from inflacion.data import pipeline


def _sample_monthly_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    return pd.DataFrame({"IndiceGeneral": [132.1, 132.8, 133.4]}, index=idx)


def test_load_local_inpc_self_heals_zero_rows(monkeypatch, tmp_path):
    target = tmp_path / "RelevantInflation.parquet"
    # Persistir DF 0×0
    pd.DataFrame().to_parquet(target)

    refreshed = _sample_monthly_df()

    def _fake_refresh(*, historic, out_path, client=None):
        refreshed.to_parquet(out_path)
        return refreshed

    monkeypatch.setattr(type(pipeline.settings), "require_token", lambda self: "fake")
    monkeypatch.setattr(pipeline, "refresh_inpc", _fake_refresh)

    loaded = pipeline.load_local_inpc(path=target)
    pd.testing.assert_frame_equal(loaded, refreshed, check_freq=False)


def test_load_local_inpc_self_heals_zero_columns(monkeypatch, tmp_path):
    target = tmp_path / "RelevantInflation.parquet"
    # DF con índice pero sin columnas
    pd.DataFrame(index=pd.DatetimeIndex([], name="fecha")).to_parquet(target)

    refreshed = _sample_monthly_df()
    monkeypatch.setattr(type(pipeline.settings), "require_token", lambda self: "fake")
    monkeypatch.setattr(
        pipeline, "refresh_inpc",
        lambda *, historic, out_path, client=None: (refreshed.to_parquet(out_path), refreshed)[1],
    )

    loaded = pipeline.load_local_inpc(path=target)
    pd.testing.assert_frame_equal(loaded, refreshed, check_freq=False)


def test_load_local_inpc_quincenal_self_heals_empty(monkeypatch, tmp_path):
    target = tmp_path / "Q.parquet"
    pd.DataFrame().to_parquet(target)

    refreshed = _sample_monthly_df()  # estructura idéntica para test
    monkeypatch.setattr(type(pipeline.settings), "require_token", lambda self: "fake")
    monkeypatch.setattr(
        pipeline, "refresh_inpc_quincenal",
        lambda *, historic, out_path, client=None: (refreshed.to_parquet(out_path), refreshed)[1],
    )

    loaded = pipeline.load_local_inpc_quincenal(path=target)
    assert not loaded.empty


def test_refresh_inpc_does_not_persist_empty(monkeypatch, tmp_path):
    """Si fetch_many levanta (caso "todo falla"), refresh_inpc no debe escribir parquet."""
    from inflacion.inegi.client import BIEError

    class _StubClient:
        def fetch_many(self, *args, **kwargs):
            raise BIEError("ninguna serie")

        def close(self):
            pass

    out = tmp_path / "out.parquet"
    with pytest.raises(BIEError):
        pipeline.refresh_inpc(historic=False, out_path=out, client=_StubClient())

    assert not out.exists()
