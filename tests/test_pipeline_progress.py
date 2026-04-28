"""refresh_inpc y refresh_inpc_quincenal pasan progress_cb al cliente."""
from __future__ import annotations

import pandas as pd

from inflacion.data import pipeline


class _StubClient:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[tuple[int, int, str]] = []

    def fetch_many(self, ids, *, historic, progress_cb=None):
        ids_list = list(ids)
        for i, ind in enumerate(ids_list, start=1):
            if progress_cb is not None:
                progress_cb(i, len(ids_list), str(ind))
        return self._df

    def close(self):
        pass


def test_refresh_inpc_forwards_progress_cb(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {"IndiceGeneral": [1.0, 2.0]},
        index=pd.date_range("2024-01-31", periods=2, freq="ME"),
    )

    monkeypatch.setattr(
        "inflacion.data.pipeline.load_series_catalog",
        lambda: pd.Series({"IndiceGeneral": "583766"}, name="Serie"),
    )

    captured: list[tuple[int, int, str]] = []

    pipeline.refresh_inpc(
        historic=False,
        out_path=tmp_path / "out.parquet",
        client=_StubClient(df),
        progress_cb=lambda d, t, n: captured.append((d, t, n)),
    )

    assert len(captured) >= 1
    assert captured[0][1] >= 1  # total > 0


def test_refresh_inpc_quincenal_forwards_progress_cb(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {"IndiceGeneral": [1.0, 2.0]},
        index=pd.date_range("2024-01-31", periods=2, freq="ME"),
    )

    monkeypatch.setattr(
        "inflacion.data.pipeline.load_quincenal_catalog",
        lambda: pd.Series({"IndiceGeneral": "Q1"}, name="Serie"),
    )

    captured: list[tuple[int, int, str]] = []

    pipeline.refresh_inpc_quincenal(
        historic=False,
        out_path=tmp_path / "q.parquet",
        client=_StubClient(df),
        progress_cb=lambda d, t, n: captured.append((d, t, n)),
    )

    assert len(captured) >= 1
