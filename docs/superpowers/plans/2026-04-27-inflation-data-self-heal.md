# Inflation data self-heal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hacer que la app de inflación México arranque y descargue datos de INEGI BIE de forma seamless aún cuando el cache está corrupto, el token está caducado, o los IDs quincenales no han sido descubiertos. Sin agregar nuevos datasets.

**Architecture:** Tres capas de robustez sobre el cliente y pipeline existentes: (1) `BIEClient.health_check()` distingue auth-fail (HTTP 400 + "No se encontraron resultados") de "serie no existe", (2) pipeline detecta parquet 0×0 y se autocura, (3) un resolver de IDs quincenales prueba candidatos, clasifica por formato `TIME_PERIOD`, y persiste a sidecar JSON. La UI Streamlit obtiene barra de progreso real vía callback y onboarding sin token.

**Tech Stack:** Python 3.12+, httpx + tenacity (cliente), pandas + pyarrow (parquet), Streamlit (UI), pytest + respx (tests). Paquete `inflacion`. Venv en `.venv/`. Comando de pruebas: `.venv/Scripts/python.exe -m pytest -q`.

**Spec:** [`docs/superpowers/specs/2026-04-27-inflation-data-self-heal-design.md`](../specs/2026-04-27-inflation-data-self-heal-design.md)

---

## Convenciones

- **Comando python**: `.venv/Scripts/python.exe` (Windows). El `python` global no tiene las deps instaladas.
- **Comando ruff**: `.venv/Scripts/python.exe -m ruff check src tests app`.
- **Comando pytest**: `.venv/Scripts/python.exe -m pytest -q`.
- Cada tarea termina con un commit. Mensajes en estilo del repo (commits recientes en `git log --oneline -10`): `feat(scope):`, `fix(scope):`, `refactor(scope):`, `test(scope):`.

---

## Task 1: Función pura `classify_period` en client.py

**Files:**
- Modify: `src/inflacion/inegi/client.py`
- Test: `tests/test_client_classify.py`

- [ ] **Step 1.1: Crear test failing**

Crear `tests/test_client_classify.py`:

```python
"""classify_period: clasifica un payload BIE por formato TIME_PERIOD."""
from __future__ import annotations

import pytest

from inflacion.inegi.client import classify_period


def _payload(time_period: str) -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": time_period, "OBS_VALUE": "1.0"}]}]}


@pytest.mark.parametrize(
    "tp,expected",
    [
        ("2024/03", "mensual"),
        ("2026/04", "mensual"),
        ("2024/03/Q1", "quincenal"),
        ("2024/03/Q2", "quincenal"),
        ("2024/03/1", "quincenal"),
        ("2024/03/2", "quincenal"),
        ("", "desconocido"),
        ("2024", "desconocido"),
        ("2024/03/Q3", "desconocido"),
    ],
)
def test_classify_period(tp, expected):
    assert classify_period(_payload(tp)) == expected


def test_classify_period_empty_observations_is_desconocido():
    assert classify_period({"Series": [{"OBSERVATIONS": []}]}) == "desconocido"


def test_classify_period_malformed_payload_is_desconocido():
    assert classify_period({}) == "desconocido"
    assert classify_period({"Series": []}) == "desconocido"
```

- [ ] **Step 1.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client_classify.py -v
```

Expected: ImportError "cannot import name 'classify_period'".

- [ ] **Step 1.3: Implementar `classify_period`**

En `src/inflacion/inegi/client.py`, después de la función `_redact` (alrededor de línea 49), añadir:

```python
def classify_period(payload: dict[str, Any]) -> str:
    """Clasifica un payload BIE por su primer ``TIME_PERIOD``.

    Retorna ``"mensual"`` (formato ``YYYY/MM``), ``"quincenal"``
    (``YYYY/MM/Q1|Q2`` o ``YYYY/MM/1|2``), o ``"desconocido"``.
    Tolera payloads malformados devolviendo ``"desconocido"``.
    """
    try:
        observations = payload["Series"][0]["OBSERVATIONS"]
    except (KeyError, IndexError, TypeError):
        return "desconocido"
    if not observations:
        return "desconocido"
    tp = observations[0].get("TIME_PERIOD", "")
    parts = tp.split("/") if tp else []
    if len(parts) == 2:
        return "mensual"
    if len(parts) == 3 and _QUINCENA_RE.match(parts[2].strip()):
        return "quincenal"
    return "desconocido"
```

- [ ] **Step 1.4: Verificar que pasa**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client_classify.py -v
```

Expected: `9 passed`.

- [ ] **Step 1.5: Verificar que el resto sigue verde**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: 46 tests previos + 11 nuevos (9 parametrizados de `classify_period` + 2 casos extra) = 57 passed. Confirmar el número exacto al ejecutar.

- [ ] **Step 1.6: Commit**

```bash
git add src/inflacion/inegi/client.py tests/test_client_classify.py
git commit -m "feat(client): classify_period clasifica payload BIE por TIME_PERIOD"
```

---

## Task 2: `BIEClient.health_check()` distingue auth-fail

**Files:**
- Modify: `src/inflacion/inegi/client.py`
- Test: `tests/test_client_health.py`

- [ ] **Step 2.1: Crear test failing**

Crear `tests/test_client_health.py`:

```python
"""BIEClient.health_check: valida token contra ID maestro conocido."""
from __future__ import annotations

import httpx
import pytest
import respx

from inflacion.inegi.client import BIEClient, BIEError, MissingTokenError

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"
HEALTH_ID = "583766"  # IndiceGeneral mensual; mismo valor usado en client.py


def _ok_payload() -> dict:
    return {
        "Series": [
            {"OBSERVATIONS": [{"TIME_PERIOD": "2024/01", "OBS_VALUE": "133.555"}]}
        ]
    }


@respx.mock
def test_health_check_ok_when_master_id_responds_200():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(200, json=_ok_payload()))
    with BIEClient(token=TOKEN) as client:
        client.health_check()  # no debe levantar


@respx.mock
def test_health_check_400_with_no_se_encontraron_raises_missing_token():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    body = '["ErrorInfo:No se encontraron resultados","ErrorDetails:No se encontraron resultados","ErrorCode:100"]'
    respx.get(url).mock(return_value=httpx.Response(400, text=body))
    with BIEClient(token=TOKEN) as client, pytest.raises(MissingTokenError, match="rechazó el token"):
        client.health_check()


@respx.mock
def test_health_check_500_raises_bie_error():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(500, text="server error"))
    with BIEClient(token=TOKEN) as client, pytest.raises(BIEError):
        client.health_check()


@respx.mock
def test_health_check_400_with_other_body_raises_bie_error():
    url = f"{BASE}/INDICATOR/{HEALTH_ID}/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url).mock(return_value=httpx.Response(400, text="Some other 400"))
    with BIEClient(token=TOKEN) as client, pytest.raises(BIEError):
        client.health_check()
```

- [ ] **Step 2.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client_health.py -v
```

Expected: AttributeError "BIEClient has no attribute 'health_check'".

- [ ] **Step 2.3: Implementar `health_check`**

En `src/inflacion/inegi/client.py`:

a) Añadir constante de clase a `BIEClient` (justo después de `_TOKEN_RE`):

```python
HEALTH_PROBE_ID = "583766"  # IndiceGeneral mensual (INPC base 2018)
```

b) Modificar `_get` para preservar el body en errores 4xx (necesitamos inspeccionar el cuerpo en health_check):

Reemplazar el bloque actual (líneas 137-138 aprox):

```python
        if response.status_code != 200:
            raise BIEError(f"Respuesta inesperada del BIE: HTTP {response.status_code}")
```

por:

```python
        if response.status_code != 200:
            raise BIEError(
                f"Respuesta inesperada del BIE: HTTP {response.status_code} body={response.text[:200]!r}"
            )
```

c) Añadir el método `health_check` después de `fetch_many`:

```python
    def health_check(self) -> None:
        """Verifica que el token sea aceptado por INEGI sondeando un ID maestro.

        Raises:
            MissingTokenError: el token es sintácticamente válido pero INEGI lo rechaza
                (HTTP 400 con cuerpo conteniendo "No se encontraron resultados").
            BIEError: cualquier otra falla (5xx, 400 con cuerpo distinto, etc.).
        """
        url = self._url(self.HEALTH_PROBE_ID, historic=False)
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise BIEError(f"Error de red en health_check: {_redact(str(exc))}") from None
        if response.status_code == 200:
            return
        body = response.text
        if response.status_code == 400 and "No se encontraron resultados" in body:
            raise MissingTokenError(
                "INEGI rechazó el token (HTTP 400). Renueva en "
                "https://www.inegi.org.mx/servicios/api_indicadores.html y actualiza .env."
            )
        raise BIEError(
            f"health_check falló: HTTP {response.status_code} body={body[:200]!r}"
        )
```

- [ ] **Step 2.4: Verificar que pasa**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client_health.py -v
```

Expected: `4 passed`.

- [ ] **Step 2.5: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todos los tests previos + los 4 nuevos pasan.

- [ ] **Step 2.6: Commit**

```bash
git add src/inflacion/inegi/client.py tests/test_client_health.py
git commit -m "feat(client): health_check distingue auth-fail INEGI (HTTP 400)"
```

---

## Task 3: `fetch_many` no devuelve DataFrame vacío silenciosamente

**Files:**
- Modify: `src/inflacion/inegi/client.py`
- Test: `tests/test_client.py` (extender)

- [ ] **Step 3.1: Añadir test failing al final de `tests/test_client.py`**

```python
@respx.mock
def test_fetch_many_raises_when_all_series_fail():
    """Si todas las series fallan, fetch_many debe levantar (no devolver DF vacío)."""
    url1 = f"{BASE}/INDICATOR/1/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    url2 = f"{BASE}/INDICATOR/2/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url1).mock(return_value=httpx.Response(403, text="x"))
    respx.get(url2).mock(return_value=httpx.Response(403, text="x"))

    with BIEClient(token=TOKEN) as client, pytest.raises(BIEError, match="ninguna serie"):
        client.fetch_many([1, 2])
```

- [ ] **Step 3.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client.py::test_fetch_many_raises_when_all_series_fail -v
```

Expected: FAIL — el método actual devuelve DF vacío en vez de levantar.

- [ ] **Step 3.3: Modificar `fetch_many` en `src/inflacion/inegi/client.py`**

Reemplazar el bloque actual:

```python
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()
```

por:

```python
        if not frames:
            raise BIEError(
                "BIE: ninguna serie respondió OK. Verifica el token o el catálogo."
            )
        return pd.concat(frames, axis=1).sort_index()
```

- [ ] **Step 3.4: Verificar que el nuevo test pasa**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client.py -v
```

Expected: todos los tests de test_client.py pasan, incluido el nuevo.

- [ ] **Step 3.5: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde. (Nota: si algún test del pipeline asumía DF vacío como OK, fallará — corregir en Tasks 4-6.)

- [ ] **Step 3.6: Commit**

```bash
git add src/inflacion/inegi/client.py tests/test_client.py
git commit -m "fix(client): fetch_many levanta BIEError si todas las series fallan"
```

---

## Task 4: `fetch_many` acepta progress callback

**Files:**
- Modify: `src/inflacion/inegi/client.py`
- Test: `tests/test_client.py` (extender)

- [ ] **Step 4.1: Añadir test failing**

Añadir al final de `tests/test_client.py`:

```python
@respx.mock
def test_fetch_many_invokes_progress_cb_in_order():
    """progress_cb se llama tras cada serie con (done, total, name)."""
    url1 = f"{BASE}/INDICATOR/1/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    url2 = f"{BASE}/INDICATOR/2/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url1).mock(return_value=httpx.Response(200, json=_ok_payload()))
    respx.get(url2).mock(return_value=httpx.Response(200, json=_ok_payload()))

    calls: list[tuple[int, int, str]] = []

    def cb(done: int, total: int, name: str) -> None:
        calls.append((done, total, name))

    with BIEClient(token=TOKEN) as client:
        client.fetch_many([1, 2], progress_cb=cb)

    assert calls == [(1, 2, "1"), (2, 2, "2")]


@respx.mock
def test_fetch_many_progress_cb_called_even_on_failures():
    """progress_cb se llama también cuando una serie falla individualmente."""
    url1 = f"{BASE}/INDICATOR/1/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    url2 = f"{BASE}/INDICATOR/2/es/0700/false/BIE/2.0/{TOKEN}?type=json"
    respx.get(url1).mock(return_value=httpx.Response(200, json=_ok_payload()))
    respx.get(url2).mock(return_value=httpx.Response(403, text="bad"))

    calls: list[int] = []
    with BIEClient(token=TOKEN) as client:
        client.fetch_many([1, 2], progress_cb=lambda d, t, n: calls.append(d))

    assert calls == [1, 2]
```

- [ ] **Step 4.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client.py::test_fetch_many_invokes_progress_cb_in_order -v
```

Expected: TypeError "unexpected keyword argument 'progress_cb'".

- [ ] **Step 4.3: Modificar firma e implementación de `fetch_many`**

En `src/inflacion/inegi/client.py`, reemplazar la firma actual y el cuerpo:

```python
    def fetch_many(
        self,
        indicators: Iterable[str | int],
        *,
        historic: bool = False,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        """Une varias series por columna (clave = id como string).

        Args:
            indicators: IDs BIE a descargar.
            historic: ``True`` trae serie completa, ``False`` reciente.
            progress_cb: opcional. Se invoca tras procesar cada serie con
                ``(done, total, name)``. ``done`` empieza en 1, ``name`` es el ID.
        """
        ids = [str(ind) for ind in indicators]
        total = len(ids)
        frames: list[pd.DataFrame] = []
        for index, ind in enumerate(ids, start=1):
            try:
                df = self.fetch_series(ind, historic=historic).rename(columns={"valor": ind})
                frames.append(df)
            except BIEError as exc:
                logger.warning("No se pudo descargar %s: %s", ind, _redact(str(exc)))
            if progress_cb is not None:
                progress_cb(index, total, ind)
        if not frames:
            raise BIEError(
                "BIE: ninguna serie respondió OK. Verifica el token o el catálogo."
            )
        return pd.concat(frames, axis=1).sort_index()
```

Y al inicio del archivo, añadir `Callable` al import de typing:

```python
from collections.abc import Callable, Iterable
```

(Si `Iterable` ya está, asegurarse de añadir `Callable` al mismo import.)

- [ ] **Step 4.4: Verificar que los tests pasan**

```bash
.venv/Scripts/python.exe -m pytest tests/test_client.py -v
```

Expected: todos verdes.

- [ ] **Step 4.5: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde.

- [ ] **Step 4.6: Commit**

```bash
git add src/inflacion/inegi/client.py tests/test_client.py
git commit -m "feat(client): fetch_many soporta progress_cb"
```

---

## Task 5: Auto-heal de parquet vacío en `load_local_inpc` y `load_local_inpc_quincenal`

**Files:**
- Modify: `src/inflacion/data/pipeline.py`
- Test: `tests/test_pipeline_self_heal.py`

- [ ] **Step 5.1: Crear test failing**

Crear `tests/test_pipeline_self_heal.py`:

```python
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
```

- [ ] **Step 5.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_self_heal.py -v
```

Expected: tests fallan (no auto-heal aún) o errors si la lógica no propaga `BIEError`.

- [ ] **Step 5.3: Implementar auto-heal en `load_local_inpc` y `load_local_inpc_quincenal`**

En `src/inflacion/data/pipeline.py`, **reemplazar** la función `load_local_inpc` completa por:

```python
def load_local_inpc(path: Path | None = None) -> pd.DataFrame:
    """Carga el INPC local (parquet) o lo descarga de INEGI si falta o está vacío."""
    target = Path(path) if path else settings.data_dir / "RelevantInflation.parquet"
    if target.exists():
        df = pd.read_parquet(target)
        if df.empty or df.shape[1] == 0:
            logger.warning("Parquet vacío en %s — descartando y re-descargando.", target)
            target.unlink()
        else:
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

    try:
        settings.require_token()
    except RuntimeError as exc:
        raise FileNotFoundError(
            "No se encontró cache local del INPC y falta `INEGI_API_TOKEN`. "
            "Configura `.env` o ejecuta `inflacion refresh` en un entorno con token."
        ) from exc

    logger.info("INPC local no encontrado; descargando desde INEGI BIE a %s", target)
    refresh_inpc(historic=True, out_path=target)
    if target.exists():
        df = pd.read_parquet(target)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    raise FileNotFoundError("No se pudo generar `RelevantInflation.parquet` desde INEGI.")
```

Y de forma análoga, **reemplazar** `load_local_inpc_quincenal` por:

```python
def load_local_inpc_quincenal(path: Path | None = None) -> pd.DataFrame:
    """Carga el INPC quincenal local (parquet) o lo descarga de INEGI si falta o está vacío."""
    target = Path(path) if path else settings.data_dir / "RelevantInflation_Q.parquet"
    if target.exists():
        df = pd.read_parquet(target)
        if df.empty or df.shape[1] == 0:
            logger.warning("Parquet quincenal vacío en %s — descartando.", target)
            target.unlink()
        else:
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

    try:
        settings.require_token()
    except RuntimeError as exc:
        raise FileNotFoundError(
            "No se encontró cache local quincenal y falta `INEGI_API_TOKEN`. "
            "Configura `.env` o ejecuta `inflacion refresh --frequency quincenal`."
        ) from exc

    logger.info("INPC quincenal local no encontrado; descargando desde INEGI BIE a %s", target)
    refresh_inpc_quincenal(historic=True, out_path=target)
    if target.exists():
        df = pd.read_parquet(target)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    raise FileNotFoundError("No se pudo generar `RelevantInflation_Q.parquet` desde INEGI.")
```

- [ ] **Step 5.4: Verificar tests del archivo**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_self_heal.py -v
```

Expected: 4 passed.

- [ ] **Step 5.5: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde.

- [ ] **Step 5.6: Commit**

```bash
git add src/inflacion/data/pipeline.py tests/test_pipeline_self_heal.py
git commit -m "fix(pipeline): auto-heal de parquet INPC vacío en load_local_inpc(_quincenal)"
```

---

## Task 6: `refresh_inpc(_quincenal)` acepta progress callback

**Files:**
- Modify: `src/inflacion/data/pipeline.py`
- Test: `tests/test_pipeline_progress.py`

- [ ] **Step 6.1: Crear test failing**

Crear `tests/test_pipeline_progress.py`:

```python
"""refresh_inpc y refresh_inpc_quincenal pasan progress_cb al cliente."""
from __future__ import annotations

import pandas as pd
import pytest

from inflacion.data import pipeline


class _StubClient:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[tuple[int, int, str]] = []

    def fetch_many(self, ids, *, historic, progress_cb=None):
        # Simular 2 series con progreso
        for i, ind in enumerate(ids, start=1):
            if progress_cb is not None:
                progress_cb(i, len(list(ids)) if not isinstance(ids, list) else len(ids), str(ind))
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
```

- [ ] **Step 6.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_progress.py -v
```

Expected: TypeError 'progress_cb' no soportado.

- [ ] **Step 6.3: Modificar `refresh_inpc` y `refresh_inpc_quincenal`**

En `src/inflacion/data/pipeline.py`:

a) Imports al tope:

```python
from collections.abc import Callable
```

b) Modificar firma y cuerpo de `refresh_inpc`:

```python
def refresh_inpc(
    *,
    historic: bool = True,
    out_path: Path | None = None,
    client: BIEClient | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Descarga todas las series del catálogo y devuelve un DataFrame ancho.

    Args:
        historic: ``True`` trae la serie completa; ``False`` solo lo más reciente.
        out_path: si se indica, guarda en parquet (recomendado) o xlsx.
        client: cliente a reutilizar (útil para tests o múltiples llamadas).
        progress_cb: opcional. Llamado tras cada serie como ``(done, total, id)``.
    """
    catalog = load_series_catalog()
    owns_client = client is None
    bie = client or BIEClient()
    try:
        wide = bie.fetch_many(catalog.tolist(), historic=historic, progress_cb=progress_cb)
    finally:
        if owns_client:
            bie.close()

    id_to_name = {str(v): str(k) for k, v in catalog.items()}
    wide = wide.rename(columns=id_to_name)

    if out_path:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".parquet":
            wide.to_parquet(target)
        else:
            wide.to_excel(target)
        logger.info("INPC guardado en %s (%d filas, %d columnas)", target, *wide.shape)
    return wide
```

c) Modificar análogamente `refresh_inpc_quincenal` añadiendo `progress_cb` a la firma y pasándolo a `bie.fetch_many(...)`.

- [ ] **Step 6.4: Verificar tests del archivo**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_progress.py -v
```

Expected: 2 passed.

- [ ] **Step 6.5: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde.

- [ ] **Step 6.6: Commit**

```bash
git add src/inflacion/data/pipeline.py tests/test_pipeline_progress.py
git commit -m "feat(pipeline): refresh_inpc(_quincenal) reenvía progress_cb"
```

---

## Task 7: Seed de candidatos quincenales y resolver

**Files:**
- Modify: `src/inflacion/inegi/series_quincenal.py`
- Modify: `tests/test_pipeline_quincenal.py` (1 test existente cambia)
- Create: `src/inflacion/data/quincenal_resolver.py`
- Create: `tests/test_quincenal_resolver.py`

- [ ] **Step 7.1: Refactor `series_quincenal.py` — añadir candidatos sin romper API existente**

Reemplazar el contenido completo de `src/inflacion/inegi/series_quincenal.py` por:

```python
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
```

- [ ] **Step 7.2: Actualizar test obsoleto en `tests/test_pipeline_quincenal.py`**

El test `test_quincenal_catalog_has_expected_keys` espera que `QUINCENAL_HEADLINE_IDS` tenga ciertas keys. Ahora ese dict puede estar vacío y los nombres canónicos viven en `QUINCENAL_HEADLINE_CANDIDATES`. Reemplazar ese test por:

```python
def test_quincenal_candidates_has_expected_keys():
    """Los nombres canónicos de cabecera están en CANDIDATES (los IDs concretos los resuelve el resolver)."""
    from inflacion.inegi.series_quincenal import QUINCENAL_HEADLINE_CANDIDATES

    expected = {
        "IndiceGeneral",
        "Subyacente",
        "NoSubyacente",
        "Mercancías",
        "Servicios",
    }
    assert expected.issubset(QUINCENAL_HEADLINE_CANDIDATES.keys())
    # Cada candidato debe ser una lista no-vacía de strings
    for name, ids in QUINCENAL_HEADLINE_CANDIDATES.items():
        assert isinstance(ids, list) and ids, f"{name} sin candidatos"
        assert all(isinstance(i, str) and i for i in ids), f"{name} con candidatos vacíos"
```

Actualizar el import al tope del archivo de tests:

```python
from inflacion.inegi.series_quincenal import QUINCENAL_HEADLINE_CANDIDATES, load_quincenal_catalog
```

(Eliminar el import de `QUINCENAL_HEADLINE_IDS` si ya no se usa en el archivo.)

- [ ] **Step 7.3: Verificar que las pruebas existentes siguen verdes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_quincenal.py -v
```

Expected: todo verde (la prueba `test_refresh_quincenal_partial_success` no toca el catálogo de cabecera, monkeypatcha `load_quincenal_catalog`).

- [ ] **Step 7.4: Crear test failing del resolver**

Crear `tests/test_quincenal_resolver.py`:

```python
"""resolve_quincenal_ids: sondea candidatos y persiste sidecar."""
from __future__ import annotations

import json

import httpx
import respx

from inflacion.data.quincenal_resolver import resolve_quincenal_ids
from inflacion.inegi.client import BIEClient

BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
TOKEN = "test-token-0000"


def _q_payload(tp: str = "2024/03/Q1") -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": tp, "OBS_VALUE": "133.0"}]}]}


def _m_payload(tp: str = "2024/03") -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": tp, "OBS_VALUE": "133.0"}]}]}


@respx.mock
def test_resolver_picks_first_quincenal_candidate(tmp_path):
    candidates = {
        "IndiceGeneral": ["AAA", "BBB"],
        "Subyacente":    ["CCC"],
    }
    # AAA responde mensual → no
    respx.get(f"{BASE}/INDICATOR/AAA/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_m_payload())
    )
    # BBB responde quincenal → sí
    respx.get(f"{BASE}/INDICATOR/BBB/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    # CCC responde quincenal → sí
    respx.get(f"{BASE}/INDICATOR/CCC/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    sidecar = tmp_path / "ids.json"
    with BIEClient(token=TOKEN) as client:
        resolved = resolve_quincenal_ids(client, candidates, sidecar_path=sidecar)

    assert resolved == {"IndiceGeneral": "BBB", "Subyacente": "CCC"}
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "verified_at" in saved
    assert saved["ids"]["IndiceGeneral"]["id"] == "BBB"
    assert saved["ids"]["Subyacente"]["id"] == "CCC"


@respx.mock
def test_resolver_skips_concept_when_all_candidates_fail(tmp_path):
    candidates = {"IndiceGeneral": ["AAA"], "Subyacente": ["BBB"]}
    respx.get(f"{BASE}/INDICATOR/AAA/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(403, text="bad")
    )
    respx.get(f"{BASE}/INDICATOR/BBB/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    sidecar = tmp_path / "ids.json"
    with BIEClient(token=TOKEN) as client:
        resolved = resolve_quincenal_ids(client, candidates, sidecar_path=sidecar)

    assert resolved == {"Subyacente": "BBB"}


@respx.mock
def test_resolver_invokes_progress_cb(tmp_path):
    candidates = {"IndiceGeneral": ["AAA"], "Subyacente": ["BBB"]}
    respx.get(f"{BASE}/INDICATOR/AAA/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    respx.get(f"{BASE}/INDICATOR/BBB/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    progress: list[tuple[int, int, str]] = []
    with BIEClient(token=TOKEN) as client:
        resolve_quincenal_ids(
            client, candidates, sidecar_path=tmp_path / "ids.json",
            progress_cb=lambda d, t, n: progress.append((d, t, n)),
        )

    assert [c[0] for c in progress] == [1, 2]
    assert progress[0][1] == 2
```

- [ ] **Step 7.5: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_quincenal_resolver.py -v
```

Expected: ImportError (módulo no existe).

- [ ] **Step 7.6: Implementar `quincenal_resolver.py`**

Crear `src/inflacion/data/quincenal_resolver.py`:

```python
"""Resolver de IDs quincenales: sondea candidatos y persiste un sidecar JSON."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from inflacion.inegi.client import BIEClient, BIEError, classify_period

logger = logging.getLogger(__name__)


def resolve_quincenal_ids(
    client: BIEClient,
    candidates: Mapping[str, list[str]],
    *,
    sidecar_path: Path,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    """Para cada nombre, prueba los IDs candidatos y se queda con el primero quincenal.

    Persiste los resueltos en ``sidecar_path`` (JSON con metadata) y devuelve el
    mapping ``{nombre: id}``. Nombres sin candidato exitoso quedan fuera.
    """
    resolved: dict[str, dict[str, str]] = {}
    total = len(candidates)
    for index, (name, ids) in enumerate(candidates.items(), start=1):
        winner = _try_candidates(client, ids)
        if winner is not None:
            ind, sample_period = winner
            resolved[name] = {
                "id": ind,
                "period_format": "YYYY/MM/Qn",
                "last_period": sample_period,
            }
            logger.info("Quincenal '%s' resuelto a ID %s", name, ind)
        else:
            logger.warning("Quincenal '%s' no resuelto (todos los candidatos fallaron)", name)
        if progress_cb is not None:
            progress_cb(index, total, name)

    _write_sidecar(sidecar_path, resolved)
    return {name: meta["id"] for name, meta in resolved.items()}


def _try_candidates(client: BIEClient, ids: list[str]) -> tuple[str, str] | None:
    """Devuelve ``(id, último TIME_PERIOD)`` del primer ID que responda quincenal."""
    for ind in ids:
        url = client._url(ind, historic=False)  # noqa: SLF001
        try:
            payload = client._get(url)  # noqa: SLF001
        except BIEError as exc:
            logger.debug("Candidato %s falló: %s", ind, exc)
            continue
        if classify_period(payload) != "quincenal":
            logger.debug("Candidato %s no es quincenal", ind)
            continue
        try:
            sample = payload["Series"][0]["OBSERVATIONS"][-1]["TIME_PERIOD"]
        except (KeyError, IndexError):
            sample = ""
        return ind, sample
    return None


def _write_sidecar(path: Path, resolved: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ids": resolved,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 7.7: Verificar tests del resolver**

```bash
.venv/Scripts/python.exe -m pytest tests/test_quincenal_resolver.py -v
```

Expected: 3 passed.

- [ ] **Step 7.8: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde.

- [ ] **Step 7.9: Commit**

```bash
git add src/inflacion/inegi/series_quincenal.py src/inflacion/data/quincenal_resolver.py tests/test_pipeline_quincenal.py tests/test_quincenal_resolver.py
git commit -m "feat(quincenal): seed de candidatos + resolver con sidecar JSON"
```

---

## Task 8: Wiring del resolver en el pipeline quincenal

**Files:**
- Modify: `src/inflacion/data/pipeline.py`
- Test: `tests/test_pipeline_quincenal.py` (extender)

- [ ] **Step 8.1: Crear test failing**

Añadir al final de `tests/test_pipeline_quincenal.py`:

```python
@respx.mock
def test_refresh_quincenal_runs_resolver_when_catalog_empty(monkeypatch, tmp_path):
    """Si load_quincenal_catalog devuelve Series vacía, refresh corre el resolver."""
    import pandas as pd
    from inflacion.data.pipeline import refresh_inpc_quincenal_with_discovery
    from inflacion.inegi.client import BIEClient

    monkeypatch.setattr(
        "inflacion.data.pipeline.load_quincenal_catalog",
        lambda: pd.Series(dtype=object, name="Serie"),
    )
    monkeypatch.setattr(
        "inflacion.inegi.series_quincenal.QUINCENAL_HEADLINE_CANDIDATES",
        {"IndiceGeneral": ["XYZ"]},
    )

    # XYZ responde quincenal
    respx.get(f"{BASE}/INDICATOR/XYZ/es/0700/false/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )
    # Y luego XYZ se descarga histórico
    respx.get(f"{BASE}/INDICATOR/XYZ/es/0700/true/BIE/2.0/{TOKEN}?type=json").mock(
        return_value=httpx.Response(200, json=_q_payload())
    )

    out = tmp_path / "Q.parquet"
    sidecar = tmp_path / "ids.json"
    df = refresh_inpc_quincenal_with_discovery(
        client=BIEClient(token=TOKEN),
        out_path=out,
        sidecar_path=sidecar,
        historic=True,
    )
    assert "IndiceGeneral" in df.columns
    assert sidecar.exists()
    assert out.exists()
```

- [ ] **Step 8.2: Verificar que falla**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_quincenal.py::test_refresh_quincenal_runs_resolver_when_catalog_empty -v
```

Expected: ImportError.

- [ ] **Step 8.3: Añadir `refresh_inpc_quincenal_with_discovery`**

En `src/inflacion/data/pipeline.py`, añadir después de `refresh_inpc_quincenal`:

```python
def refresh_inpc_quincenal_with_discovery(
    *,
    client: BIEClient | None = None,
    out_path: Path | None = None,
    sidecar_path: Path | None = None,
    historic: bool = True,
    discovery_progress_cb: Callable[[int, int, str], None] | None = None,
    fetch_progress_cb: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Resuelve IDs quincenales (si hace falta) y descarga el histórico.

    Si ``load_quincenal_catalog`` ya devuelve un mapping no-vacío (sidecar previo
    o overrides manuales), salta el resolver. Si no, sondea
    ``QUINCENAL_HEADLINE_CANDIDATES`` y persiste el sidecar antes del fetch.
    """
    from inflacion.data.quincenal_resolver import resolve_quincenal_ids
    from inflacion.inegi.series_quincenal import (
        QUINCENAL_HEADLINE_CANDIDATES,
        SIDECAR_PATH,
    )

    sidecar = sidecar_path or SIDECAR_PATH
    catalog = load_quincenal_catalog()
    owns_client = client is None
    bie = client or BIEClient()
    try:
        if catalog.empty:
            resolved = resolve_quincenal_ids(
                bie,
                QUINCENAL_HEADLINE_CANDIDATES,
                sidecar_path=sidecar,
                progress_cb=discovery_progress_cb,
            )
            if not resolved:
                raise RuntimeError(
                    "Ningún candidato quincenal respondió. Revisa token o agrega IDs en "
                    "QUINCENAL_HEADLINE_IDS / QUINCENAL_HEADLINE_CANDIDATES."
                )
            catalog = pd.Series(resolved, name="Serie")

        wide = bie.fetch_many(
            catalog.tolist(), historic=historic, progress_cb=fetch_progress_cb
        )
    finally:
        if owns_client:
            bie.close()

    id_to_name = {str(v): str(k) for k, v in catalog.items()}
    wide = wide.rename(columns=id_to_name)

    if out_path:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".parquet":
            wide.to_parquet(target)
        else:
            wide.to_excel(target)
        logger.info(
            "INPC quincenal guardado en %s (%d filas, %d columnas)", target, *wide.shape
        )
    return wide
```

- [ ] **Step 8.4: Verificar test del wiring**

```bash
.venv/Scripts/python.exe -m pytest tests/test_pipeline_quincenal.py::test_refresh_quincenal_runs_resolver_when_catalog_empty -v
```

Expected: pasa.

- [ ] **Step 8.5: Verificar suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde.

- [ ] **Step 8.6: Commit**

```bash
git add src/inflacion/data/pipeline.py tests/test_pipeline_quincenal.py
git commit -m "feat(pipeline): refresh_inpc_quincenal_with_discovery integra el resolver"
```

---

## Task 9: Streamlit — onboarding sin token, no `st.stop` en arranque

**Files:**
- Modify: `app/streamlit_app.py`

(Sin tests automatizados — Streamlit no se testea fácilmente. Validación manual al final.)

- [ ] **Step 9.1: Reemplazar el bloque de carga inicial en `app/streamlit_app.py`**

Localizar (aprox. líneas 67-75):

```python
try:
    inpc, ponderadores = _load_monthly()
except FileNotFoundError as exc:
    st.error(
        "No se pudo cargar el INPC mensual desde cache local ni descargarlo de INEGI. "
        "Verifica `.env` con `INEGI_API_TOKEN` y vuelve a intentar."
    )
    st.exception(exc)
    st.stop()
```

Reemplazar por:

```python
def _render_onboarding(reason: str, exc: Exception | None = None) -> None:
    """Renderiza la página de bienvenida cuando no hay datos para mostrar."""
    st.title("📈 Inflación México — primera ejecución")
    st.markdown(
        """
        Esta app analiza el INPC publicado por **INEGI** (Banco de Indicadores
        Económicos). Para descargar datos, necesitas un token gratuito.

        **Pasos:**
        1. Solicita un token en [INEGI API Indicadores](https://www.inegi.org.mx/servicios/api_indicadores.html).
        2. Copia `.env.example` a `.env` y pega tu token en `INEGI_API_TOKEN`.
        3. Recarga esta página.
        """
    )
    st.warning(reason)
    if exc is not None:
        with st.expander("Detalles técnicos"):
            st.exception(exc)


# Nota: este import al pie del archivo evita el módulo-top noqa; ruff no
# se queja porque está antes del primer uso. Si prefieres ponerlo arriba con
# los demás imports, hazlo y verifica que `ruff check` siga limpio.
from inflacion.inegi.client import MissingTokenError  # noqa: E402

try:
    inpc, ponderadores = _load_monthly()
except MissingTokenError as exc:
    _render_onboarding(
        "Tu `INEGI_API_TOKEN` actual fue rechazado por INEGI (HTTP 400). "
        "Probablemente fue rotado o caducó. Renuévalo y reinicia.",
        exc,
    )
    st.stop()
except FileNotFoundError as exc:
    _render_onboarding(
        "No se encontró cache local del INPC y no hay token configurado.", exc,
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error("Falla inesperada al cargar el INPC mensual.")
    st.exception(exc)
    st.stop()
```

- [ ] **Step 9.2: Sanity check**

```bash
.venv/Scripts/python.exe -m ruff check app/streamlit_app.py
```

Expected: limpio (sin errores). Si reporta `BLE001`, suprimir con `# noqa: BLE001` ya está incluido.

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde.

- [ ] **Step 9.3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(app): onboarding amigable cuando falta o falla token INEGI"
```

---

## Task 10: Streamlit — barra de progreso real durante refresh mensual

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 10.1: Localizar el botón "Descargar desde INEGI (mensual)" en `app/streamlit_app.py`**

Aprox. líneas 134-146:

```python
    if col_b1.button("Descargar desde INEGI (mensual)"):
        try:
            settings.require_token()
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            with st.status("Descargando mensual desde INEGI…"):
                refresh_inpc(
                    historic=True,
                    out_path=settings.data_dir / "RelevantInflation.parquet",
                )
            st.cache_data.clear()
            st.rerun()
```

Reemplazar por:

```python
    if col_b1.button("Descargar desde INEGI (mensual)"):
        try:
            settings.require_token()
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            progress = st.progress(0, text="Descargando 0/?")
            status_box = st.empty()

            def _cb(done: int, total: int, name: str) -> None:
                progress.progress(done / total, text=f"Descargando {done}/{total}: {name[:40]}")

            try:
                from inflacion.inegi.client import MissingTokenError, BIEClient
                client = BIEClient()
                try:
                    client.health_check()
                except MissingTokenError as exc:
                    st.error(str(exc))
                    client.close()
                    st.stop()
                refresh_inpc(
                    historic=True,
                    out_path=settings.data_dir / "RelevantInflation.parquet",
                    client=client,
                    progress_cb=_cb,
                )
                status_box.success("Descarga mensual completada.")
            except Exception as exc:  # noqa: BLE001
                status_box.error(f"Falla al descargar: {exc}")
                st.stop()
            st.cache_data.clear()
            st.rerun()
```

- [ ] **Step 10.2: Verificar lint**

```bash
.venv/Scripts/python.exe -m ruff check app/streamlit_app.py
```

Expected: limpio.

- [ ] **Step 10.3: Verificar tests**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde (no rompimos nada).

- [ ] **Step 10.4: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(app): barra de progreso real X/N durante descarga mensual"
```

---

## Task 11: Streamlit — quincenal con resolver + barra de progreso

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 11.1: Reemplazar el botón "Descargar desde INEGI (quincenal)"**

Aprox. líneas 147-162:

```python
    if col_b2.button("Descargar desde INEGI (quincenal)"):
        try:
            settings.require_token()
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            try:
                with st.status("Descargando quincenal desde INEGI…"):
                    refresh_inpc_quincenal(
                        historic=True,
                        out_path=settings.data_dir / "RelevantInflation_Q.parquet",
                    )
                st.cache_data.clear()
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
```

Reemplazar por:

```python
    if col_b2.button("Descargar desde INEGI (quincenal)"):
        try:
            settings.require_token()
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            from inflacion.data.pipeline import refresh_inpc_quincenal_with_discovery
            from inflacion.inegi.client import BIEClient, MissingTokenError

            disc_progress = st.progress(0, text="Resolviendo IDs quincenales…")
            fetch_progress = st.progress(0, text="Esperando descubrimiento…")
            status_box = st.empty()

            def _disc_cb(done: int, total: int, name: str) -> None:
                disc_progress.progress(done / total, text=f"Resolviendo {done}/{total}: {name}")

            def _fetch_cb(done: int, total: int, name: str) -> None:
                fetch_progress.progress(
                    done / total, text=f"Descargando {done}/{total}: {name}"
                )

            try:
                client = BIEClient()
                try:
                    client.health_check()
                except MissingTokenError as exc:
                    st.error(str(exc))
                    client.close()
                    st.stop()
                df = refresh_inpc_quincenal_with_discovery(
                    client=client,
                    out_path=settings.data_dir / "RelevantInflation_Q.parquet",
                    historic=True,
                    discovery_progress_cb=_disc_cb,
                    fetch_progress_cb=_fetch_cb,
                )
                status_box.success(
                    f"Quincenal: {df.shape[1]} de 10 conceptos cabecera disponibles."
                )
            except Exception as exc:  # noqa: BLE001
                status_box.error(f"Falla al descargar quincenal: {exc}")
                st.stop()
            st.cache_data.clear()
            st.rerun()
```

- [ ] **Step 11.2: Verificar lint y tests**

```bash
.venv/Scripts/python.exe -m ruff check app/streamlit_app.py
.venv/Scripts/python.exe -m pytest -q
```

Expected: ambos verdes.

- [ ] **Step 11.3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(app): quincenal con resolver + progreso de descubrimiento + descarga"
```

---

## Task 12: `.gitignore` y limpieza de cache envenenada

**Files:**
- Modify: `.gitignore`
- Delete: `data/RelevantInflation.parquet` (cache envenenada actual)

- [ ] **Step 12.1: Añadir entrada al `.gitignore`**

Editar `.gitignore` y añadir (al final, en sección de datos):

```
# Sidecar de IDs quincenales resueltos (regenerable)
data/quincenal_ids_resolved.json
```

- [ ] **Step 12.2: Eliminar la cache envenenada actual**

```bash
rm data/RelevantInflation.parquet
```

(El archivo está 0×0; la app lo regenerará en la siguiente ejecución con token válido. Si por alguna razón ya tienes una cache válida, **no hagas este paso** — verifica con `.venv/Scripts/python.exe -c "import pandas as pd; df=pd.read_parquet('data/RelevantInflation.parquet'); print(df.shape)"` antes.)

- [ ] **Step 12.3: Commit**

```bash
git add .gitignore data/RelevantInflation.parquet
git commit -m "chore(data): gitignorar sidecar quincenal y eliminar cache mensual envenenada"
```

---

## Task 13: README — documentar el flujo nuevo

**Files:**
- Modify: `README.md`

- [ ] **Step 13.1: Actualizar la sección "Frecuencia quincenal" del README**

Localizar la sección actual (aprox. líneas 118-132) que empieza con "INEGI publica el INPC cada 10 y 25 de mes…" y reemplazarla por:

```markdown
### Frecuencia quincenal

INEGI publica el INPC cada 10 y 25 de mes. La app **descubre los IDs reales del
BIE automáticamente** la primera vez que pulsas *Descargar (quincenal)*:

1. La app sondea cada candidato listado en
   `src/inflacion/inegi/series_quincenal.py::QUINCENAL_HEADLINE_CANDIDATES`.
2. Clasifica cada respuesta por formato de `TIME_PERIOD` (`YYYY/MM` = mensual,
   `YYYY/MM/Q1|Q2` = quincenal).
3. Persiste los resueltos en `data/quincenal_ids_resolved.json` (sidecar
   regenerable, ignorado por git).
4. Descarga el histórico de los IDs resueltos.

Si conoces los IDs reales y prefieres fijarlos manualmente, agrégalos a
`QUINCENAL_HEADLINE_IDS` en el mismo archivo — el resolver los respeta.
```

- [ ] **Step 13.2: Actualizar la sección "Datos" / "Pruebas"**

Si el bloque de tests menciona un número específico (e.g. "40 pruebas"), actualizarlo. Localizar:

```bash
python3 -m pytest -q     # 40 pruebas
```

Y reemplazar por:

```bash
.venv/Scripts/python.exe -m pytest -q   # ~60 pruebas
```

(Verificar el número exacto con `.venv/Scripts/python.exe -m pytest -q | tail -1` y poner el número observado.)

- [ ] **Step 13.3: Commit**

```bash
git add README.md
git commit -m "docs(readme): documentar resolver quincenal y nuevo flujo de descarga"
```

---

## Task 14: Verificación final

- [ ] **Step 14.1: Suite completa**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: todo verde. Anotar el número total de tests.

- [ ] **Step 14.2: Lint completo**

```bash
.venv/Scripts/python.exe -m ruff check src tests app
```

Expected: `All checks passed!`.

- [ ] **Step 14.3: Smoke check de imports**

```bash
.venv/Scripts/python.exe -c "from inflacion.inegi.client import BIEClient, classify_period; from inflacion.data.quincenal_resolver import resolve_quincenal_ids; from inflacion.data.pipeline import refresh_inpc_quincenal_with_discovery; print('imports OK')"
```

Expected: `imports OK`.

- [ ] **Step 14.4: (Opcional, requiere token válido) Smoke E2E**

```bash
.venv/Scripts/python.exe -c "
from inflacion.inegi.client import BIEClient
with BIEClient() as c:
    c.health_check()
print('health_check passed')
"
```

Si el token es válido, imprime `health_check passed`. Si no, muestra el mensaje accionable de `MissingTokenError`. **No hace falta que el token sea válido para que la suite pase** — los tests usan respx.

- [ ] **Step 14.5: Verificar git status limpio**

```bash
git status
```

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 14.6: Commit final si quedó algún ajuste cosmético**

Si el lint o el smoke check requirió correcciones menores, hacer un commit de cierre:

```bash
git add -A
git commit -m "chore: cierre del flujo self-heal"
```

(Saltar si no hay cambios pendientes.)

---

## Resumen de archivos tocados

**Modificados:**
- `src/inflacion/inegi/client.py` (Tasks 1, 2, 3, 4)
- `src/inflacion/data/pipeline.py` (Tasks 5, 6, 8)
- `src/inflacion/inegi/series_quincenal.py` (Task 7)
- `app/streamlit_app.py` (Tasks 9, 10, 11)
- `.gitignore` (Task 12)
- `README.md` (Task 13)
- `tests/test_client.py` (Tasks 3, 4)
- `tests/test_pipeline_quincenal.py` (Tasks 7, 8)

**Creados:**
- `src/inflacion/data/quincenal_resolver.py` (Task 7)
- `tests/test_client_classify.py` (Task 1)
- `tests/test_client_health.py` (Task 2)
- `tests/test_pipeline_self_heal.py` (Task 5)
- `tests/test_pipeline_progress.py` (Task 6)
- `tests/test_quincenal_resolver.py` (Task 7)

**Eliminados:**
- `data/RelevantInflation.parquet` (Task 12; cache 0×0 envenenada)
