# Inflation data self-heal — Design Spec

**Date:** 2026-04-27
**Author:** alanvaa06 (with Claude)
**Status:** Approved (architecture section approved during brainstorming; spec captures decisions)

## Goal

Cerrar el flujo "primera ejecución" de la app de inflación México: la descarga desde INEGI BIE, la persistencia de cache local y la activación de la pestaña quincenal deben funcionar sin que el usuario edite código, sin estados silenciosamente corruptos, y con feedback visible. Sin agregar datasets nuevos.

## Non-Goals

- Agregar nuevos catálogos (INPP productor, INPC por entidad/zona metropolitana, otras frecuencias).
- Rediseñar la navegación de la app o agregar pestañas.
- Cambiar la analítica existente (contribuciones, núcleo alternativo, distribución, atípicos).
- Construir un panel persistente "Estado de datos".

## Background y diagnóstico

Estado observado en el repositorio al iniciar este spec:

1. `data/RelevantInflation.parquet` existe pero tiene **0×0** (cache envenenada).
2. `src/inflacion/inegi/series_quincenal.py` define `QUINCENAL_HEADLINE_IDS` con strings vacíos como placeholders. La pestaña Panorama-quincenal queda gateada.
3. **46 tests pasan** ejecutados con `.venv/Scripts/python.exe -m pytest`.
4. Sondas en vivo contra BIE INEGI desde el entorno del usuario:
   - `GET INDICATOR/583766` (IndiceGeneral mensual conocido) → **HTTP 400** `"ErrorInfo:No se encontraron resultados"`.
   - `GET INDICATOR/493663` (PIB Trimestral, ID público de ejemplo) → mismo 400.
   - Mismo error con un token bogus (`00000000-0000-...`).
   - **Conclusión**: el token actual del `.env` es sintácticamente válido (UUID, 36 chars, prefijo `4f988b8a-…`) pero INEGI lo rechaza. INEGI devuelve 400 (no 403) para tokens inválidos.

El cliente actual (`BIEClient._get`) trata 400 como "respuesta inesperada" y lo propaga a `fetch_many`, que **silenciosamente** lo descarta y devuelve `DataFrame()` vacío. `refresh_inpc` lo persiste sin validar. Eso explica el parquet 0×0.

## Decisiones de scope

| ID | Decisión | Impacto |
|---|---|---|
| D1 | Scope = "cerrar lo roto" (A). Sin nuevos datasets. | Cambios localizados en 4 archivos existentes + tests. |
| D2 | Quincenal = auto-descubrimiento (B), pero implementado como **seed verificado** (Enfoque 1). | `series_quincenal.py` lleva candidatos curados; pipeline verifica al primer refresh. |
| D3 | Self-heal = híbrido (C). Auto-recuperación de cache + barra de progreso real X/N. Sin panel "Estado de datos" persistente. | Mensajes informativos como `st.toast` o `st.expander` post-refresh. |
| D4 | Asumimos que el usuario renueva el token antes de la primera ejecución; el código se hace robusto contra todo este caso. | Token-health-probe distingue auth-fail; UX con onboarding claro. |

## Arquitectura

### Capas tocadas

```
src/inflacion/
├── inegi/
│   ├── client.py              [EDIT] health_check(), classify_period(), fetch_many no devuelve DF vacío
│   └── series_quincenal.py    [EDIT] candidatos curados (seed) + metadata por nombre
├── data/
│   └── pipeline.py            [EDIT] auto-heal parquet vacío/corrupto, resolver quincenal con sidecar, progress callback
└── (sin cambios en analytics/, viz/, config.py, cli.py)

app/
└── streamlit_app.py           [EDIT] progress bar real, onboarding sin token, mensajes auth-fail

data/
└── quincenal_ids_resolved.json [NEW]  cache de IDs verificados {nombre: {id, period_format, last_period, verified_at}}

tests/
├── test_client_health.py       [NEW]
├── test_client_classify.py     [NEW]
├── test_pipeline_self_heal.py  [NEW]
├── test_pipeline_progress.py   [NEW]
├── test_quincenal_resolver.py  [NEW]
└── (existentes intactos)
```

### Flujo end-to-end primera ejecución

```
streamlit run
  └─> _load_monthly() → load_local_inpc()
        ├─ parquet existe Y filas>0 Y columnas>0 → render
        ├─ parquet existe pero (0 filas O 0 columnas) → log + delete + refetch
        └─ no existe
              ├─ token ausente → onboarding (no st.stop, link a INEGI)
              └─ token presente → BIEClient.health_check()
                    ├─ auth-fail → mensaje accionable "token rechazado por INEGI…"
                    └─ ok → refresh_inpc(progress_cb=…) con barra X/471
                            ├─ DF resultante vacío → raise (no persistir)
                            └─ DF con ≥1 col → persist parquet + render
```

### Flujo quincenal (cuando el usuario elige radio "Quincenal" o pulsa "Descargar quincenal")

```
load_local_inpc_quincenal()
  ├─ parquet quincenal existe Y filas>0 → render
  └─ no
        ├─ sidecar quincenal_ids_resolved.json existe → usar IDs cacheados → fetch
        └─ no
              └─ resolve_quincenal_ids(seed_candidates) → para cada candidato:
                    ├─ probe ID con historic=False (1 obs reciente)
                    ├─ classify_period: YYYY/MM=mensual, YYYY/MM/Qn=quincenal, otro=desconocido
                    ├─ keep solo los quincenales
                    └─ persist sidecar → fetch histórico de los IDs OK
```

## Componentes

### 1. `BIEClient.health_check(self) -> None`

Hace una sonda contra un ID maestro conocido (IndiceGeneral mensual = `583766`) usando `historic=False`. Distingue entre:

- **OK** (200 con OBSERVATIONS no vacías): retorna sin error.
- **Auth-fail** (400 con cuerpo conteniendo `"No se encontraron resultados"`): raise `MissingTokenError("INEGI rechazó el token (HTTP 400). Renueva en https://www.inegi.org.mx/servicios/api_indicadores.html y actualiza .env.")`.
- **Otro 4xx/5xx**: raise `BIEError(...)`.

La heurística "400 + texto" es necesaria porque BIE no devuelve 403 ni JSON estructurado de error.

### 2. `BIEClient.classify_period(payload: dict) -> Literal["mensual","quincenal","desconocido"]`

Función pura que mira `payload["Series"][0]["OBSERVATIONS"][0]["TIME_PERIOD"]`:

- `"YYYY/MM"` (2 partes) → `"mensual"`
- `"YYYY/MM/Q1"` o `"YYYY/MM/Q2"` o `"YYYY/MM/1"` o `"YYYY/MM/2"` → `"quincenal"`
- Otro o vacío → `"desconocido"`

### 3. `BIEClient.fetch_many` — comportamiento actualizado

- Si el loop interior termina con 0 frames acumulados (todas las series fallaron), **raise `BIEError("BIE: ninguna serie respondió OK; revisa token o catálogo.")`** en vez de `return pd.DataFrame()`.
- Acepta nuevo parámetro `progress_cb: Callable[[int, int, str], None] | None = None`. Se llama tras cada serie con `(done, total, current_name)`. Cuando es `None`, comportamiento idéntico al actual.

### 4. `series_quincenal.py` — seed con candidatos curados

Reemplazar el dict de IDs vacíos por una lista de **candidatos** con IDs que el resolver probará. Como BIE no permite "listar series", el seed contiene IDs candidatos publicados en documentación INEGI (los que sean conocidos por la comunidad para INPC quincenal). Si un candidato falla, se documenta como "pendiente" — el resolver guarda los que sí responden quincenal.

Formato:

```python
QUINCENAL_HEADLINE_CANDIDATES: dict[str, list[str]] = {
    "IndiceGeneral":   ["628194", "910392"],   # IDs candidatos a probar en orden
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
```

Los IDs concretos se afinan en la primera implementación con el token válido. Si ningún candidato funciona para un nombre, ese concepto queda fuera del sidecar y la app muestra "INPC quincenal disponible para X de 10 conceptos cabecera".

### 5. `pipeline.resolve_quincenal_ids(client, candidates) -> dict[str, str]`

Para cada `(nombre, [ids…])`:

- Probar cada ID en orden con `client.fetch_series(id, historic=False)`.
- Si responde y `classify_period(payload) == "quincenal"`, registrar `{nombre: id}` y avanzar al siguiente nombre.
- Si todos los IDs candidatos fallan o son mensuales, omitir ese nombre.

Devuelve el mapping resuelto. Persiste en `data/quincenal_ids_resolved.json` con metadata:

```json
{
  "verified_at": "2026-04-27T18:42:00Z",
  "ids": {
    "IndiceGeneral": {"id": "628194", "period_format": "YYYY/MM/Qn", "last_period": "2026/04/Q1"},
    ...
  }
}
```

### 6. `pipeline.load_local_inpc(path)` — auto-heal

```
if target.exists():
    df = pd.read_parquet(target)
    if df.empty or df.shape[1] == 0:
        logger.warning("Parquet vacío en %s — re-descargando", target)
        target.unlink()
    else:
        return df.sort_index()
# fallthrough: refresh
```

Idem para `load_local_inpc_quincenal`.

### 7. `pipeline.refresh_inpc(progress_cb=None)`

- Pasa `progress_cb` al cliente.
- Tras `fetch_many`, si DF vacío → raise (no escribe parquet).

### 8. `app/streamlit_app.py`

Cambios:

- **Onboarding sin token**: al inicio, si `settings.inegi_api_token` está vacío y no hay parquet con datos, renderizar página de bienvenida con instrucciones (cómo obtener token + dónde ponerlo) en vez de `st.error + st.stop`.
- **Auth-fail explícito**: capturar `MissingTokenError` específicamente con `st.error` accionable.
- **Progress bar real** durante `refresh_inpc`: usar `st.progress` actualizado vía callback. Mostrar nombre del componente actual.
- **Auto-heal silencioso**: la lógica de detección de parquet vacío vive en `pipeline`, no en la app — la app solo se beneficia.
- **Quincenal con resolver**: el botón "Descargar quincenal" llama a un nuevo helper que primero ejecuta `resolve_quincenal_ids` (con su propia barra de progreso de 10 probes), luego `refresh_inpc_quincenal` con los IDs resueltos.
- **Mensaje de cobertura quincenal**: tras resolver, mostrar `st.info` con "X de 10 conceptos cabecera disponibles".

## Datos y persistencia

- `data/RelevantInflation.parquet` — sin cambio de esquema. Auto-curado.
- `data/RelevantInflation_Q.parquet` — sin cambio de esquema. Auto-curado.
- `data/quincenal_ids_resolved.json` — **nuevo**. Sidecar leído por `refresh_inpc_quincenal` cuando exista; ignorado si está corrupto. Re-generable.
- `.gitignore` debe incluir el sidecar (es regenerable y depende del token).

## Estrategia de testing

Todos los tests con `respx` (igual que el resto del repo). Sin live calls (las live calls van marcadas `@pytest.mark.live` y siguen excluidas por defecto).

| Test file | Casos |
|---|---|
| `test_client_health.py` | health_check OK con 200; auth-fail con 400 + "No se encontraron resultados" → MissingTokenError; otro 5xx → BIEError. |
| `test_client_classify.py` | classify_period devuelve "mensual"/"quincenal"/"desconocido" para los formatos esperados. |
| `test_pipeline_self_heal.py` | load_local_inpc detecta parquet 0×0 y vuelve a llamar refresh; refresh nunca persiste DF vacío. |
| `test_quincenal_resolver.py` | resolver clasifica candidatos correctamente; si todos fallan, ese nombre queda fuera; sidecar persistido y leído. |
| `test_pipeline_progress.py` | progress_cb se invoca el número correcto de veces con done/total/name correctos. |

Existing tests (46) se mantienen verdes. Si alguno toca `fetch_many` esperando DF vacío como "OK", se ajusta.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Los IDs candidatos en seed están todos mal | El resolver omite y la app muestra "0 de 10 disponibles"; no rompe nada. Documentamos en el README que el usuario puede agregar candidatos manualmente. |
| INEGI cambia el formato de error 400 | health_check usa heurística string-match con fallback (`raise BIEError` si no es match). |
| Parquet vacío persiste en repos clonados | El cleanup automático corre en `load_local_inpc`; un `git pull` que traiga parquet vacío se autocura al primer arranque. |
| Pruebas live requieren token válido | Marcadas `@pytest.mark.live`, excluidas por defecto. CI no las corre. |

## Criterios de aceptación

1. `python -m pytest -q` con el venv del proyecto pasa al 100% (≥51 tests, +5 nuevos archivos).
2. `ruff check src tests app` limpio.
3. Con token inválido en `.env`, la app arranca sin tracebacks; muestra mensaje accionable.
4. Con token válido y sin caches, la app descarga las 471 series mostrando barra de progreso visible (no spinner ciego).
5. Con parquet 0×0 en disco, la app lo detecta, lo borra y re-descarga sin intervención.
6. Click en "Descargar quincenal" con token válido resuelve ≥1 ID quincenal, persiste sidecar, y descarga el histórico. Si 0 resuelven, mensaje claro y la app no se rompe.

## Out of scope (explícito)

- Nuevos datasets / pestañas / análisis.
- Modificar el catálogo mensual existente.
- Cache busting global o estrategias de TTL más sofisticadas que las actuales.
- Migrar a httpx async, paralelizar fetch_many.
- Internacionalización (la app sigue en español).
