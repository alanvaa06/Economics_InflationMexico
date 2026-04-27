# Economics_InflationMexico

Análisis y visualización de la inflación en México (INPC) usando datos del
Banco de Indicadores Económicos (BIE) de INEGI.

> **⚠️ Nota de seguridad:** este repositorio fue refactorizado tras detectar un
> token del INEGI expuesto en el historial. El token original fue rotado y la
> historia de git se reescribió para purgarlo. **Cualquier clon previo está
> obsoleto** — re-clona el repo.

## Características

- Cliente HTTP tipado para el BIE con reintentos, caché y redacción de tokens
  en logs (`src/inflacion/inegi/`).
- Cálculo modular de contribuciones MoM/YoY, descomposición de la subyacente
  (mercancías/servicios, alimentos/no-alimentos, educación/vivienda/otros) y
  detección de cambios atípicos.
- **Bug del notebook original (cell 56) corregido**: ``others_list_100``
  ahora usa ``others_list / others_list.sum()`` en lugar del ``food_list``
  que se había colado por copy-paste.
- Notebook ejecutivo (`Inflation_Analysis.ipynb`, 18 celdas) que solo
  importa la librería.
- Dashboard interactivo en Streamlit (`app/streamlit_app.py`), en español.
- Suite de pruebas con `pytest` + `respx` (cliente, contribuciones, bug de
  cell 56, outliers, distribución, ponderadores).
- CI en GitHub Actions y `pre-commit` con `ruff`, `nbstripout` y `gitleaks`.

## Inicio rápido

```bash
git clone https://github.com/alanvaa06/Economics_InflationMexico.git
cd Economics_InflationMexico

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# editar .env y poner tu INEGI_API_TOKEN
# obtenerlo en: https://www.inegi.org.mx/servicios/api_indicadores.html

# Descargar datos del INEGI (opcional; el repo trae un snapshot 1969-2024-04)
inflacion refresh

# Dashboard
streamlit run app/streamlit_app.py

# Notebook
jupyter lab Inflation_Analysis.ipynb
```

## Estructura

```
src/inflacion/
├── config.py            # carga INEGI_API_TOKEN desde .env (pydantic-settings)
├── inegi/
│   ├── client.py        # cliente httpx + tenacity + redacción de tokens
│   └── series.py        # catálogo de series del BIE
├── data/
│   ├── ponderadores.py  # ponderadores INPC (rebase a 100 por rubro)
│   └── pipeline.py      # orquestación: fetch → DataFrame ancho → parquet
├── analytics/
│   ├── contributions.py # MoM / YoY / incidencias
│   ├── core_breakdown.py# Mercancías/Servicios, Alimentos/NoAlim, Edu/Viv/Otros
│   ├── distributions.py # distribución por rangos de cambio YoY
│   ├── base_effect.py   # outliers (z-score expansivo)
│   └── zscore.py
├── viz/
│   └── charts.py        # figuras Plotly compartidas con la app
└── cli.py               # `inflacion refresh`

app/streamlit_app.py     # dashboard
tests/                   # pruebas unitarias
notebooks/               # (opcional) cuadernos de exploración
```

## Pruebas

```bash
python3 -m pytest -q     # 23 pruebas
python3 -m ruff check src tests app
```

Pruebas marcadas como `@pytest.mark.live` consultan el BIE real (requieren
`INEGI_API_TOKEN`); por defecto se omiten.

## Seguridad y manejo del token

- El token siempre se carga desde `.env` o variable de entorno
  `INEGI_API_TOKEN` (vía `pydantic-settings`).
- El cliente registra URLs **sin** el token: cualquier UUID detectado en
  mensajes/excepciones se sustituye por `REDACTED`.
- `pre-commit` corre `gitleaks` y `nbstripout` para evitar fugas futuras.
- Si necesitas regenerar tu token: https://www.inegi.org.mx/servicios/api_indicadores.html

## Datos

- `SeriesInflation_ids.xlsx`: catálogo de IDs del BIE (471 series).
- `ponderadores.xlsx` (hoja `ObjetoGasto`): pesos del INPC y marcas `X`
  para cada rubro/agregado. INEGI actualizó la canasta y ponderadores en
  2024 (base 2ª quincena de julio 2018 = 100); mantenlos sincronizados.
- `RelevantInflation.xlsx` / `FullInflation.xlsx`: snapshots históricos
  generados con `inflacion refresh`.

## Licencia

MIT.
