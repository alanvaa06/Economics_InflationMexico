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
- **Banda objetivo Banxico (3% ± 1pp)** sombreada sobre la línea YoY del
  INPC general y la subyacente en la pestaña Panorama.
- **Núcleo alternativo**: media truncada y mediana ponderada de la sección
  transversal de YoY (filtros que detectan ruido idiosincrático que la
  subyacente oficial deja pasar). 5ta pestaña dedicada con selectbox de
  universo y slider de % de recorte.
- **Frecuencia quincenal**: el cliente y el dashboard manejan los formatos
  ``YYYY/MM`` y ``YYYY/MM/Q1|Q2`` del BIE. Un radio en la barra lateral
  alterna Panorama entre mensual y quincenal; las demás pestañas requieren
  componentes con peso (sólo mensual).

## Inicio rápido

```bash
git clone https://github.com/alanvaa06/Economics_InflationMexico.git
cd Economics_InflationMexico

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# editar .env y poner tu INEGI_API_TOKEN
# obtenerlo en: https://www.inegi.org.mx/servicios/api_indicadores.html

# Descargar/generar cache local desde INEGI
inflacion refresh                       # mensual (471 series)
inflacion refresh --frequency quincenal # quincenal (~10 agregados de cabecera)

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
│   ├── client.py            # cliente httpx + tenacity + redacción de tokens
│   ├── series.py            # catálogo mensual (471 series, sin dependencia XLSX)
│   └── series_quincenal.py  # catálogo quincenal (~10 IDs cabecera)
├── data/
│   ├── ponderadores.py      # ponderadores oficiales INEGI + cache local normalizado
│   └── pipeline.py          # refresh_inpc / refresh_inpc_quincenal
├── analytics/
│   ├── contributions.py     # MoM / YoY / incidencias
│   ├── core_breakdown.py    # Mercancías/Servicios, Alimentos/NoAlim, Edu/Viv/Otros
│   ├── alt_core.py          # media truncada + mediana ponderada (núcleo alt.)
│   ├── distributions.py     # distribución por rangos de cambio YoY
│   ├── base_effect.py       # outliers (z-score expansivo)
│   └── zscore.py
├── viz/
│   └── charts.py            # incl. yoy_line_with_band (banda Banxico)
└── cli.py                   # `inflacion refresh [--frequency …]`

app/streamlit_app.py     # dashboard
tests/                   # pruebas unitarias
notebooks/               # (opcional) cuadernos de exploración
```

## Pruebas

```bash
python3 -m pytest -q     # 40 pruebas
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

- **Fuente primaria mensual/quincenal**: API BIE de INEGI (`inflacion refresh`).
- **Fuente primaria de canasta/ponderadores**: RNM de INEGI
  (`https://www.inegi.org.mx/rnm/index.php/catalog/1015/download/32034`).
- **Cache local generado**: `data/RelevantInflation.parquet`,
  `data/RelevantInflation_Q.parquet` y `data/ponderadores_inpc_official.parquet`.
- Los Excel históricos del repositorio ya no son requeridos para ejecutar la app.

### Frecuencia quincenal

INEGI publica el INPC cada 10 y 25 de mes. El catálogo en
`src/inflacion/inegi/series_quincenal.py` viene con **placeholders** para
los IDs del BIE. Para activarlo:

1. Abre el [Banco de Indicadores](https://www.inegi.org.mx/app/indicadores/)
   filtrando por frecuencia *Quincenal*.
2. Copia los IDs numéricos de los agregados (`IndiceGeneral`, `Subyacente`,
   `NoSubyacente`, etc.) en `QUINCENAL_HEADLINE_IDS`.
3. Corre `inflacion refresh --frequency quincenal`. El cliente acepta los
   formatos ``YYYY/MM/Q1`` y ``YYYY/MM/Q2`` que devuelve el BIE.

El parquet resultante (`data/RelevantInflation_Q.parquet`) lo consume el
dashboard cuando se elige *Quincenal* en la barra lateral.

## Licencia

MIT.
