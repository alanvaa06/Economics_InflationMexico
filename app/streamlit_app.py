"""Dashboard de inflación México (Streamlit, español).

Ejecutar localmente::

    streamlit run app/streamlit_app.py

Requiere:
- ``data/RelevantInflation.parquet`` o ``RelevantInflation.xlsx`` en la raíz.
- ``ponderadores.xlsx`` en la raíz.
- ``INEGI_API_TOKEN`` en ``.env`` solo si se usa el botón "Actualizar datos".
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from inflacion.analytics import (
    breakdown_core_food_nonfood,
    breakdown_core_goods_services,
    breakdown_core_services,
    bucket_distribution,
    contributions_yoy,
    incidencias,
)
from inflacion.config import settings
from inflacion.data import load_ponderadores
from inflacion.data.pipeline import load_local_inpc
from inflacion.viz import (
    contributions_bar,
    distribution_area,
    incidencias_bar,
    yoy_line_with_band,
)

st.set_page_config(
    page_title="Inflación México",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Carga (con caché)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=6 * 60 * 60)
def _load() -> tuple[pd.DataFrame, object]:
    inpc = load_local_inpc()
    pond = load_ponderadores()
    return inpc, pond


try:
    inpc, ponderadores = _load()
except FileNotFoundError as exc:
    st.error(
        "No se encontró el INPC local. Ejecuta `inflacion refresh` o usa el botón "
        "**Actualizar datos** (requiere `.env` con `INEGI_API_TOKEN`)."
    )
    st.exception(exc)
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filtros")
    min_date = inpc.index.min().to_pydatetime()
    max_date = inpc.index.max().to_pydatetime()
    default_start = max(min_date, datetime(2018, 1, 1))
    rango = st.slider(
        "Periodo",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM",
    )
    rubro = st.selectbox(
        "Agregado",
        options=list(ponderadores.weights_inpc.keys()),
        index=0,
    )
    st.divider()
    st.caption(f"Última observación local: **{max_date:%Y-%m}**")
    if st.button("Actualizar desde INEGI"):
        try:
            settings.require_token()
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            with st.status("Descargando…"):
                from inflacion.data.pipeline import refresh_inpc

                refresh_inpc(
                    historic=True,
                    out_path=settings.data_dir / "RelevantInflation.parquet",
                )
            st.cache_data.clear()
            st.rerun()

inpc_filt = inpc.loc[rango[0] : rango[1]]
since_iso = rango[0].strftime("%Y-%m")

# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

tab_overview, tab_contrib, tab_dist, tab_outliers = st.tabs(
    ["📊 Panorama", "🧩 Contribuciones", "📈 Distribución", "⚠️ Atípicos"]
)


# ----- Panorama ------------------------------------------------------------
with tab_overview:
    st.subheader("Panorama general")

    headline = inpc_filt.get("IndiceGeneral")
    if headline is not None and len(headline.dropna()) >= 13:
        yoy = headline.pct_change(12).iloc[-1]
        mom = headline.pct_change(1).iloc[-1]
        prev_yoy = headline.pct_change(12).iloc[-2]
        col1, col2, col3 = st.columns(3)
        col1.metric("Inflación YoY", f"{yoy * 100:.2f}%", delta=f"{(yoy - prev_yoy) * 100:+.2f} pp")
        col2.metric("Inflación MoM", f"{mom * 100:.2f}%")
        col3.metric("Última fecha", f"{headline.index[-1]:%b %Y}")

    if "IndiceGeneral" in inpc_filt.columns and len(inpc_filt) >= 13:
        yoy_general = inpc_filt["IndiceGeneral"].pct_change(12).rename("INPC general")
        # Subyacente sintética: media ponderada de YoY de los componentes del rubro Total_Subyacente
        sub_w = ponderadores.weights_inpc.get("Total_Subyacente")
        if sub_w is not None:
            sub_cols = [c for c in inpc_filt.columns if c in sub_w.index]
            if sub_cols:
                w = sub_w.loc[sub_cols] / sub_w.loc[sub_cols].sum()
                sub_yoy = inpc_filt[sub_cols].pct_change(12).mul(w, axis=1).sum(
                    axis=1, min_count=1
                ).rename("Subyacente")
                yoy_df = pd.concat([yoy_general, sub_yoy], axis=1)
            else:
                yoy_df = yoy_general.to_frame()
        else:
            yoy_df = yoy_general.to_frame()
        st.plotly_chart(
            yoy_line_with_band(yoy_df.dropna(how="all"), "Inflación YoY vs objetivo Banxico"),
            width="stretch",
        )

    st.markdown("##### Subyacente: Mercancías vs Servicios")
    core = breakdown_core_goods_services(inpc, ponderadores, since=since_iso)
    st.plotly_chart(
        contributions_bar(core, "Contribución Mercancías vs Servicios"),
        width="stretch",
    )


# ----- Contribuciones -------------------------------------------------------
with tab_contrib:
    st.subheader(f"Contribuciones — {rubro}")
    weights = ponderadores.weights_inpc[rubro]
    cols = [c for c in inpc.columns if c in weights.index]
    contrib = contributions_yoy(inpc[cols], weights, since=since_iso)

    col_l, col_r = st.columns([2, 3])
    with col_l:
        n_top = st.slider("Top-N por contribución", min_value=5, max_value=25, value=10)
        inc = incidencias(contrib, n=n_top)
        st.dataframe(inc.style.format({"contribucion": "{:.2%}", "pct_total": "{:.1%}"}))
    with col_r:
        st.plotly_chart(
            incidencias_bar(inc, f"Incidencias — {contrib.index[-1]:%Y-%m}"),
            width="stretch",
        )

    st.markdown("##### Mercancías → Alimentos / No-alimentos")
    food = breakdown_core_food_nonfood(inpc, ponderadores, since=since_iso)
    st.plotly_chart(
        contributions_bar(food, "Alimentos vs No-alimentos (al INPC)"),
        width="stretch",
    )

    st.markdown("##### Servicios → Educación / Vivienda / Otros (bug del notebook original corregido)")
    serv = breakdown_core_services(inpc, ponderadores, since=since_iso)
    st.plotly_chart(
        contributions_bar(serv, "Servicios (al INPC)"),
        width="stretch",
    )


# ----- Distribución --------------------------------------------------------
with tab_dist:
    st.subheader("Distribución de cambios YoY por componente")
    yoy = inpc_filt.pct_change(12).dropna(how="all")
    dist = bucket_distribution(yoy)
    st.plotly_chart(
        distribution_area(dist, "% de componentes por rango YoY"),
        width="stretch",
    )


# ----- Atípicos ------------------------------------------------------------
with tab_outliers:
    st.subheader("Detección de cambios atípicos (efectos base)")
    col1, col2 = st.columns(2)
    window = col1.slider("Ventana evaluada (meses)", 6, 60, 24)
    threshold = col2.slider("Umbral |z|", 1.0, 4.0, 2.0, step=0.25)

    from inflacion.analytics import identify_outliers_expanding_window

    yoy = inpc.pct_change(12).dropna(how="all", axis=1)
    flags = identify_outliers_expanding_window(yoy, window=window, threshold=threshold)
    pct_flag = flags.sum(axis=1) / max(flags.shape[1], 1) * 100
    st.line_chart(pct_flag.tail(window * 3))
    st.caption(
        f"Componentes marcados en la última fecha: {int(flags.iloc[-1].sum())} de {flags.shape[1]}"
    )
