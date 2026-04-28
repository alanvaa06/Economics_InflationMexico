"""Dashboard de inflación México (Streamlit, español).

Ejecutar localmente::

    streamlit run app/streamlit_app.py

Requiere:
- ``INEGI_API_TOKEN`` en ``.env`` para descargar/actualizar desde INEGI.
- Cache local generado en ``data/`` (parquet) tras la primera descarga.
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
    trimmed_mean_yoy,
    weighted_median_yoy,
)
from inflacion.config import settings
from inflacion.data import load_ponderadores
from inflacion.data.pipeline import (
    load_local_inpc,
    load_local_inpc_quincenal,
    refresh_inpc,
)
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
def _load_monthly() -> tuple[pd.DataFrame, object]:
    inpc = load_local_inpc()
    pond = load_ponderadores()
    return inpc, pond


@st.cache_data(ttl=6 * 60 * 60)
def _load_quincenal() -> pd.DataFrame:
    return load_local_inpc_quincenal()


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
except Exception as exc:
    st.error("Falla inesperada al cargar el INPC mensual.")
    st.exception(exc)
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filtros")
    freq = st.radio(
        "Frecuencia",
        options=["Mensual", "Quincenal"],
        horizontal=True,
        index=0,
        help=(
            "Mensual: 471 series del INPC. "
            "Quincenal: solo agregados de cabecera (~10 series)."
        ),
    )

    if freq == "Quincenal":
        try:
            inpc_active = _load_quincenal()
        except FileNotFoundError:
            st.warning(
                "Aún no hay datos quincenales en disco. "
                "Pulsa **Actualizar (quincenal)** abajo."
            )
            inpc_active = pd.DataFrame()
    else:
        inpc_active = inpc

    if not inpc_active.empty:
        min_date = inpc_active.index.min().to_pydatetime()
        max_date = inpc_active.index.max().to_pydatetime()
        default_start = max(min_date, datetime(2018, 1, 1))
        rango = st.slider(
            "Periodo",
            min_value=min_date,
            max_value=max_date,
            value=(default_start, max_date),
            format="YYYY-MM",
        )
    else:
        rango = (datetime(2018, 1, 1), datetime.now())

    rubro = st.selectbox(
        "Agregado",
        options=list(ponderadores.weights_inpc.keys()),
        index=0,
        help="Sólo aplica en frecuencia mensual (los pesos son anuales).",
    )
    st.divider()
    if not inpc_active.empty:
        st.caption(
            f"Última observación local ({freq.lower()}): **{inpc_active.index.max():%Y-%m-%d}**"
        )
    st.caption("Fuente: **INEGI (BIE/RNM)** | cache local generado por la app.")

    col_b1, col_b2 = st.columns(2)
    if col_b1.button("Descargar desde INEGI (mensual)"):
        try:
            settings.require_token()
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            from inflacion.inegi.client import BIEClient, MissingTokenError

            progress = st.progress(0, text="Descargando 0/?")
            status_box = st.empty()

            def _cb(done: int, total: int, name: str) -> None:
                progress.progress(done / total, text=f"Descargando {done}/{total}: {name[:40]}")

            try:
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
            except Exception as exc:
                status_box.error(f"Falla al descargar: {exc}")
                st.stop()
            st.cache_data.clear()
            st.rerun()
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
            except Exception as exc:
                status_box.error(f"Falla al descargar quincenal: {exc}")
                st.stop()
            st.cache_data.clear()
            st.rerun()

if inpc_active.empty:
    st.info(
        "Frecuencia quincenal sin datos locales. "
        "Pulsa **Actualizar (quincenal)** en la barra lateral."
    )
    st.stop()

inpc_filt = inpc_active.loc[rango[0] : rango[1]]
since_iso = rango[0].strftime("%Y-%m")
is_monthly = freq == "Mensual"

# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

tab_overview, tab_contrib, tab_dist, tab_outliers, tab_alt_core = st.tabs(
    [
        "📊 Panorama",
        "🧩 Contribuciones",
        "📈 Distribución",
        "⚠️ Atípicos",
        "🧪 Núcleo alternativo",
    ]
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
        yoy_general = (
            inpc_filt["IndiceGeneral"].pct_change(12, fill_method=None).rename("INPC general")
        )

        if is_monthly:
            # Subyacente sintética: media ponderada YoY de los componentes del
            # rubro Total_Subyacente (los pesos son anuales, sólo aplica mensual).
            sub_w = ponderadores.weights_inpc.get("Total_Subyacente")
            sub_cols = (
                [c for c in inpc_filt.columns if c in sub_w.index] if sub_w is not None else []
            )
            if sub_cols:
                w = sub_w.loc[sub_cols] / sub_w.loc[sub_cols].sum()
                sub_yoy = (
                    inpc_filt[sub_cols]
                    .pct_change(12, fill_method=None)
                    .mul(w, axis=1)
                    .sum(axis=1, min_count=1)
                    .rename("Subyacente")
                )
                yoy_df = pd.concat([yoy_general, sub_yoy], axis=1)
            else:
                yoy_df = yoy_general.to_frame()
        elif "Subyacente" in inpc_filt.columns:
            yoy_sub = (
                inpc_filt["Subyacente"].pct_change(12, fill_method=None).rename("Subyacente")
            )
            yoy_df = pd.concat([yoy_general, yoy_sub], axis=1)
        else:
            yoy_df = yoy_general.to_frame()

        st.plotly_chart(
            yoy_line_with_band(yoy_df.dropna(how="all"), "Inflación YoY vs objetivo Banxico"),
            width="stretch",
        )

    if is_monthly:
        st.markdown("##### Subyacente: Mercancías vs Servicios")
        core = breakdown_core_goods_services(inpc, ponderadores, since=since_iso)
        st.plotly_chart(
            contributions_bar(core, "Contribución Mercancías vs Servicios"),
            width="stretch",
        )
    else:
        st.info(
            "Mercancías vs Servicios y otras descomposiciones requieren datos a "
            "nivel componente (sólo publicados mensualmente)."
        )


# ----- Contribuciones -------------------------------------------------------
with tab_contrib:
    st.subheader(f"Contribuciones — {rubro}")
    if not is_monthly:
        st.info("Disponible en frecuencia mensual.")
    else:
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

        st.markdown(
            "##### Servicios → Educación / Vivienda / Otros (bug del notebook original corregido)"
        )
        serv = breakdown_core_services(inpc, ponderadores, since=since_iso)
        st.plotly_chart(
            contributions_bar(serv, "Servicios (al INPC)"),
            width="stretch",
        )


# ----- Distribución --------------------------------------------------------
with tab_dist:
    st.subheader("Distribución de cambios YoY por componente")
    if not is_monthly:
        st.info("Disponible en frecuencia mensual.")
    else:
        yoy = inpc_filt.pct_change(12, fill_method=None).dropna(how="all")
        dist = bucket_distribution(yoy)
        st.plotly_chart(
            distribution_area(dist, "% de componentes por rango YoY"),
            width="stretch",
        )


# ----- Atípicos ------------------------------------------------------------
with tab_outliers:
    st.subheader("Detección de cambios atípicos (efectos base)")
    if not is_monthly:
        st.info("Disponible en frecuencia mensual.")
    else:
        col1, col2 = st.columns(2)
        window = col1.slider("Ventana evaluada (meses)", 6, 60, 24)
        threshold = col2.slider("Umbral |z|", 1.0, 4.0, 2.0, step=0.25)

        from inflacion.analytics import identify_outliers_expanding_window

        yoy = inpc.pct_change(12, fill_method=None).dropna(how="all", axis=1)
        flags = identify_outliers_expanding_window(yoy, window=window, threshold=threshold)
        pct_flag = flags.sum(axis=1) / max(flags.shape[1], 1) * 100
        st.line_chart(pct_flag.tail(window * 3))
        st.caption(
            f"Componentes marcados en la última fecha: {int(flags.iloc[-1].sum())} de {flags.shape[1]}"
        )


# ----- Núcleo alternativo --------------------------------------------------
with tab_alt_core:
    st.subheader("Núcleo alternativo: media truncada y mediana ponderada")
    if not is_monthly:
        st.info("Disponible en frecuencia mensual (los pesos del INPC son anuales).")
    else:
        st.caption(
            "Filtros de sección transversal: ordenan los componentes por YoY y "
            "descartan colas (truncada) o eligen el valor cuya masa acumulada cruza "
            "0.5 (mediana). Útiles cuando un solo subgrupo ensucia la subyacente."
        )

        universos = list(ponderadores.weights_inpc.keys())
        default_idx = (
            universos.index("IndiceGeneral") if "IndiceGeneral" in universos else 0
        )
        universo = st.selectbox("Universo de componentes", universos, index=default_idx)
        trim_pct = st.slider("Recorte por cola (%)", 0, 40, 10, step=5) / 100

        weights_universe = ponderadores.weights_inpc[universo]
        cols_universe = [c for c in inpc_filt.columns if c in weights_universe.index]

        if not cols_universe or len(inpc_filt) < 13:
            st.info("No hay suficientes datos para calcular núcleo alternativo.")
        else:
            sub_w = ponderadores.weights_inpc.get("Total_Subyacente")
            sub_yoy = None
            if sub_w is not None:
                sub_cols = [c for c in inpc_filt.columns if c in sub_w.index]
                if sub_cols:
                    w_sub = sub_w.loc[sub_cols] / sub_w.loc[sub_cols].sum()
                    sub_yoy = (
                        inpc_filt[sub_cols]
                        .pct_change(12, fill_method=None)
                        .mul(w_sub, axis=1)
                        .sum(axis=1, min_count=1)
                        .rename("Subyacente oficial")
                    )

            tm = trimmed_mean_yoy(
                inpc_filt[cols_universe], weights_universe, trim=trim_pct, since=None
            ).rename(f"Media truncada {int(trim_pct * 100)}%")
            wm = weighted_median_yoy(
                inpc_filt[cols_universe], weights_universe, since=None
            ).rename("Mediana ponderada")

            parts = [s for s in (sub_yoy, tm, wm) if s is not None]
            df_alt = pd.concat(parts, axis=1).dropna(how="all")
            st.plotly_chart(
                yoy_line_with_band(df_alt, "Núcleo: oficial vs alternativos"),
                width="stretch",
            )

            last_row = df_alt.dropna(how="all").iloc[-1]
            cols_metrics = st.columns(len(last_row))
            for col_widget, (name, val) in zip(cols_metrics, last_row.items(), strict=False):
                col_widget.metric(name, f"{val * 100:.2f}%" if pd.notna(val) else "—")
