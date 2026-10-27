"""App de Streamlit: análisis de riesgo de una acción.

Ejecutar:  streamlit run streamlit_app.py
La lógica de cálculo vive en ``core/`` y los textos de ayuda («?») en ``glosario.py``;
este archivo solo arma la interfaz.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from core import backtesting as bt
from core import datos, riesgo
from core import estadisticas as est
from core import volatilidad as vol
from glosario import AYUDA as AY

st.set_page_config(page_title="Riesgo de una acción", page_icon="📉", layout="wide")

AZUL, ROJO, GRIS, NARANJA = "#2563eb", "#dc2626", "#6b7280", "#f59e0b"


# ------------------------------------------------------------------ utilidades de interfaz
def mostrar(fig: go.Figure, titulo: str = "", ayuda: str | None = None, alto: int = 360) -> None:
    """Gráfico compacto con título y botón «?» (el título va fuera de Plotly para poder llevar ayuda)."""
    if titulo:
        st.markdown(f"**{titulo}**", help=ayuda)
    fig.update_layout(
        height=alto,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig)


@st.cache_data(ttl=3600, show_spinner="Descargando precios…")
def cargar_precios(ticker: str, inicio: dt.date, fin: dt.date) -> pd.Series:
    return datos.descargar_precios(ticker, inicio, fin)


def pct(x: float, d: int = 2) -> str:
    return f"{x:.{d}%}"


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("Parámetros")
    with st.form("parametros"):
        ticker = st.text_input("Ticker", value="AAPL", help=AY["ticker"])
        hoy = dt.date.today()
        periodo = st.date_input(
            "Periodo",
            value=(hoy - dt.timedelta(days=5 * 365), hoy),
            max_value=hoy,
            help=AY["periodo"],
        )
        confianza = st.selectbox(
            "Nivel de confianza",
            [0.90, 0.95, 0.975, 0.99],
            index=1,
            format_func=lambda a: f"{a:.1%}",
            help=AY["confianza"],
        )
        horizonte = st.number_input(
            "Horizonte (días)", min_value=1, max_value=30, value=1, step=1, help=AY["horizonte"]
        )
        posicion = st.number_input(
            "Monto de la posición", min_value=1.0, value=100_000.0, step=10_000.0, help=AY["posicion"]
        )
        rf = (
            st.number_input(
                "Tasa libre de riesgo anual (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.25,
                help=AY["rf"],
            )
            / 100
        )
        with st.expander("Avanzado"):
            tipo = st.radio(
                "Tipo de rendimiento",
                ["simple", "log"],
                format_func=lambda t: "Simple" if t == "simple" else "Logarítmico",
                help=AY["tipo"],
            )
            ventana = st.slider("Ventana móvil (días)", 60, 500, 252, step=21, help=AY["ventana"])
            lam = st.slider("Lambda de EWMA", 0.80, 0.99, 0.94, step=0.01, help=AY["lambda"])
        st.form_submit_button("Calcular", type="primary")

if not isinstance(periodo, tuple) or len(periodo) != 2:
    st.info("Selecciona fecha inicial y final del periodo.")
    st.stop()
inicio, fin = periodo

# ------------------------------------------------------------------ datos
st.title("Riesgo de una acción")

try:
    precios = cargar_precios(ticker, inicio, fin)
except datos.ErrorDatos as exc:
    st.error(str(exc))
    st.stop()

r = est.rendimientos(precios, tipo)
if len(r) < 120:
    st.error("Hay muy pocos datos en ese rango (mínimo ~120 observaciones). Amplía el periodo.")
    st.stop()

res = est.resumen_estadistico(r, rf)
dd = est.drawdown(precios)
st.caption(
    f"{precios.name} · {len(r):,} rendimientos diarios · {precios.index[0]:%d-%b-%Y} a {precios.index[-1]:%d-%b-%Y}"
)
st.caption("Pasa el cursor (o toca en el celular) el botón ? junto a cada concepto para ver qué significa.")

tab_res, tab_dist, tab_var, tab_vol, tab_bt = st.tabs(
    ["Resumen", "Distribución", "VaR y ES", "Volatilidad", "Backtesting"]
)

# ------------------------------------------------------------------ resumen
with tab_res:
    c = st.columns(4)
    c[0].metric("Último precio", f"{precios.iloc[-1]:,.2f}", help=AY["precio"])
    c[1].metric("Rendimiento anualizado", pct(res["media_anual"]), help=AY["rend_anual"])
    c[2].metric("Volatilidad anual", pct(res["vol_anual"]), help=AY["vol_anual"])
    c[3].metric("Máximo drawdown", pct(dd.min()), help=AY["drawdown"])
    c = st.columns(4)
    c[0].metric("Varianza anual", f"{res['varianza_anual']:.4f}", help=AY["varianza"])
    c[1].metric("Asimetría", f"{res['asimetria']:.2f}", help=AY["asimetria"])
    c[2].metric("Curtosis (exceso)", f"{res['curtosis_exceso']:.2f}", help=AY["curtosis"])
    c[3].metric("Sharpe", f"{res['sharpe']:.2f}", help=AY["sharpe"])

    st.markdown("**Prueba de normalidad (Jarque-Bera)**", help=AY["jb"])
    if res["jb_pvalor"] < 0.05:
        st.warning(
            f"Se rechaza normalidad (p = {res['jb_pvalor']:.4f}). "
            f"Asimetría = {res['asimetria']:.2f} y curtosis en exceso = {res['curtosis_exceso']:.2f}"
            + (
                ": las colas son más pesadas que las de una normal, así que el VaR normal tiende a subestimar el riesgo."
                if res["curtosis_exceso"] > 0
                else "."
            )
        )
    else:
        st.success(f"No se rechaza normalidad al 5 % (p = {res['jb_pvalor']:.4f}).")

    fig = go.Figure(go.Scatter(x=precios.index, y=precios, line=dict(color=AZUL, width=1.5), name="Precio"))
    mostrar(fig, "Precio ajustado", AY["g_precio"])
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=ROJO, width=1), name="Drawdown"))
    fig.update_yaxes(tickformat=".0%")
    mostrar(fig, "Drawdown", AY["g_dd"], alto=260)

    with st.expander("Todas las estadísticas"):
        filas = [
            ("Observaciones", f"{int(res['n']):,}", "n"),
            ("Media diaria", pct(res["media_diaria"], 4), "media_diaria"),
            ("Media anualizada", pct(res["media_anual"]), "rend_anual"),
            ("Varianza diaria", f"{res['varianza_diaria']:.6f}", "varianza_diaria"),
            ("Varianza anualizada", f"{res['varianza_anual']:.4f}", "varianza"),
            ("Desviación estándar diaria", pct(res["desv_diaria"], 3), "desv_diaria"),
            ("Volatilidad anualizada", pct(res["vol_anual"]), "vol_anual"),
            ("Asimetría", f"{res['asimetria']:.3f}", "asimetria"),
            ("Curtosis en exceso", f"{res['curtosis_exceso']:.3f}", "curtosis"),
            ("Mínimo diario", pct(res["minimo"]), "minimo"),
            ("Máximo diario", pct(res["maximo"]), "maximo"),
            ("Jarque-Bera (estadístico)", f"{res['jb_estadistico']:.2f}", "jb_est"),
            ("Jarque-Bera (p-valor)", f"{res['jb_pvalor']:.4f}", "jb_p"),
            ("Sharpe", f"{res['sharpe']:.3f}", "sharpe"),
            ("Sortino", f"{res['sortino']:.3f}", "sortino"),
        ]
        st.dataframe(
            pd.DataFrame([(a, b, AY[k]) for a, b, k in filas], columns=["Estadístico", "Valor", "Qué significa"]),
            hide_index=True,
        )

# ------------------------------------------------------------------ distribución
with tab_dist:
    gl, loc_t, esc_t = riesgo.ajustar_t(r)
    x = np.linspace(r.min(), r.max(), 400)
    fig = go.Figure()
    fig.add_histogram(x=r, histnorm="probability density", nbinsx=60, name="Rendimientos", marker_color=AZUL, opacity=0.55)
    fig.add_scatter(x=x, y=stats.norm.pdf(x, r.mean(), r.std(ddof=1)), name="Normal", line=dict(color=ROJO, width=2))
    fig.add_scatter(x=x, y=stats.t.pdf(x, gl, loc_t, esc_t), name=f"t (gl = {gl:.1f})", line=dict(color=NARANJA, width=2))
    fig.update_xaxes(tickformat=".1%")
    mostrar(fig, "Histograma vs. distribuciones ajustadas", AY["g_hist"])

    (osm, osr), (pendiente, intercepto, _) = stats.probplot(r, dist="norm")
    fig = go.Figure()
    fig.add_scatter(x=osm, y=osr, mode="markers", name="Datos", marker=dict(color=AZUL, size=4))
    fig.add_scatter(x=osm, y=pendiente * osm + intercepto, name="Normal", line=dict(color=ROJO))
    fig.update_xaxes(title="Cuantiles teóricos (normal)")
    fig.update_yaxes(title="Cuantiles observados", tickformat=".0%")
    mostrar(fig, "Gráfico Q-Q", AY["g_qq"])

# ------------------------------------------------------------------ VaR y ES
with tab_var:
    tabla = riesgo.tabla_var_es(r, confianza, int(horizonte))
    st.subheader(f"VaR y ES al {confianza:.1%} a {int(horizonte)} día(s)", help=AY["var_es"])

    c = st.columns(2)
    hist = tabla.loc[tabla["Método"] == "Histórico"].iloc[0]
    c[0].metric("VaR histórico", pct(hist["VaR"]), help=AY["var_hist"])
    c[0].caption(f"≈ {hist['VaR'] * posicion:,.0f} sobre una posición de {posicion:,.0f}")
    c[1].metric("ES histórico", pct(hist["ES"]), help=AY["es_hist"])
    c[1].caption(f"≈ {hist['ES'] * posicion:,.0f} sobre una posición de {posicion:,.0f}")

    st.markdown("**Comparación de métodos**", help=AY["metodos"])
    vista = tabla.copy()
    vista["VaR $"] = vista["VaR"] * posicion
    vista["ES $"] = vista["ES"] * posicion
    vista = vista[["Método", "VaR", "ES", "VaR $", "ES $", "Notas"]].set_index("Método")
    st.dataframe(
        vista.style.format(
            {"VaR": "{:.2%}", "ES": "{:.2%}", "VaR $": "{:,.0f}", "ES $": "{:,.0f}"}, na_rep="—"
        )
    )

    fig = go.Figure()
    fig.add_bar(x=tabla["Método"], y=tabla["VaR"], name="VaR", marker_color=AZUL)
    fig.add_bar(x=tabla["Método"], y=tabla["ES"], name="ES", marker_color=ROJO)
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".1%")
    mostrar(fig, "Pérdida (% de la posición)", AY["g_barras"])

    st.caption(
        "Para horizontes mayores a 1 día se usa la regla de la raíz del tiempo. "
        "Para un solo activo, Monte Carlo converge a la t de Student paramétrica; su ventaja aparece en portafolios."
    )

# ------------------------------------------------------------------ volatilidad
with tab_vol:
    v21 = vol.anualizar(vol.vol_movil(r, 21))
    vew = vol.anualizar(vol.vol_ewma(r, lam))
    fig = go.Figure()
    fig.add_scatter(x=v21.index, y=v21, name="Móvil 21 días", line=dict(color=AZUL, width=1.3))
    fig.add_scatter(x=vew.index, y=vew, name=f"EWMA (λ = {lam:.2f})", line=dict(color=NARANJA, width=1.3))
    fig.add_hline(y=res["vol_anual"], line_dash="dash", line_color=GRIS, annotation_text="Volatilidad de toda la muestra")
    fig.update_yaxes(tickformat=".0%")
    mostrar(fig, "Volatilidad anualizada", AY["g_vol"])

# ------------------------------------------------------------------ backtesting
with tab_bt:
    if len(r) <= ventana + 60:
        st.info("Amplía el periodo o reduce la ventana móvil para poder hacer backtesting.")
    else:
        metodo = st.radio(
            "Modelo de VaR a evaluar",
            list(riesgo.METODOS_MOVILES),
            horizontal=True,
            format_func=lambda m: {"Histórico": "Histórico móvil", "Normal": "Normal móvil", "EWMA": "EWMA"}[m],
            help=AY["bt_modelo"],
        )
        var_serie = riesgo.var_movil(r, metodo, confianza, ventana, lam)
        viol, var_prev = bt.alinear(r, var_serie)
        resultado = bt.evaluar(viol.to_numpy(), confianza)

        c = st.columns(4)
        c[0].metric("Días evaluados", f"{resultado.n:,}", help=AY["bt_dias"])
        c[1].metric("Violaciones", f"{resultado.violaciones}", help=AY["bt_viol"])
        c[2].metric("Tasa observada", pct(resultado.tasa_observada), help=AY["bt_obs"])
        c[3].metric("Tasa esperada", pct(resultado.tasa_esperada), help=AY["bt_esp"])

        r_eval = r.loc[var_prev.index]
        fig = go.Figure()
        fig.add_scatter(x=r_eval.index, y=r_eval, name="Rendimiento", line=dict(color=GRIS, width=0.8))
        fig.add_scatter(x=var_prev.index, y=-var_prev, name="-VaR", line=dict(color=AZUL, width=1.5))
        fig.add_scatter(
            x=r_eval.index[viol.to_numpy()],
            y=r_eval[viol.to_numpy()],
            mode="markers",
            name="Violación",
            marker=dict(color=ROJO, size=7),
        )
        fig.update_yaxes(tickformat=".0%")
        mostrar(fig, "Rendimientos vs. -VaR", AY["g_bt"])

        st.markdown("**Pruebas estadísticas**", help=AY["pruebas"])
        tabla_bt = pd.DataFrame(
            {
                "Prueba": ["Kupiec (cobertura)", "Christoffersen (independencia)", "Cobertura condicional"],
                "Estadístico LR": [resultado.lr_pof, resultado.lr_ind, resultado.lr_cc],
                "p-valor": [resultado.p_pof, resultado.p_ind, resultado.p_cc],
                "Resultado (5 %)": [
                    "Rechaza" if p < 0.05 else "No rechaza" for p in (resultado.p_pof, resultado.p_ind, resultado.p_cc)
                ],
                "Qué evalúa": [AY["kupiec"], AY["christoffersen"], AY["cc"]],
            }
        )
        st.dataframe(tabla_bt.style.format({"Estadístico LR": "{:.3f}", "p-valor": "{:.4f}"}), hide_index=True)

        if resultado.p_pof < 0.05 and resultado.tasa_observada > resultado.tasa_esperada:
            st.warning("El modelo subestima el riesgo: hubo más violaciones de las esperadas.")
        elif resultado.p_pof < 0.05:
            st.info("El modelo es demasiado conservador: hubo menos violaciones de las esperadas.")
        else:
            st.success("La frecuencia de violaciones es consistente con el nivel de confianza.")
        if resultado.p_ind < 0.05:
            st.warning("Las violaciones se agrupan en el tiempo: el modelo reacciona lento a los cambios de volatilidad.")

st.divider()
st.caption(
    "Herramienta informativa y educativa; no constituye asesoría financiera. "
    "Los datos provienen de Yahoo Finance y pueden contener errores o retrasos. "
    "Rendimientos pasados no garantizan resultados futuros."
)
