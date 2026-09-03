"""
Dashboard de Forecasting de Produccion Agricola Mundial (FAOSTAT)
TFM - Prediccion multihorizonte de la produccion agricola mundial

Modelo: LightGBM, Experimento A (solo QCL), target = % de crecimiento interanual.
Justificacion de esta eleccion: ver notebook de analisis (secciones 11-12) -- se comprobo
empiricamente que anadir TCL/QV no mejora el resultado con los datos y el enfoque actuales.

Para ejecutar localmente:
    pip install streamlit pandas numpy lightgbm shap matplotlib
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Forecasting Agricola FAOSTAT", layout="wide")

# ==========================================================================
# Carga de datos y modelos
# ==========================================================================
@st.cache_resource
def cargar_modelos():
    with open("modelos_dashboard.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def cargar_datos():
    ultima_fila = pd.read_csv("app_ultima_fila_por_item.csv")
    historico = pd.read_csv("app_historico_produccion.csv")
    with open("mape_por_familia.json") as f:
        mape_familia = json.load(f)
    backtest = pd.read_csv("app_backtest_predictions.csv")
    return ultima_fila, historico, mape_familia, backtest

paquete_modelos = cargar_modelos()
modelos = paquete_modelos["modelos"]
FEATURES = paquete_modelos["features"]
FEATURE_CAT = paquete_modelos["feature_cat"]

ultima_fila, historico, mape_familia, backtest = cargar_datos()

@st.cache_data
def calcular_predicciones_todos_los_productos(_modelos, _features, _feature_cat, _ultima_fila):
    """Prediccion de % de crecimiento y produccion para los 3 horizontes, para TODOS los productos.
    Ademas marca predicciones estadisticamente anomalas: cuando la magnitud del crecimiento
    predicho supera en mucho la volatilidad historica normal de ESE producto especifico -- esto
    puede pasar incluso en familias con buena confianza promedio (el MAPE de familia es un
    agregado, no protege contra un caso individual inestable)."""
    resultados = []
    X_todos = _ultima_fila[_features + [_feature_cat]].copy()
    X_todos[_feature_cat] = X_todos[_feature_cat].astype("category")
    for c in _features:
        X_todos[c] = pd.to_numeric(X_todos[c])

    for horizonte in [1, 3, 5]:
        modelo = _modelos[horizonte]
        pred_growth = modelo.predict(X_todos)
        df_h = pd.DataFrame({
            "Item": _ultima_fila["Item"].values,
            "Familia": _ultima_fila["Familia"].values,
            "Production_t": _ultima_fila["Production_t"].values,
            "Vol_cv5": _ultima_fila["Production_vol_cv5"].values,
            "Horizonte": f"t+{horizonte}",
            "pred_growth_pct": pred_growth,
        })
        df_h["Pred_Production"] = df_h["Production_t"] * (1 + df_h["pred_growth_pct"] / 100)
        # Umbral de anomalia: la magnitud del crecimiento predicho supera 5x la volatilidad
        # historica normal (coeficiente de variacion) de ese producto especifico
        umbral = 5 * (df_h["Vol_cv5"].fillna(0) * 100)
        df_h["Posible_anomalia"] = df_h["pred_growth_pct"].abs() > umbral.clip(lower=15)
        resultados.append(df_h)

    return pd.concat(resultados, ignore_index=True)

ranking_todos = calcular_predicciones_todos_los_productos(modelos, FEATURES, FEATURE_CAT, ultima_fila)

def nivel_confianza(mape):
    if mape is None:
        return "N/D", "[N/D]"
    if mape < 8:
        return "Alta", "[Alta]"
    elif mape < 15:
        return "Media", "[Media]"
    return "Baja", "[Baja]"

def predecir_producto(item, overrides=None):
    """Genera la prediccion para un producto, permitiendo sobreescribir features (para el simulador)."""
    fila = ultima_fila[ultima_fila["Item"] == item].iloc[0].copy()
    if overrides:
        for k, v in overrides.items():
            fila[k] = v

    X_pred = fila[FEATURES + [FEATURE_CAT]].to_frame().T
    X_pred[FEATURE_CAT] = X_pred[FEATURE_CAT].astype("category")
    for c in FEATURES:
        X_pred[c] = pd.to_numeric(X_pred[c])

    predicciones = {}
    for horizonte in [1, 3, 5]:
        pred_growth_pct = modelos[horizonte].predict(X_pred)[0]
        pred_produccion = fila["Production_t"] * (1 + pred_growth_pct / 100)
        predicciones[horizonte] = {"growth_pct": pred_growth_pct, "produccion": pred_produccion}
    return fila, X_pred, predicciones

# ==========================================================================
# Sidebar
# ==========================================================================
st.sidebar.title("Forecasting Agricola")
st.sidebar.markdown("Sistema de prediccion de produccion mundial por producto, basado en FAOSTAT (QCL).")
st.sidebar.markdown("---")

modo_vista = st.sidebar.radio(
    "Modo de vista",
    ["Analizar un producto", "Ranking de crecimiento", "Cuadrante crecimiento vs confianza", "Comparador de productos"]
)

productos_todos = sorted(ultima_fila["Item"].unique())
familias_todas = sorted(ultima_fila["Familia"].unique())

if modo_vista == "Analizar un producto":
    indice_default = productos_todos.index("Wheat") if "Wheat" in productos_todos else 0
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto", productos_todos, index=indice_default)

elif modo_vista == "Ranking de crecimiento":
    horizonte_ranking = st.sidebar.selectbox("Horizonte del ranking", ["t+1", "t+3", "t+5"], index=0)
    n_mostrar = st.sidebar.slider("Cuantos productos mostrar en cada extremo", 5, 20, 10)
    familias_filtro = st.sidebar.multiselect("Filtrar por familia (opcional)", familias_todas, default=[])
    ocultar_anomalias = st.sidebar.checkbox("Ocultar predicciones estadisticamente anomalas", value=True,
        help="Oculta productos cuya prediccion se aleja mucho de su propia volatilidad historica -- suelen ser inestabilidades puntuales del modelo, no tendencias reales.")

elif modo_vista == "Cuadrante crecimiento vs confianza":
    horizonte_cuadrante = st.sidebar.selectbox("Horizonte", ["t+1", "t+3", "t+5"], index=0)

elif modo_vista == "Comparador de productos":
    productos_comparar = st.sidebar.multiselect(
        "Selecciona 2-3 productos", productos_todos, default=["Wheat", "Maize (corn)"]
    )
    modo_escala = st.sidebar.radio(
        "Escala del grafico",
        ["Indice (base 100 = primer ano)", "Toneladas absolutas"],
        help="Si comparas productos de volumenes muy distintos (ej. Wheat vs Almonds), usa Indice -- si no, los productos pequenos se ven aplastados contra el 0."
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sobre el modelo**: LightGBM entrenado unicamente con datos de produccion, "
    "superficie y rendimiento (QCL). Se comprobo que anadir datos de comercio (TCL) "
    "y valor economico (QV) no mejora la prediccion con el enfoque actual."
)

# ==========================================================================
# MODO: ANALIZAR UN PRODUCTO
# ==========================================================================
if modo_vista == "Analizar un producto":
    fila, X_pred, predicciones = predecir_producto(producto_seleccionado)
    familia = fila["Familia"]

    st.title(producto_seleccionado)
    st.caption(f"Familia: {familia}  |  Ultimo dato disponible: {int(fila['Year'])} - Produccion: {fila['Production_t']:,.0f} toneladas")

    col1, col2, col3 = st.columns(3)
    for horizonte, col in zip([1, 3, 5], [col1, col2, col3]):
        mape_esperado = mape_familia.get(familia, {}).get(f"t{horizonte}", None)
        nivel, icono = nivel_confianza(mape_esperado)
        with col:
            anio_objetivo = int(fila["Year"]) + horizonte
            st.metric(
                label=f"Prediccion {anio_objetivo} (t+{horizonte})",
                value=f"{predicciones[horizonte]['produccion']:,.0f} t",
                delta=f"{predicciones[horizonte]['growth_pct']:+.1f}% vs ultimo dato"
            )
            if mape_esperado is not None:
                st.caption(f"{icono} Confianza esperada: **{nivel}** (error tipico de *{familia}*: ~{mape_esperado:.1f}%)")

    st.markdown("---")
    st.subheader("Evolucion historica, backtesting y prediccion")
    st.caption(
        "La linea azul es produccion real. Los puntos naranjas (2017-2024) son lo que el modelo "
        "HABRIA predicho en su momento, usando solo datos anteriores a esa fecha -- asi se puede "
        "comparar contra la produccion real de esos mismos anios. Los puntos rojos son la prediccion "
        "hacia el futuro, todavia sin dato real con el que comparar."
    )

    hist_producto = historico[historico["Item"] == producto_seleccionado].sort_values("Year")
    backtest_producto = backtest[(backtest["Item"] == producto_seleccionado) & (backtest["Horizonte"] == "t+1")].sort_values("Anio_prediccion")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist_producto["Year"], hist_producto["Production_t"], label="Produccion historica (real)", linewidth=2, color="#4C72B0")

    if len(backtest_producto):
        ax.plot(backtest_producto["Anio_prediccion"], backtest_producto["Pred_Production"],
                label="Backtesting (prediccion t+1 en el pasado)", linestyle=":", marker="s", color="#DD8452", markersize=6)

    anios_futuros = [int(fila["Year"]) + h for h in [1, 3, 5]]
    valores_futuros = [predicciones[h]["produccion"] for h in [1, 3, 5]]
    ax.plot([int(fila["Year"])] + anios_futuros, [fila["Production_t"]] + valores_futuros,
            label="Prediccion (futuro)", linestyle="--", marker="o", color="#C44E52")

    ax.set_xlabel("Ano"); ax.set_ylabel("Produccion (toneladas)")
    ax.set_title(f"Produccion mundial de {producto_seleccionado}")
    ax.legend()
    st.pyplot(fig)

    if len(backtest_producto):
        error_medio = np.mean(np.abs((backtest_producto["Real_Production"] - backtest_producto["Pred_Production"]) / backtest_producto["Real_Production"])) * 100
        st.caption(f"Error medio del backtesting (t+1) para {producto_seleccionado}: {error_medio:.1f}%")

    st.markdown("---")
    st.subheader("Por que el modelo predice esto (SHAP, horizonte t+1)")

    explainer = shap.TreeExplainer(modelos[1])
    shap_values = explainer.shap_values(X_pred)
    df_shap = pd.DataFrame({"Variable": FEATURES + [FEATURE_CAT], "Impacto_SHAP": shap_values[0]}).sort_values("Impacto_SHAP", key=abs, ascending=True)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    colores = ["#C44E52" if v < 0 else "#55A868" for v in df_shap["Impacto_SHAP"]]
    ax2.barh(df_shap["Variable"], df_shap["Impacto_SHAP"], color=colores)
    ax2.set_xlabel("Impacto en la prediccion (puntos de % de crecimiento)")
    ax2.axvline(0, color="black", linewidth=0.8)
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Simulador: ¿que pasaria si...?")
    st.caption("Ajusta manualmente la tendencia reciente o el rendimiento y observa como cambia la prediccion a 1 ano.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        ajuste_tendencia_pct = st.slider("Variacion de la tendencia reciente (%)", -100, 200, 0, step=10)
    with col_sim2:
        ajuste_yield_pct = st.slider("Variacion del rendimiento -- Yield (%)", -80, 200, 0, step=10)

    if ajuste_tendencia_pct != 0 or ajuste_yield_pct != 0:
        nuevo_slope = fila["Production_trend_slope5"] * (1 + ajuste_tendencia_pct / 100)
        nuevo_yield = fila["Yield_hg_ha"] * (1 + ajuste_yield_pct / 100)
        _, _, pred_simulada = predecir_producto(
            producto_seleccionado,
            overrides={"Production_trend_slope5": nuevo_slope, "Yield_hg_ha": nuevo_yield}
        )
        delta_vs_original = pred_simulada[1]["produccion"] - predicciones[1]["produccion"]
        st.metric(
            "Prediccion t+1 simulada",
            f"{pred_simulada[1]['produccion']:,.0f} t",
            delta=f"{delta_vs_original:+,.0f} t vs prediccion original"
        )
        if abs(delta_vs_original) < 1:
            st.caption(
                "El modelo no cambio su prediccion con este ajuste. Esto es un comportamiento normal "
                "de los modelos de arboles de decision (LightGBM): la prediccion solo cambia cuando el "
                "nuevo valor cruza un umbral de decision interno del arbol, no de forma proporcional y "
                "continua como en un modelo lineal. Prueba con un ajuste mas grande o de signo contrario."
            )
    else:
        st.info("Mueve alguno de los controles para simular un escenario distinto al actual.")

    st.markdown("---")
    st.subheader("Resumen de predicciones")
    tabla_resumen = pd.DataFrame([
        {
            "Horizonte": f"t+{h}", "Ano objetivo": int(fila["Year"]) + h,
            "Produccion predicha (t)": round(predicciones[h]["produccion"]),
            "Crecimiento predicho (%)": round(predicciones[h]["growth_pct"], 2),
            "Confianza esperada (MAPE tipico %)": mape_familia.get(familia, {}).get(f"t{h}", "N/D"),
        } for h in [1, 3, 5]
    ])
    st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)
    st.download_button("Descargar prediccion (CSV)", tabla_resumen.to_csv(index=False).encode("utf-8"),
                        file_name=f"prediccion_{producto_seleccionado.replace(' ','_').replace(',','')}.csv", mime="text/csv")

# ==========================================================================
# MODO: RANKING DE CRECIMIENTO
# ==========================================================================
elif modo_vista == "Ranking de crecimiento":
    st.title(f"Ranking de crecimiento predicho — {horizonte_ranking}")
    st.caption("Productos con mayor y menor variacion de produccion predicha, segun el modelo (Experimento A, solo QCL).")

    df_h = ranking_todos[ranking_todos["Horizonte"] == horizonte_ranking].copy()
    if familias_filtro:
        df_h = df_h[df_h["Familia"].isin(familias_filtro)]

    n_anomalas = df_h["Posible_anomalia"].sum()
    if ocultar_anomalias:
        df_h = df_h[~df_h["Posible_anomalia"]]
        if n_anomalas > 0:
            st.caption(f"Se ocultaron {n_anomalas} productos con predicciones estadisticamente anomalas (desactiva el filtro en la barra lateral para verlas).")

    df_h = df_h.sort_values("pred_growth_pct", ascending=False)

    if len(df_h) == 0:
        st.warning("No hay productos para ese filtro de familia.")
    else:
        n_ef = min(n_mostrar, len(df_h))
        top_suben = df_h.head(n_ef)
        top_bajan = df_h.tail(n_ef).sort_values("pred_growth_pct", ascending=True)

        col_izq, col_der = st.columns(2)
        with col_izq:
            st.subheader(f"Top {n_ef} que MAS suben")
            fig1, ax1 = plt.subplots(figsize=(7, max(4, n_ef * 0.35)))
            ax1.barh(top_suben["Item"][::-1], top_suben["pred_growth_pct"][::-1], color="#55A868")
            ax1.set_xlabel("Crecimiento predicho (%)")
            st.pyplot(fig1)
        with col_der:
            st.subheader(f"Top {n_ef} que MAS bajan")
            fig2, ax2 = plt.subplots(figsize=(7, max(4, n_ef * 0.35)))
            ax2.barh(top_bajan["Item"][::-1], top_bajan["pred_growth_pct"][::-1], color="#C44E52")
            ax2.set_xlabel("Crecimiento predicho (%)")
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Tabla completa del ranking")
        tabla_ranking = df_h[["Item", "Familia", "Production_t", "pred_growth_pct", "Pred_Production", "Posible_anomalia"]].copy()
        tabla_ranking.columns = ["Producto", "Familia", "Produccion actual (t)", "Crecimiento predicho (%)", "Produccion predicha (t)", "¿Posible anomalia?"]
        tabla_ranking["Crecimiento predicho (%)"] = tabla_ranking["Crecimiento predicho (%)"].round(2)
        st.dataframe(tabla_ranking, use_container_width=True, hide_index=True)
        st.download_button(f"Descargar ranking {horizonte_ranking} (CSV)", tabla_ranking.to_csv(index=False).encode("utf-8"),
                            file_name=f"ranking_{horizonte_ranking.replace('+','')}.csv", mime="text/csv")

# ==========================================================================
# MODO: CUADRANTE CRECIMIENTO VS CONFIANZA
# ==========================================================================
elif modo_vista == "Cuadrante crecimiento vs confianza":
    st.title(f"Cuadrante: crecimiento predicho vs confianza — {horizonte_cuadrante}")
    st.caption(
        "Cada punto es un producto. Eje X: % de crecimiento predicho. Eje Y: confianza esperada "
        "(100 - MAPE tipico de su familia -- mas arriba es mas confiable). Util para identificar "
        "oportunidades de alto crecimiento Y alta confianza (cuadrante superior derecho)."
    )

    h_num = horizonte_cuadrante.replace("t+", "")
    df_q = ranking_todos[ranking_todos["Horizonte"] == horizonte_cuadrante].copy()
    df_q["MAPE_familia"] = df_q["Familia"].apply(lambda f: mape_familia.get(f, {}).get(f"t{h_num}", 15))
    df_q["Confianza"] = 100 - df_q["MAPE_familia"]

    fig3, ax3 = plt.subplots(figsize=(11, 7))
    familias_unicas = df_q["Familia"].unique()
    cmap = plt.get_cmap("tab10")
    for i, fam in enumerate(sorted(familias_unicas)):
        sub = df_q[df_q["Familia"] == fam]
        ax3.scatter(sub["pred_growth_pct"], sub["Confianza"], label=fam, alpha=0.75, s=60, color=cmap(i % 10))

    ax3.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Crecimiento predicho (%)")
    ax3.set_ylabel("Confianza esperada (100 - MAPE tipico de la familia)")
    ax3.set_title(f"Cuadrante crecimiento vs confianza — {horizonte_cuadrante}")
    ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Mejores oportunidades (alto crecimiento + alta confianza)")
    mejores = df_q[df_q["Confianza"] > df_q["Confianza"].median()].sort_values("pred_growth_pct", ascending=False).head(10)
    st.dataframe(mejores[["Item","Familia","pred_growth_pct","Confianza"]].round(2), use_container_width=True, hide_index=True)

# ==========================================================================
# MODO: COMPARADOR DE PRODUCTOS
# ==========================================================================
elif modo_vista == "Comparador de productos":
    st.title("Comparador de productos")

    if len(productos_comparar) < 2:
        st.info("Selecciona al menos 2 productos en la barra lateral para comparar.")
    else:
        usar_indice = modo_escala.startswith("Indice")
        if usar_indice:
            st.caption(
                "Cada serie se muestra como % respecto a su propio valor en el primer ano disponible "
                "(base 100). Asi se pueden comparar tasas de crecimiento entre productos de volumenes "
                "muy distintos (ej. Wheat, que produce cientos de millones de toneladas, vs Almonds, "
                "que produce unos pocos millones) sin que el mas pequeno quede aplastado contra el 0."
            )

        fig4, ax4 = plt.subplots(figsize=(11, 5))
        tabla_comparativa = []
        for item in productos_comparar:
            hist_item = historico[historico["Item"] == item].sort_values("Year")
            fila_i, _, pred_i = predecir_producto(item)
            anios_fut = [int(fila_i["Year"]) + h for h in [1, 3, 5]]
            valores_fut = [pred_i[h]["produccion"] for h in [1, 3, 5]]

            if usar_indice:
                base = hist_item["Production_t"].iloc[0]
                y_hist = hist_item["Production_t"] / base * 100
                y_fut = [fila_i["Production_t"] / base * 100] + [v / base * 100 for v in valores_fut]
            else:
                y_hist = hist_item["Production_t"]
                y_fut = [fila_i["Production_t"]] + valores_fut

            linea, = ax4.plot(hist_item["Year"], y_hist, label=f"{item} (historico)", linewidth=2)
            ax4.plot([int(fila_i["Year"])] + anios_fut, y_fut, linestyle="--", marker="o", color=linea.get_color())

            for h in [1, 3, 5]:
                tabla_comparativa.append({
                    "Producto": item, "Horizonte": f"t+{h}",
                    "Produccion predicha (t)": round(pred_i[h]["produccion"]),
                    "Crecimiento (%)": round(pred_i[h]["growth_pct"], 2),
                })

        ax4.set_xlabel("Ano")
        ax4.set_ylabel("Indice (base 100 = primer ano)" if usar_indice else "Produccion (toneladas)")
        ax4.set_title("Comparacion historica y predicha")
        ax4.legend()
        st.pyplot(fig4)

        st.markdown("---")
        st.subheader("Tabla comparativa de predicciones")
        st.dataframe(pd.DataFrame(tabla_comparativa), use_container_width=True, hide_index=True)
