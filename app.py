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

# ------------------------------------------------------------------
# Carga de datos y modelos (con cache para que no se recargue en cada clic)
# ------------------------------------------------------------------
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
    return ultima_fila, historico, mape_familia

paquete_modelos = cargar_modelos()
modelos = paquete_modelos["modelos"]
FEATURES = paquete_modelos["features"]
FEATURE_CAT = paquete_modelos["feature_cat"]

ultima_fila, historico, mape_familia = cargar_datos()

@st.cache_data
def calcular_predicciones_todos_los_productos(_modelos, _features, _feature_cat, _ultima_fila):
    """Calcula la prediccion de % de crecimiento y produccion para los 3 horizontes,
    para TODOS los productos a la vez -- es lo que necesita el ranking."""
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
            "Horizonte": f"t+{horizonte}",
            "pred_growth_pct": pred_growth,
        })
        df_h["Pred_Production"] = df_h["Production_t"] * (1 + df_h["pred_growth_pct"] / 100)
        resultados.append(df_h)

    return pd.concat(resultados, ignore_index=True)

ranking_todos = calcular_predicciones_todos_los_productos(modelos, FEATURES, FEATURE_CAT, ultima_fila)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("Forecasting Agricola")
st.sidebar.markdown("Sistema de prediccion de produccion mundial por producto, basado en FAOSTAT (QCL).")
st.sidebar.markdown("---")

modo_vista = st.sidebar.radio("Modo de vista", ["Analizar un producto", "Ranking de crecimiento"])

if modo_vista == "Analizar un producto":
    productos = sorted(ultima_fila["Item"].unique())
    indice_default = productos.index("Wheat") if "Wheat" in productos else 0
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto", productos, index=indice_default)
else:
    horizonte_ranking = st.sidebar.selectbox("Horizonte del ranking", ["t+1", "t+3", "t+5"], index=0)
    n_mostrar = st.sidebar.slider("Cuantos productos mostrar en cada extremo", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sobre el modelo**: LightGBM entrenado unicamente con datos de produccion, "
    "superficie y rendimiento (QCL). Se comprobo que anadir datos de comercio (TCL) "
    "y valor economico (QV) no mejora la prediccion con el enfoque actual -- ver "
    "detalle metodologico en el notebook del proyecto."
)

# ==========================================================================
# MODO: RANKING DE CRECIMIENTO
# ==========================================================================
if modo_vista == "Ranking de crecimiento":
    st.title(f"Ranking de crecimiento predicho — {horizonte_ranking}")
    st.caption(
        "Productos con mayor y menor variacion de produccion predicha (pred_growth_pct), "
        "segun el modelo LightGBM (Experimento A, solo QCL)."
    )

    df_h = ranking_todos[ranking_todos["Horizonte"] == horizonte_ranking].copy()
    df_h = df_h.sort_values("pred_growth_pct", ascending=False)

    top_suben = df_h.head(n_mostrar)
    top_bajan = df_h.tail(n_mostrar).sort_values("pred_growth_pct", ascending=True)

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader(f"Top {n_mostrar} que MAS suben")
        fig1, ax1 = plt.subplots(figsize=(7, max(4, n_mostrar * 0.35)))
        ax1.barh(top_suben["Item"][::-1], top_suben["pred_growth_pct"][::-1], color="#55A868")
        ax1.set_xlabel("Crecimiento predicho (%)")
        ax1.set_title(f"Mayor crecimiento — {horizonte_ranking}")
        st.pyplot(fig1)

    with col_der:
        st.subheader(f"Top {n_mostrar} que MAS bajan")
        fig2, ax2 = plt.subplots(figsize=(7, max(4, n_mostrar * 0.35)))
        ax2.barh(top_bajan["Item"][::-1], top_bajan["pred_growth_pct"][::-1], color="#C44E52")
        ax2.set_xlabel("Crecimiento predicho (%)")
        ax2.set_title(f"Mayor caida — {horizonte_ranking}")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Tabla completa del ranking")

    tabla_ranking = df_h[["Item", "Familia", "Production_t", "pred_growth_pct", "Pred_Production"]].copy()
    tabla_ranking.columns = ["Producto", "Familia", "Produccion actual (t)", "Crecimiento predicho (%)", "Produccion predicha (t)"]
    tabla_ranking["Crecimiento predicho (%)"] = tabla_ranking["Crecimiento predicho (%)"].round(2)
    tabla_ranking["Produccion actual (t)"] = tabla_ranking["Produccion actual (t)"].round(0)
    tabla_ranking["Produccion predicha (t)"] = tabla_ranking["Produccion predicha (t)"].round(0)

    st.dataframe(tabla_ranking, use_container_width=True, hide_index=True)

    st.download_button(
        f"Descargar ranking completo {horizonte_ranking} (CSV)",
        tabla_ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_crecimiento_{horizonte_ranking.replace('+','')}.csv",
        mime="text/csv",
    )

    st.caption(
        "Nota: estas predicciones no incorporan un indicador de confianza por fila -- para eso, "
        "consulta el modo 'Analizar un producto', que muestra el error tipico esperado segun la "
        "familia de cada producto."
    )

# ==========================================================================
# MODO: ANALIZAR UN PRODUCTO (vista original)
# ==========================================================================
else:
    fila = ultima_fila[ultima_fila["Item"] == producto_seleccionado].iloc[0]
    familia = fila["Familia"]

    st.title(producto_seleccionado)
    st.caption(f"Familia: {familia}  |  Ultimo dato disponible: {int(fila['Year'])} - Produccion: {fila['Production_t']:,.0f} toneladas")

    col1, col2, col3 = st.columns(3)

    X_pred = fila[FEATURES + [FEATURE_CAT]].to_frame().T
    X_pred[FEATURE_CAT] = X_pred[FEATURE_CAT].astype("category")
    for c in FEATURES:
        X_pred[c] = pd.to_numeric(X_pred[c])

    predicciones = {}
    for horizonte, col in zip([1, 3, 5], [col1, col2, col3]):
        modelo = modelos[horizonte]
        pred_growth_pct = modelo.predict(X_pred)[0]
        pred_produccion = fila["Production_t"] * (1 + pred_growth_pct / 100)
        predicciones[horizonte] = {"growth_pct": pred_growth_pct, "produccion": pred_produccion}

        mape_esperado = mape_familia.get(familia, {}).get(f"t{horizonte}", None)

        with col:
            anio_objetivo = int(fila["Year"]) + horizonte
            st.metric(
                label=f"Prediccion {anio_objetivo} (t+{horizonte})",
                value=f"{pred_produccion:,.0f} t",
                delta=f"{pred_growth_pct:+.1f}% vs ultimo dato"
            )
            if mape_esperado is not None:
                if mape_esperado < 8:
                    nivel, icono = "Alta", "[Alta]"
                elif mape_esperado < 15:
                    nivel, icono = "Media", "[Media]"
                else:
                    nivel, icono = "Baja", "[Baja]"
                st.caption(f"{icono} Confianza esperada: **{nivel}** (error tipico de la familia *{familia}* en este horizonte: ~{mape_esperado:.1f}%)")

    st.markdown("---")
    st.subheader("Evolucion historica y prediccion")

    hist_producto = historico[historico["Item"] == producto_seleccionado].sort_values("Year")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist_producto["Year"], hist_producto["Production_t"], label="Produccion historica", linewidth=2, color="#4C72B0")

    anios_futuros = [int(fila["Year"]) + h for h in [1, 3, 5]]
    valores_futuros = [predicciones[h]["produccion"] for h in [1, 3, 5]]
    ax.plot([int(fila["Year"])] + anios_futuros, [fila["Production_t"]] + valores_futuros,
            label="Prediccion", linestyle="--", marker="o", color="#C44E52")

    ax.set_xlabel("Ano")
    ax.set_ylabel("Produccion (toneladas)")
    ax.set_title(f"Produccion mundial de {producto_seleccionado}")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Por que el modelo predice esto (interpretabilidad SHAP, horizonte t+1)")

    explainer = shap.TreeExplainer(modelos[1])
    shap_values = explainer.shap_values(X_pred)

    df_shap = pd.DataFrame({
        "Variable": FEATURES + [FEATURE_CAT],
        "Impacto_SHAP": shap_values[0]
    }).sort_values("Impacto_SHAP", key=abs, ascending=True)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    colores = ["#C44E52" if v < 0 else "#55A868" for v in df_shap["Impacto_SHAP"]]
    ax2.barh(df_shap["Variable"], df_shap["Impacto_SHAP"], color=colores)
    ax2.set_xlabel("Impacto en la prediccion (puntos de % de crecimiento)")
    ax2.set_title(f"Explicacion de la prediccion para {producto_seleccionado} (t+1)")
    ax2.axvline(0, color="black", linewidth=0.8)
    st.pyplot(fig2)

    st.caption(
        "Barras verdes: empujan la prediccion hacia un crecimiento mayor. "
        "Barras rojas: empujan hacia un crecimiento menor (o caida)."
    )

    st.markdown("---")
    st.subheader("Resumen de predicciones")

    tabla_resumen = pd.DataFrame([
        {
            "Horizonte": f"t+{h}",
            "Ano objetivo": int(fila["Year"]) + h,
            "Produccion predicha (t)": round(predicciones[h]["produccion"]),
            "Crecimiento predicho (%)": round(predicciones[h]["growth_pct"], 2),
            "Confianza esperada (MAPE tipico %)": mape_familia.get(familia, {}).get(f"t{h}", "N/D"),
        }
        for h in [1, 3, 5]
    ])
    st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)

    st.download_button(
        "Descargar prediccion (CSV)",
        tabla_resumen.to_csv(index=False).encode("utf-8"),
        file_name=f"prediccion_{producto_seleccionado.replace(' ','_').replace(',','')}.csv",
        mime="text/csv",
    )
