"""
Dashboard de Forecasting de Produccion Agricola Mundial (FAOSTAT)
TFM - Sistema de Alerta Temprana / Analisis de Produccion

Modelo: LightGBM, Experimento A (solo QCL), target = % de crecimiento interanual.
Justificacion de esta eleccion: ver notebook de analisis (seccion 11-12) -- se comprobo
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

# ------------------------------------------------------------------
# Sidebar: seleccion de producto
# ------------------------------------------------------------------
st.sidebar.title("Forecasting Agricola")
st.sidebar.markdown("Sistema de prediccion de produccion mundial por producto, basado en FAOSTAT (QCL).")

productos = sorted(ultima_fila["Item"].unique())
indice_default = productos.index("Wheat") if "Wheat" in productos else 0
producto_seleccionado = st.sidebar.selectbox("Selecciona un producto", productos, index=indice_default)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sobre el modelo**: LightGBM entrenado unicamente con datos de produccion, "
    "superficie y rendimiento (QCL). Se comprobo que anadir datos de comercio (TCL) "
    "y valor economico (QV) no mejora la prediccion con el enfoque actual -- ver "
    "detalle metodologico en el notebook del proyecto."
)

# ------------------------------------------------------------------
# Cuerpo principal
# ------------------------------------------------------------------
fila = ultima_fila[ultima_fila["Item"] == producto_seleccionado].iloc[0]
familia = fila["Familia"]

st.title(producto_seleccionado)
st.caption(f"Familia: {familia}  |  Ultimo dato disponible: {int(fila['Year'])} - Produccion: {fila['Production_t']:,.0f} toneladas")

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------------
# Predicciones para t+1, t+3, t+5
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Grafico historico + predicciones
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Interpretabilidad (SHAP) para la prediccion a 1 ano
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Tabla resumen descargable
# ------------------------------------------------------------------
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
