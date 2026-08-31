# Dashboard de Forecasting Agrícola — FAOSTAT

Predice la producción mundial de 121 productos agrícolas a 1, 3 y 5 años, con interpretabilidad (SHAP)
e indicador de confianza por familia de producto.

**Modelo usado**: LightGBM, Experimento A (solo QCL — producción, superficie, rendimiento), target =
% de crecimiento interanual. Se eligió este y no QCL+TCL+QV porque, tras el experimento central del
proyecto (ver notebook, secciones 11-12), se comprobó que añadir comercio (TCL) y valor económico (QV)
empeora la predicción con el enfoque y los datos actuales.

## Archivos incluidos

- `app.py` — la aplicación Streamlit.
- `modelos_dashboard.pkl` — los 3 modelos entrenados (t+1, t+3, t+5), ya listos para usar.
- `app_ultima_fila_por_item.csv` — última fila de datos disponible de cada producto (input para predecir).
- `app_historico_produccion.csv` — serie histórica completa (1961-2024) de los 121 productos, para el gráfico.
- `mape_por_familia.json` — error típico por familia de producto y horizonte (para el indicador de confianza).
- `requirements.txt` — librerías necesarias.

## Cómo ejecutarlo en tu ordenador

1. Instala las dependencias (idealmente en un entorno virtual):
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # en Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Ejecuta la app desde la misma carpeta donde están todos estos archivos:
   ```bash
   streamlit run app.py
   ```

3. Se abrirá automáticamente en tu navegador (normalmente `http://localhost:8501`).

## Cómo desplegarlo gratis en internet (para que el equipo/tutores lo vean sin instalar nada)

1. Crea un repositorio en GitHub y sube estos 5 archivos (`app.py`, `modelos_dashboard.pkl`,
   `app_ultima_fila_por_item.csv`, `app_historico_produccion.csv`, `mape_por_familia.json`,
   `requirements.txt`).
2. Ve a [streamlit.io/cloud](https://streamlit.io/cloud) (Streamlit Community Cloud, gratis) e inicia
   sesión con tu cuenta de GitHub.
3. Click en "New app", selecciona el repositorio y el archivo `app.py`.
4. En un par de minutos tendrás una URL pública (tipo `tuusuario-tfm-faostat.streamlit.app`) que puedes
   incluir en el informe final y compartir con los tutores sin que tengan que instalar nada.

## Qué muestra el dashboard

1. **Selector de producto** en la barra lateral (los 121 productos elegibles del proyecto).
2. **3 métricas** con la predicción a 1, 3 y 5 años, con un indicador de confianza (Alta/Media/Baja)
   basado en el error típico de esa familia de producto (ver sección 10 del notebook de análisis).
3. **Gráfico** de la serie histórica completa junto con los puntos de predicción.
4. **Gráfico SHAP** explicando qué variables influyeron más en la predicción a 1 año, para ese producto
   concreto.
5. **Tabla descargable** con el resumen numérico de las 3 predicciones.

## Limitaciones a tener presentes (para mencionar en el informe)

- Las predicciones a horizontes largos (t+5) son menos fiables, especialmente para familias de producto
  pequeñas o volátiles (fibras, oleaginosas) — el indicador de confianza del dashboard ya avisa de esto.
- El modelo no usa datos de comercio ni valor económico (decisión justificada empíricamente, no una
  limitación de diseño accidental).
- Los modelos se entrenaron con datos hasta 2024; conviene reentrenar periódicamente a medida que
  FAOSTAT publique años nuevos.
