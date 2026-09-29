import streamlit as st
from common import inicializar_pagina, obtener_dataset, obtener_senales

inicializar_pagina("Sistema Inteligente de Auditorías de Proceso", "🏭")

st.markdown(
    """
Bienvenido/a. Esta aplicación integra **todo el recorrido del Diplomado de
Python y Análisis de Datos** (Universidad Marista) sobre un problema real de
manufactura: la **auditoría de proceso en línea de producción**, con
preguntas como:

- ¿El operador trae puesto su EPP completo?
- ¿El operador utiliza el vernier para medir la pieza correctamente?
- ¿El operador sigue su instrucción de trabajo?
- ¿El operador se encuentra dentro de su matriz de polivalencia?

Cada pregunta se responde con una escala de 3 niveles: **OK**,
**No OK con acción** o **No OK sin acción**.
"""
)

df = obtener_dataset()
senales = obtener_senales()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Auditorías registradas", f"{len(df):,}")
col2.metric("Operadores", df["operador_id"].nunique())
col3.metric("Líneas de producción", df["linea"].nunique())
col4.metric("Score de cumplimiento promedio", f"{df['score_cumplimiento'].mean():.1%}")

st.divider()

st.subheader("🗺️ Mapa del proyecto (navega con el menú de la izquierda)")

mapa = [
    ("📊 Pipeline y EDA", "Bloques A y B", "Carga, limpieza, join con operadores, dashboard exploratorio."),
    ("📈 Estadística", "Bloque C", "Prueba de hipótesis (turno nocturno), correlación, intervalos de confianza."),
    ("⏱️ Series de Tiempo", "Bloques D y E", "Tendencia, estacionalidad, rolling, lags, feature engineering."),
    ("🤖 Machine Learning", "Bloques F, G y H", "Clasificación de riesgo, clustering de operadores, ensembles, validación cruzada."),
    ("🧠 Deep Learning", "Bloque I", "Red neuronal (MLP) comparada contra un modelo clásico."),
    ("💬 NLP y Embeddings", "Bloques J y K", "Clasificación de texto, TF-IDF vs. embeddings semánticos."),
    ("📄 RAG - Instrucciones", "Bloque M", "Preguntas sobre Instrucciones de Trabajo con fuentes citadas y evaluación del retriever."),
    ("🕹️ Agente IA", "Bloques L y N", "Router de herramientas (Tool Calling), tarea multi-step, LLM."),
    ("🌊 Fourier y Wavelets", "Bloques O y P", "Espectro de frecuencias y descomposición multiresolución de la señal dimensional."),
    ("📝 Registrar Auditoría", "Bloque A (captura)", "Formulario para capturar una nueva auditoría de piso."),
]

for nombre, bloque, desc in mapa:
    c1, c2, c3 = st.columns([2.2, 1.3, 5])
    c1.markdown(f"**{nombre}**")
    c2.caption(bloque)
    c3.write(desc)

st.divider()
with st.expander("ℹ️ Sobre los datos de esta demo"):
    st.markdown(
        """
        Este proyecto usa un **dataset sintético pero estadísticamente
        realista** (estacionalidad, efecto de turno, efecto de operador
        fuera de su matriz de polivalencia, tendencia de mejora en el
        tiempo) generado con `src/data/generate_synthetic_data.py`.

        Para usar datos reales de tu planta, reemplaza ese script por tu
        propia fuente (ERP, MES, Excel, base de datos) manteniendo el mismo
        esquema de columnas, y el resto del pipeline funciona sin cambios.
        """
    )
