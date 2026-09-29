import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import streamlit as st
from common import inicializar_pagina, obtener_dataset
from src.config import PREGUNTA_IDS, PREGUNTA_TEXTO

inicializar_pagina("Pipeline de Datos y EDA", "📊")

df = obtener_dataset()

st.header("Pipeline de Datos (Bloque A)")
st.markdown(
    """
**De dónde llegan los datos:** tres archivos CSV que simulan la exportación
del sistema de auditorías de piso y del padrón de RH (`auditorias.csv`,
`operadores.csv`). **Transformaciones aplicadas:** tipado de columnas,
eliminación de duplicados por `audit_id`, imputación conservadora de
respuestas faltantes (`No OK sin acción`, nunca se asume cumplimiento),
marcado de mediciones dimensionales atípicas (>3 tolerancias), `merge` con
el catálogo de operadores y cálculo del score de cumplimiento.
"""
)
log = df.attrs.get("log_limpieza")
if log:
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicados removidos", log["duplicados_removidos"])
    c2.metric("Respuestas imputadas", log["filas_con_respuesta_faltante_imputada"])
    c3.metric("Mediciones atípicas marcadas", log["mediciones_atipicas_marcadas"])

with st.expander("Ver muestra del dataset final"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.header("Análisis Exploratorio de Datos (Bloque B)")

tab1, tab2, tab3, tab4 = st.tabs(["Distribución de respuestas", "Por línea/turno", "Segmentación operadores", "Estadísticas descriptivas"])

with tab1:
    pregunta_sel = st.selectbox("Selecciona una pregunta de la cédula", PREGUNTA_IDS, format_func=lambda p: PREGUNTA_TEXTO[p])
    conteo = df[pregunta_sel].value_counts().reset_index()
    conteo.columns = ["respuesta", "conteo"]
    fig = px.bar(conteo, x="respuesta", y="conteo", color="respuesta",
                 title=f"Distribución de respuestas — {PREGUNTA_TEXTO[pregunta_sel]}",
                 color_discrete_map={"OK": "#2ca02c", "No OK con acción": "#ff9f1c", "No OK sin acción": "#d62728"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hallazgo: las respuestas 'No OK sin acción' son el foco prioritario de mejora continua.")

with tab2:
    agrupador = st.radio("Agrupar por", ["linea", "turno"], horizontal=True)
    prom = df.groupby(agrupador, observed=True)["score_cumplimiento"].mean().reset_index()
    fig2 = px.bar(prom, x=agrupador, y="score_cumplimiento", title=f"Score de cumplimiento promedio por {agrupador}",
                  range_y=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x=agrupador, y="score_cumplimiento", title=f"Distribución del score por {agrupador}")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    fig4 = px.scatter(df, x="antiguedad_meses", y="score_cumplimiento", color="linea",
                       trendline="ols", opacity=0.4,
                       title="Score de cumplimiento vs. antigüedad del operador (por línea)")
    st.plotly_chart(fig4, use_container_width=True)

    fuera = df.groupby("operador_fuera_de_matriz", observed=True)["score_cumplimiento"].mean().reset_index()
    fuera["operador_fuera_de_matriz"] = fuera["operador_fuera_de_matriz"].map({True: "Fuera de matriz", False: "Dentro de matriz"})
    fig5 = px.bar(fuera, x="operador_fuera_de_matriz", y="score_cumplimiento",
                  title="Efecto de cubrir una línea fuera de la matriz de polivalencia")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Conclusión derivada de los datos: asignar operadores fuera de su matriz de "
        "polivalencia se asocia con una caída marcada en el score de cumplimiento, "
        "concentrada en la pregunta P4."
    )

with tab4:
    st.dataframe(df[["score_cumplimiento", "antiguedad_meses", "medicion_vernier_mm", "n_desviaciones"]].describe().round(3),
                 use_container_width=True)
