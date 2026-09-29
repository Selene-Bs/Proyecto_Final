import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import streamlit as st
from common import inicializar_pagina, obtener_dataset
from src.features.estadistica import (
    descriptivos_por_grupo, ic_por_linea, prueba_hipotesis_turno_nocturno,
    correlacion_antiguedad_score, matriz_correlacion,
)

inicializar_pagina("Estadística", "📈")
df = obtener_dataset()

st.markdown("Esta sección responde **preguntas concretas de negocio**, no ejecuta pruebas sin propósito.")

st.header("1) ¿El turno Nocturno cumple significativamente menos que el resto?")
res = prueba_hipotesis_turno_nocturno(df)
c1, c2, c3 = st.columns(3)
c1.metric("Media Nocturno", f"{res['media_nocturno']:.3f}")
c2.metric("Media resto de turnos", f"{res['media_resto']:.3f}")
c3.metric("p-valor (1 cola)", f"{res['p_valor_una_cola']:.4f}")
st.info(f"**H0:** {res['hipotesis_nula']}  \n**H1:** {res['hipotesis_alterna']}")
(st.success if res["rechaza_h0"] else st.warning)(res["interpretacion"])

st.divider()
st.header("2) ¿La antigüedad del operador se relaciona con su cumplimiento?")
corr = correlacion_antiguedad_score(df)
c1, c2 = st.columns(2)
c1.metric("Correlación de Pearson", f"{corr['pearson_r']:.3f}", help=f"p={corr['pearson_p']:.4f}")
c2.metric("Correlación de Spearman", f"{corr['spearman_r']:.3f}", help=f"p={corr['spearman_p']:.4f}")
st.write(corr["interpretacion"])
fig = px.scatter(df, x="antiguedad_meses", y="score_cumplimiento", trendline="ols", opacity=0.35,
                  title="Antigüedad (meses) vs. Score de cumplimiento")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("3) Intervalo de confianza (95%) del cumplimiento promedio por línea")
tabla_ic = ic_por_linea(df)
fig_ic = px.scatter(tabla_ic, x="linea", y="media", error_y=tabla_ic["ic_sup"] - tabla_ic["media"],
                     error_y_minus=tabla_ic["media"] - tabla_ic["ic_inf"],
                     title="Score promedio por línea con IC 95%")
st.plotly_chart(fig_ic, use_container_width=True)
st.dataframe(tabla_ic.round(4), use_container_width=True)
st.caption(
    "Interpretación: si los intervalos de dos líneas NO se traslapan, hay evidencia de una "
    "diferencia real de desempeño entre ellas, no solo variación muestral."
)

st.divider()
st.header("4) Comparación de grupos y matriz de correlación")
grupo = st.selectbox("Agrupar descriptivos por:", ["linea", "turno", "auditor"])
st.dataframe(descriptivos_por_grupo(df, "score_cumplimiento", grupo), use_container_width=True)

fig_corr = px.imshow(matriz_correlacion(df), text_auto=True, title="Matriz de correlación (variables numéricas)")
st.plotly_chart(fig_corr, use_container_width=True)
