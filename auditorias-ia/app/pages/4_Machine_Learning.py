import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import plotly.express as px
import streamlit as st
from common import inicializar_pagina, obtener_dataset, obtener_modelos_ml
from src.ml.modelos import predecir_riesgo
from src.config import LINEAS, TURNOS, AUDITORES

inicializar_pagina("Machine Learning", "🤖")
df = obtener_dataset()

st.info(
    "**Problema supervisado:** clasificar si una auditoría terminará con al menos una "
    "respuesta *'No OK sin acción'* (riesgo alto), a partir de variables conocidas "
    "de antemano (línea, turno, auditor, antigüedad del operador, fin de mes, si está "
    "fuera de su matriz de polivalencia)."
)

resultados = obtener_modelos_ml(df)

st.header("Comparación de modelos (Bloque F y H)")
st.dataframe(resultados["tabla_resultados"], use_container_width=True)
st.caption(
    "Bloque H aplicado: separación train/test estratificada, validación cruzada (5-fold) "
    "reportada en `f1_cv_media`/`f1_cv_std`, y un **ensemble Voting (RF+GB+LR)**. "
    "El preprocesador (One-Hot + escalado) se ajusta únicamente con el set de entrenamiento "
    "dentro de un `Pipeline` de scikit-learn, evitando *data leakage*."
)

st.subheader(f"Mejor modelo: {resultados['mejor_modelo']}")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Matriz de confusión**")
    fig_cm = px.imshow(resultados["matriz_confusion"], text_auto=True,
                        labels=dict(x="Predicho", y="Real"),
                        x=["Sin riesgo alto", "Riesgo alto"], y=["Sin riesgo alto", "Riesgo alto"])
    st.plotly_chart(fig_cm, use_container_width=True)
with c2:
    st.markdown("**Reporte de clasificación**")
    st.text(resultados["reporte_texto"])

if resultados["importancias"] is not None:
    st.markdown("**Importancia de variables**")
    fig_imp = px.bar(resultados["importancias"], x="importancia", y="variable", orientation="h")
    st.plotly_chart(fig_imp, use_container_width=True)

st.divider()
st.header("Simulador de riesgo (usa el mejor modelo entrenado)")
c1, c2, c3 = st.columns(3)
linea = c1.selectbox("Línea", LINEAS)
turno = c2.selectbox("Turno", TURNOS)
auditor = c3.selectbox("Auditor", AUDITORES)
c4, c5, c6 = st.columns(3)
antiguedad = c4.slider("Antigüedad del operador (meses)", 1, 96, 24)
propension = c5.slider("Propensión de riesgo del operador (0-1)", 0.0, 1.0, 0.3)
fuera_matriz = c6.checkbox("¿Operador fuera de su matriz de polivalencia?")
fin_de_mes = st.checkbox("¿Es fin de mes (día ≥ 25)?")

pipe = resultados["pipelines"][resultados["mejor_modelo"]]
proba = predecir_riesgo(pipe, linea, turno, auditor, antiguedad, propension, fin_de_mes, fuera_matriz)
st.metric("Probabilidad estimada de hallazgo crítico ('No OK sin acción')", f"{proba:.1%}")
if proba > 0.5:
    st.error("Riesgo alto: se recomienda priorizar esta combinación línea/turno/operador en la próxima auditoría.")
elif proba > 0.3:
    st.warning("Riesgo medio.")
else:
    st.success("Riesgo bajo según el modelo.")

st.session_state["pipeline_riesgo"] = pipe  # se reutiliza en la página de Agente

st.divider()
st.header("Clustering de operadores y PCA (Bloque G)")
from src.ml.modelos import clustering_operadores
n_clusters = st.slider("Número de clusters", 2, 5, 3)
clu = clustering_operadores(df, n_clusters=n_clusters)
st.caption(f"Varianza explicada por los 2 componentes de PCA: {clu['varianza_explicada_pca']:.1%}")

fig_pca = px.scatter(clu["perfil"], x="pca_1", y="pca_2", color=clu["perfil"]["cluster"].astype(str),
                      hover_data=["operador_id", "score_promedio"],
                      title="Segmentación de operadores por patrón de desviaciones (PCA 2D)")
st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("**Perfil promedio por cluster** (tasa de desviación por categoría de pregunta)")
st.dataframe(clu["resumen_clusters"], use_container_width=True)
st.caption(
    "Utilidad para el negocio: permite dirigir capacitación específica -por ejemplo, un "
    "cluster con alta tasa de desviación en 'Seguridad' necesita refuerzo de EPP, mientras "
    "que otro con alta tasa en 'Calidad' necesita refuerzo de medición/vernier."
)
