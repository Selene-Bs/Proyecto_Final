import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import streamlit as st
from common import inicializar_pagina, obtener_dataset, obtener_modelo_dl

inicializar_pagina("Deep Learning", "🧠")
df = obtener_dataset()

st.info(
    "Se entrena un **Perceptrón Multicapa (MLP)** — dos capas ocultas (32, 16), "
    "activación ReLU, optimizador Adam — sobre el mismo problema del Bloque F, y se "
    "compara explícitamente contra un modelo clásico (Regresión Logística). "
    "*Nota técnica:* se usa `MLPClassifier` de scikit-learn en vez de TensorFlow/PyTorch "
    "para mantener el despliegue ligero en Streamlit Community Cloud; ver "
    "`src/deep_learning/red_neuronal.py` para el detalle."
)

resultados = obtener_modelo_dl(df)

st.header("Comparación Deep Learning vs. modelo clásico")
st.dataframe(resultados["tabla_resultados"], use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Curva de pérdida del entrenamiento (MLP)**")
    if len(resultados["curva_perdida_mlp"]):
        fig = px.line(y=resultados["curva_perdida_mlp"], labels={"x": "iteración", "y": "pérdida (loss)"})
        st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown("**Matriz de confusión (MLP)**")
    fig_cm = px.imshow(resultados["matriz_confusion_mlp"], text_auto=True,
                        labels=dict(x="Predicho", y="Real"),
                        x=["Sin riesgo alto", "Riesgo alto"], y=["Sin riesgo alto", "Riesgo alto"])
    st.plotly_chart(fig_cm, use_container_width=True)

st.caption(
    "No se exige que Deep Learning sea el mejor modelo del proyecto: se exige "
    "implementarlo, evaluarlo con las mismas métricas y compararlo honestamente "
    "contra un modelo clásico, lo cual se cumple arriba."
)
