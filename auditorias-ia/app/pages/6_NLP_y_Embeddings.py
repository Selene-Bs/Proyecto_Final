import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from common import inicializar_pagina, obtener_dataset, obtener_modelo_nlp

inicializar_pagina("NLP Clásico y Embeddings", "💬")
df = obtener_dataset()

st.header("NLP clásico (Bloque J)")
st.markdown(
    "Fuente textual: el comentario libre que escribe el auditor cuando encuentra una "
    "desviación. Se limpia el texto, se vectoriza con **TF-IDF** y se entrena un "
    "clasificador que predice la **severidad** de la auditoría (sin riesgo / riesgo medio "
    "/ riesgo alto) **usando solo el texto del comentario**, sin ver las respuestas."
)

nlp = obtener_modelo_nlp(df)
c1, c2 = st.columns(2)
c1.metric("Accuracy", nlp["metricas"]["accuracy"])
c2.metric("F1 macro", nlp["metricas"]["f1_macro"])
with st.expander("Ver reporte de clasificación completo"):
    st.text(nlp["metricas"]["reporte"])

st.subheader("Palabras clave por nivel de severidad")
cols = st.columns(3)
for col, (severidad, palabras) in zip(cols, nlp["palabras_clave"].items()):
    with col:
        st.markdown(f"**{severidad}**")
        st.write(", ".join(palabras))

st.divider()
st.header("Embeddings y Transformers (Bloque K)")
st.markdown(
    "Se representan los comentarios con **Sentence-Transformers** (modelo multilingüe) "
    "para hacer búsqueda semántica; si el modelo no está disponible en este entorno, la "
    "app se degrada de forma explícita a un embedding TF-IDF+SVD (nunca falla en silencio)."
)

consulta = st.text_input("Escribe una consulta en lenguaje natural sobre auditorías:",
                          "operador sin protección en los ojos")

if st.button("Buscar comentarios similares"):
    from src.nlp.embeddings import comparar_tfidf_vs_embeddings
    comparacion = comparar_tfidf_vs_embeddings(df, consulta, top_k=5)
    st.caption(f"Método de embeddings usado: **{comparacion['metodo_embeddings']}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ranking con TF-IDF (clásico)**")
        st.dataframe(comparacion["tfidf"], use_container_width=True)
    with c2:
        st.markdown("**Ranking con Embeddings (semántico)**")
        st.dataframe(comparacion["embeddings"], use_container_width=True)
    st.caption(
        "Diferencia esperada: TF-IDF encuentra coincidencias LITERALES de palabras; los "
        "embeddings pueden encontrar comentarios semánticamente relacionados aunque usen "
        "palabras distintas (por ejemplo 'lentes' vs. 'protección ocular')."
    )
