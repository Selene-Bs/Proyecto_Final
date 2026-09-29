import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from common import inicializar_pagina, obtener_indice_rag
from src.rag.rag_instrucciones import responder_con_rag, evaluar_retriever, cargar_documentos

inicializar_pagina("RAG sobre Instrucciones de Trabajo", "📄")

st.markdown(
    "Base de conocimiento: las **Instrucciones de Trabajo (IT)** vigentes de la planta. "
    "Pipeline: documentos → chunking → embeddings → índice → Top-k → contexto → "
    "respuesta del LLM **citando la fuente**, con manejo explícito de 'sin evidencia'."
)

with st.expander("📚 Ver documentos base (Instrucciones de Trabajo)"):
    for doc in cargar_documentos():
        st.markdown(f"**{doc['archivo']}**")
        st.text(doc["texto"])
        st.divider()

indice = obtener_indice_rag()
st.caption(f"Método de embeddings activo: **{indice['metodo_embeddings']}** — {len(indice['chunks'])} fragmentos indexados.")

st.header("Pregunta al asistente de piso")
pregunta = st.text_input(
    "Escribe tu pregunta sobre procedimientos de planta:",
    "¿Qué debo hacer si un operador cubre una línea sin estar certificado?",
)
k = st.slider("Top-k documentos a recuperar", 1, 5, 3)

if st.button("Preguntar"):
    salida = responder_con_rag(indice, pregunta, k=k)
    if not salida["llm_ok"]:
        st.warning(
            "No se pudo generar la respuesta con el LLM (probablemente falta configurar "
            "`ANTHROPIC_API_KEY` en los *secrets* de la app). Se muestra el contexto "
            "recuperado para que puedas verificarlo manualmente."
        )
    st.markdown("### Respuesta")
    st.write(salida["respuesta"])
    if salida["fuentes"]:
        st.info("**Fuentes citadas:** " + ", ".join(salida["fuentes"]))
    with st.expander("Ver contexto recuperado (auditable)"):
        st.text(salida["contexto_usado"] or "Sin contexto (similitud por debajo del umbral).")
    st.caption(f"Similitud máxima con la base documental: {salida['similitud_max']:.3f}")

st.divider()
st.header("Evaluación del Retriever (obligatoria del bloque)")
st.markdown("Conjunto de prueba fijo de 7 preguntas con su documento IT esperado.")
evaluacion = evaluar_retriever(indice, k=3)
c1, c2 = st.columns(2)
c1.metric(f"Recall@{evaluacion['k']}", evaluacion[f"recall_at_{evaluacion['k']}"])
c2.metric("MRR (Mean Reciprocal Rank)", evaluacion["mrr"])
st.dataframe(evaluacion["detalle"], use_container_width=True)
