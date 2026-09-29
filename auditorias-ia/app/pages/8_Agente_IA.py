import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from common import inicializar_pagina, obtener_dataset, obtener_indice_rag
from src.agents.agente import Agente, TOOLS, evaluar_tool_selection_accuracy

inicializar_pagina("Agente IA (Tool Calling)", "🕹️")
df = obtener_dataset()

st.markdown(
    "El agente decide, a partir de una instrucción en lenguaje natural, qué **herramienta** "
    "usar (o encadenar varias, tarea *multi-step*). Cada llamada queda registrada en un log."
)

with st.expander("🧰 Herramientas disponibles del agente"):
    for nombre, desc in TOOLS.items():
        st.markdown(f"- **{nombre}**: {desc}")

if "agente" not in st.session_state:
    st.session_state["agente"] = Agente(
        df_auditorias=df,
        pipeline_riesgo=st.session_state.get("pipeline_riesgo"),
        indice_rag=obtener_indice_rag(),
    )
agente = st.session_state["agente"]
agente.pipeline_riesgo = st.session_state.get("pipeline_riesgo", agente.pipeline_riesgo)

if agente.pipeline_riesgo is None:
    st.warning("Visita primero la página **🤖 Machine Learning** para entrenar el modelo de riesgo que usa la tool `predictor_riesgo`.")

st.header("Habla con el agente")
ejemplos = [
    "¿Cuánto es 12*3 + 7?",
    "¿Cuál es la probabilidad de riesgo en la Línea 2 turno nocturno?",
    "¿El turno nocturno tiene un desempeño significativamente distinto?",
    "¿Qué EPP debo usar en el área de ensamble?",
    "¿La tendencia del cumplimiento está mejorando o empeorando?",
    "Dame un reporte ejecutivo de la semana",
]
instruccion = st.selectbox("Elige un ejemplo o escribe abajo:", ["(escribir la mía)"] + ejemplos)
if instruccion == "(escribir la mía)":
    instruccion = st.text_input("Tu instrucción:", "")

if st.button("Enviar al agente") and instruccion:
    resultado = agente.responder(instruccion)
    st.success(f"Tool seleccionada por el router: **{resultado.get('tool_seleccionada')}**")
    st.json(resultado, expanded=False)

    if resultado.get("tool_seleccionada") == "generador_reporte" and resultado.get("resumen_ejecutivo"):
        st.markdown("### 📝 Resumen ejecutivo (tarea multi-step)")
        st.write(resultado["resumen_ejecutivo"])
        with st.expander("Ver pasos ejecutados y evidencia usada"):
            st.write(resultado["pasos_ejecutados"])
            st.text(resultado["evidencia_usada"])

st.divider()
st.header("Log de ejecución del agente (auditable)")
if agente.log:
    st.dataframe(agente.log, use_container_width=True)
else:
    st.caption("Aún no se han ejecutado tools en esta sesión.")

st.divider()
st.header("Métrica obligatoria: Tool Selection Accuracy")
evaluacion = evaluar_tool_selection_accuracy(agente)
st.metric("Tool Selection Accuracy", f"{evaluacion['tool_selection_accuracy']:.1%}", help=f"sobre {evaluacion['n_casos']} casos de prueba")
st.dataframe(evaluacion["detalle"], use_container_width=True)

st.divider()
with st.expander("⚙️ Configurar LLM (Claude API)"):
    st.markdown(
        "Para que las tools `consulta_rag` y `generador_reporte` generen texto con el LLM, "
        "define la variable de entorno / *secret* `ANTHROPIC_API_KEY` al desplegar la app. "
        "Sin esa clave, el agente sigue funcionando (routing, cálculo, ML, estadística) pero "
        "la redacción final del LLM mostrará un mensaje de error controlado en vez de fallar."
    )
