import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime as dt
import pandas as pd
import streamlit as st
from common import inicializar_pagina, obtener_dataset
from src.config import LINEAS, TURNOS, AUDITORES, PREGUNTAS, RESPUESTAS, AUDITORIAS_CSV, PUNTAJE_RESPUESTA

inicializar_pagina("Registrar Auditoría de Proceso", "📝")

st.markdown(
    "Formulario para capturar una auditoría de piso. En este demo, los registros se "
    "guardan en la sesión y pueden exportarse a CSV; en producción se conectarían a una "
    "base de datos remota (ver sección de despliegue del README)."
)

df = obtener_dataset()
operadores_disponibles = sorted(df["operador_id"].unique().tolist())

with st.form("form_auditoria"):
    c1, c2, c3 = st.columns(3)
    linea = c1.selectbox("Línea", LINEAS)
    turno = c2.selectbox("Turno", TURNOS)
    auditor = c3.selectbox("Auditor", AUDITORES)

    c4, c5 = st.columns(2)
    operador_id = c4.selectbox("Operador", operadores_disponibles)
    fecha = c5.date_input("Fecha de la auditoría", dt.date.today())

    st.divider()
    st.markdown("**Cédula de auditoría**")
    respuestas = {}
    for p in PREGUNTAS:
        respuestas[p["id"]] = st.radio(f"{p['texto']}  \n*({p['categoria']})*", RESPUESTAS, horizontal=True, key=p["id"])

    medicion = st.number_input("Medición dimensional con vernier (mm)", value=25.00, step=0.01, format="%.3f")
    comentario = st.text_area("Comentario del auditor (opcional)")

    enviado = st.form_submit_button("Guardar auditoría")

if enviado:
    score = sum(PUNTAJE_RESPUESTA[v] for v in respuestas.values()) / len(respuestas)
    nueva_fila = {
        "fecha": fecha.isoformat(),
        "linea": linea,
        "turno": turno,
        "auditor": auditor,
        "operador_id": operador_id,
        "medicion_vernier_mm": medicion,
        "comentario_auditor": comentario or "Sin comentario registrado.",
        "score_cumplimiento": round(score, 3),
        **respuestas,
    }
    if "auditorias_nuevas" not in st.session_state:
        st.session_state["auditorias_nuevas"] = []
    st.session_state["auditorias_nuevas"].append(nueva_fila)
    st.success(f"Auditoría registrada. Score de cumplimiento: {score:.1%}")
    if score < 0.7:
        st.error("⚠️ Score bajo: se recomienda escalar esta auditoría a mejora continua.")

st.divider()
st.header("Auditorías registradas en esta sesión")
if st.session_state.get("auditorias_nuevas"):
    df_nuevas = pd.DataFrame(st.session_state["auditorias_nuevas"])
    st.dataframe(df_nuevas, use_container_width=True)
    st.download_button(
        "⬇️ Descargar como CSV",
        df_nuevas.to_csv(index=False).encode("utf-8"),
        file_name="auditorias_nuevas.csv",
        mime="text/csv",
    )
else:
    st.caption("Aún no se ha registrado ninguna auditoría en esta sesión.")
