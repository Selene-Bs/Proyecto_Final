import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import inicializar_pagina, obtener_dataset
from src.features.series_tiempo import serie_diaria, agregar_features_temporales, detectar_cambios_de_regimen

inicializar_pagina("Series de Tiempo y Feature Engineering", "⏱️")
df = obtener_dataset()

serie = serie_diaria(df)
feat = agregar_features_temporales(serie)
feat_regimen = detectar_cambios_de_regimen(feat)

pendiente = feat.attrs.get("pendiente_tendencia", 0.0)
tendencia_txt = "MEJORANDO 📈" if pendiente > 0 else "EMPEORANDO 📉" if pendiente < 0 else "ESTABLE ➡️"

c1, c2, c3 = st.columns(3)
c1.metric("Tendencia general", tendencia_txt, f"{pendiente:+.5f} score/día")
c2.metric("Peor día de la semana", feat.groupby("dia_semana")["score_promedio"].mean().idxmin())
c3.metric("Mejor día de la semana", feat.groupby("dia_semana")["score_promedio"].mean().idxmax())

st.header("Tendencia y estacionalidad")
fig = go.Figure()
fig.add_trace(go.Scatter(x=feat["fecha"], y=feat["score_promedio"], name="Score diario", opacity=0.4))
fig.add_trace(go.Scatter(x=feat["fecha"], y=feat["media_movil_7d"], name="Media móvil 7 días"))
fig.add_trace(go.Scatter(x=feat["fecha"], y=feat["media_movil_30d"], name="Media móvil 30 días"))
fig.add_trace(go.Scatter(x=feat["fecha"], y=feat["tendencia_lineal"], name="Tendencia lineal", line=dict(dash="dash")))
fig.update_layout(title="Score de cumplimiento diario: tendencia y medias móviles", yaxis_title="Score")
st.plotly_chart(fig, use_container_width=True)

st.header("Estacionalidad semanal (día de la semana)")
est = feat.groupby("dia_semana")["score_promedio"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
).reset_index()
fig2 = px.bar(est, x="dia_semana", y="score_promedio", title="Score promedio por día de la semana")
st.plotly_chart(fig2, use_container_width=True)

st.header("Detección de cambios de régimen")
st.caption("Método: z-score de la media móvil de 14 días contra la media histórica global. |z| > 1.5 se marca como cambio de régimen.")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=feat_regimen["fecha"], y=feat_regimen["z_score_regimen"], name="z-score régimen"))
fig3.add_hline(y=1.5, line_dash="dot", line_color="red")
fig3.add_hline(y=-1.5, line_dash="dot", line_color="red")
st.plotly_chart(fig3, use_container_width=True)
n_cambios = int(feat_regimen["cambio_de_regimen"].sum())
st.write(f"Se detectaron **{n_cambios}** días marcados como posible cambio de régimen en el periodo analizado.")

st.divider()
st.header("Feature Engineering temporal (Bloque E)")
st.markdown(
    "Variables construidas: medias móviles (7/30 días), desviación estándar móvil, "
    "**rezagos (lags 1, 7 y 14 días)** y componente estacional por día de semana. "
    "Estas mismas features son las que alimentarían un modelo de forecasting supervisado."
)
st.dataframe(
    feat[["fecha", "score_promedio", "media_movil_7d", "lag_1", "lag_7", "lag_14", "componente_estacional_dow"]].tail(15).round(4),
    use_container_width=True,
)
