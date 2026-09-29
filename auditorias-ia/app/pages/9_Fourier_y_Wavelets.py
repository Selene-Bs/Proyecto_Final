import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import inicializar_pagina, obtener_senales
from src.signals.fourier import obtener_serie_linea, calcular_fft, interpretar_top_frecuencias, filtrar_señal_pasa_bajas
from src.signals.wavelets import descomponer_dwt, reconstruir_denoised, detectar_anomalias_wavelet

inicializar_pagina("Fourier y Wavelets", "🌊")
senales = obtener_senales()

st.markdown(
    "Fuente: señal dimensional de alta frecuencia (24 lecturas/día) capturada en 2 líneas "
    "instrumentadas, ligada a la medición con vernier (pregunta P2 de la cédula de auditoría)."
)

linea = st.selectbox("Línea instrumentada", sorted(senales["linea"].unique()))
serie = obtener_serie_linea(senales, linea)

st.header("Señal en el dominio del tiempo")
fig0 = px.line(serie, x="timestamp", y="desviacion_dimensional_mm", title=f"Desviación dimensional — {linea}")
st.plotly_chart(fig0, use_container_width=True)

st.divider()
st.header("Fourier — Análisis espectral (Bloque O)")
fft = calcular_fft(serie)
fig_fft = px.line(fft["espectro"], x="periodo_dias", y="amplitud", title="Espectro de amplitud vs. periodo (días)",
                   log_x=True)
fig_fft.update_xaxes(title="Periodo (días, escala log)")
st.plotly_chart(fig_fft, use_container_width=True)

st.markdown("**Frecuencias/periodos dominantes e interpretación:**")
interpretaciones = interpretar_top_frecuencias(fft["top_frecuencias"])
for linea_txt in interpretaciones:
    st.write("• " + linea_txt)
st.dataframe(fft["top_frecuencias"].round(4), use_container_width=True)

st.subheader("Filtrado / reconstrucción (pasa-bajas con las frecuencias dominantes)")
frac = st.slider("Porcentaje de frecuencias dominantes a conservar", 1, 30, 5) / 100
reconstruida = filtrar_señal_pasa_bajas(serie, frac_frecuencias=frac)
fig_filt = go.Figure()
fig_filt.add_trace(go.Scatter(x=serie["timestamp"], y=serie["desviacion_dimensional_mm"], name="Original", opacity=0.4))
fig_filt.add_trace(go.Scatter(x=serie["timestamp"], y=reconstruida, name="Reconstruida (filtrada)"))
st.plotly_chart(fig_filt, use_container_width=True)

st.divider()
st.header("Wavelets — Análisis multiresolución (Bloque P)")
wavelet = st.selectbox("Familia de wavelet", ["db4", "db2", "sym4", "coif2"])
dwt = descomponer_dwt(serie, wavelet=wavelet)
st.caption(f"Niveles de descomposición usados: {dwt['niveles_usados']}")

fig_energia = px.bar(x=list(dwt["energia_por_nivel_pct"].keys()), y=list(dwt["energia_por_nivel_pct"].values()),
                      labels={"x": "Componente", "y": "% de energía"}, title="Energía por nivel de descomposición")
st.plotly_chart(fig_energia, use_container_width=True)
st.caption(
    "La mayor parte de la energía suele concentrarse en la 'aproximación' (tendencia lenta); "
    "picos de energía en los detalles de nivel fino son evidencia de eventos rápidos "
    "(posible desgaste de herramienta o error de medición puntual)."
)

st.subheader("Denoising por umbralización Wavelet")
denoised = reconstruir_denoised(dwt)
fig_dn = go.Figure()
fig_dn.add_trace(go.Scatter(x=serie["timestamp"], y=serie["desviacion_dimensional_mm"], name="Original", opacity=0.35))
fig_dn.add_trace(go.Scatter(x=serie["timestamp"], y=denoised, name="Señal denoised (Wavelet)"))
st.plotly_chart(fig_dn, use_container_width=True)

st.subheader("Detección de anomalías por energía Wavelet")
anom = detectar_anomalias_wavelet(serie)
n_anom = int(anom["anomalia_detectada_wavelet"].sum())
st.metric("Anomalías detectadas", n_anom)
fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(x=anom["timestamp"], y=anom["desviacion_dimensional_mm"], name="Señal", opacity=0.5))
puntos_anom = anom[anom["anomalia_detectada_wavelet"]]
fig_anom.add_trace(go.Scatter(x=puntos_anom["timestamp"], y=puntos_anom["desviacion_dimensional_mm"],
                               mode="markers", name="Anomalía detectada", marker=dict(color="red", size=8)))
st.plotly_chart(fig_anom, use_container_width=True)
st.caption(
    "Localización de eventos: las anomalías detectadas coinciden con las ráfagas de "
    "desgaste de herramienta inyectadas en la simulación (validación del método)."
)
