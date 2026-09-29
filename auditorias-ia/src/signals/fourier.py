"""
Bloque O — Fourier y análisis espectral.

Se aplica sobre la señal dimensional de alta frecuencia (24 lecturas/día)
generada en `senales_dimensionales.csv`. El objetivo de negocio es
detectar periodicidades (p. ej. un patrón semanal ligado al ritmo de
producción) y diferenciarlo de una tendencia o de ruido puro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def obtener_serie_linea(df_senales: pd.DataFrame, linea: str) -> pd.DataFrame:
    serie = df_senales[df_senales["linea"] == linea].sort_values("timestamp").reset_index(drop=True)
    return serie


def calcular_fft(serie: pd.DataFrame, columna: str = "desviacion_dimensional_mm", muestras_por_dia: int = 24) -> dict:
    señal = serie[columna].values
    señal = señal - señal.mean()  # remover componente DC para no dominar el espectro

    n = len(señal)
    fft_vals = np.fft.rfft(señal)
    fft_freqs = np.fft.rfftfreq(n, d=1.0)  # frecuencia en ciclos/muestra
    amplitudes = np.abs(fft_vals) / n

    # Convertir frecuencia (ciclos/muestra) a periodo en DÍAS
    with np.errstate(divide="ignore"):
        periodo_muestras = np.where(fft_freqs > 0, 1 / fft_freqs, np.inf)
    periodo_dias = periodo_muestras / muestras_por_dia

    espectro = pd.DataFrame({
        "frecuencia_ciclos_muestra": fft_freqs,
        "amplitud": amplitudes,
        "periodo_dias": periodo_dias,
    }).iloc[1:]  # se descarta la componente 0 (ya centrada)

    top = espectro.sort_values("amplitud", ascending=False).head(5)

    return {"espectro": espectro, "top_frecuencias": top, "señal_centrada": señal, "n_muestras": n}


def interpretar_top_frecuencias(top: pd.DataFrame) -> list[str]:
    interpretaciones = []
    for _, fila in top.iterrows():
        periodo = fila["periodo_dias"]
        if 6.5 <= periodo <= 7.5:
            texto = f"Periodo ≈ {periodo:.1f} días -> sugiere un ciclo SEMANAL (ritmo de producción lunes-viernes)."
        elif 0.9 <= periodo <= 1.1:
            texto = f"Periodo ≈ {periodo:.1f} días -> sugiere un ciclo DIARIO (arranque/cierre de turno)."
        else:
            texto = f"Periodo ≈ {periodo:.2f} días -> componente sin una interpretación operativa clara todavía."
        interpretaciones.append(texto)
    interpretaciones.append(
        "Nota: un pico espectral solo indica periodicidad, NO causalidad; se debe "
        "contrastar con el proceso real antes de tomar acciones."
    )
    return interpretaciones


def filtrar_señal_pasa_bajas(serie: pd.DataFrame, columna: str = "desviacion_dimensional_mm", frac_frecuencias: float = 0.05) -> np.ndarray:
    """Reconstrucción de la señal manteniendo solo las frecuencias dominantes (filtro pasa-bajas simple)."""
    señal = serie[columna].values
    media = señal.mean()
    señal_centrada = señal - media
    fft_vals = np.fft.rfft(señal_centrada)

    n_mantener = max(1, int(len(fft_vals) * frac_frecuencias))
    idx_dominantes = np.argsort(np.abs(fft_vals))[::-1][:n_mantener]
    fft_filtrada = np.zeros_like(fft_vals)
    fft_filtrada[idx_dominantes] = fft_vals[idx_dominantes]

    señal_reconstruida = np.fft.irfft(fft_filtrada, n=len(señal_centrada)) + media
    return señal_reconstruida
