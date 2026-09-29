"""
Bloque P — Wavelets y análisis multiresolución (PyWavelets).

Sobre la misma señal dimensional de alta frecuencia, se realiza una
Transformada Wavelet Discreta (DWT) multinivel para:

  - separar aproximación (tendencia) y detalles (variaciones rápidas),
  - calcular la energía por nivel (dónde se concentra la variabilidad),
  - reconstruir la señal (denoising) eliminando detalles de alta frecuencia,
  - localizar el momento (índice/hora) donde ocurre la ráfaga de anomalía
    inyectada (herramienta desgastada), comparando la energía local del
    detalle de nivel 1 contra su media histórica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


def descomponer_dwt(serie: pd.DataFrame, columna: str = "desviacion_dimensional_mm",
                     wavelet: str = "db4", niveles: int = 4) -> dict:
    señal = serie[columna].values.astype(float)
    niveles_max = pywt.dwt_max_level(len(señal), pywt.Wavelet(wavelet).dec_len)
    niveles_usados = min(niveles, niveles_max)

    coeficientes = pywt.wavedec(señal, wavelet=wavelet, level=niveles_usados)
    # coeficientes = [cA_n, cD_n, cD_n-1, ..., cD_1]
    aproximacion = coeficientes[0]
    detalles = coeficientes[1:]

    energia_total = sum(np.sum(c ** 2) for c in coeficientes)
    energia_por_nivel = {
        "aproximación (tendencia)": float(np.sum(aproximacion ** 2) / energia_total * 100),
    }
    for i, d in enumerate(detalles):
        nivel = niveles_usados - i
        energia_por_nivel[f"detalle nivel {nivel}"] = float(np.sum(d ** 2) / energia_total * 100)

    return {
        "wavelet": wavelet,
        "niveles_usados": niveles_usados,
        "coeficientes": coeficientes,
        "energia_por_nivel_pct": energia_por_nivel,
        "señal_original": señal,
    }


def reconstruir_denoised(descomposicion: dict, umbral_pct: float = 0.15) -> np.ndarray:
    """Elimina el detalle de más alta frecuencia (nivel 1) por umbralización
    suave (soft-thresholding), una técnica estándar de denoising con Wavelets."""
    coeficientes = [c.copy() for c in descomposicion["coeficientes"]]
    detalle_mas_fino = coeficientes[-1]
    umbral = umbral_pct * np.max(np.abs(detalle_mas_fino)) if len(detalle_mas_fino) else 0.0
    coeficientes[-1] = pywt.threshold(detalle_mas_fino, umbral, mode="soft")

    señal_denoised = pywt.waverec(coeficientes, wavelet=descomposicion["wavelet"])
    señal_original = descomposicion["señal_original"]
    return señal_denoised[: len(señal_original)]


def detectar_anomalias_wavelet(serie: pd.DataFrame, columna: str = "desviacion_dimensional_mm",
                                wavelet: str = "db4", z_umbral: float = 3.0) -> pd.DataFrame:
    """
    Detección de anomalías por energía local del detalle de nivel más fino:
    se calcula la Transformada Wavelet Continua-like vía ventana deslizante
    de energía del detalle DWT nivel 1, y se marca anomalía donde esa
    energía se dispara respecto a su propia media/desv. estándar (z-score).
    """
    señal = serie[columna].values.astype(float)
    coeficientes = pywt.wavedec(señal, wavelet=wavelet, level=1)
    detalle_1 = coeficientes[-1]

    # Upsample simple del detalle a la longitud de la señal original (nivel 1 -> factor ~2)
    factor = len(señal) / len(detalle_1)
    detalle_interp = np.interp(
        np.arange(len(señal)), np.arange(len(detalle_1)) * factor, detalle_1
    )
    energia_local = detalle_interp ** 2
    z = (energia_local - energia_local.mean()) / (energia_local.std() or 1e-9)

    resultado = serie.copy().reset_index(drop=True)
    resultado["energia_wavelet_detalle1"] = energia_local
    resultado["z_score_energia"] = z
    resultado["anomalia_detectada_wavelet"] = z > z_umbral
    return resultado
