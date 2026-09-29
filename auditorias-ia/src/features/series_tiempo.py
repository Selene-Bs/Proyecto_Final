"""
Bloque D — Series de tiempo, y Bloque E — Feature Engineering.

Trabaja sobre el score de cumplimiento agregado por día (serie temporal
del "pulso" de la planta) para: tendencia, estacionalidad, ventanas
móviles, rezagos y detección simple de cambios de régimen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def serie_diaria(df: pd.DataFrame, columna: str = "score_cumplimiento") -> pd.DataFrame:
    """Agrega el dataset de auditorías a una serie diaria (una fila por día)."""
    serie = (
        df.groupby("fecha", observed=True)
        .agg(
            score_promedio=(columna, "mean"),
            n_auditorias=("audit_id", "count"),
            n_desviaciones_criticas=("tiene_no_ok_sin_accion", "sum"),
        )
        .reset_index()
        .sort_values("fecha")
    )
    serie = serie.set_index("fecha").asfreq("D")
    serie["n_auditorias"] = serie["n_auditorias"].fillna(0)
    serie["score_promedio"] = serie["score_promedio"].interpolate(limit_direction="both")
    return serie.reset_index()


def agregar_features_temporales(serie: pd.DataFrame, columna: str = "score_promedio") -> pd.DataFrame:
    """Feature engineering temporal: rolling, lags, tendencia, día de semana."""
    df = serie.copy().sort_values("fecha").reset_index(drop=True)

    # Tendencia (media móvil larga) y estacionalidad semanal (media móvil corta)
    df["media_movil_7d"] = df[columna].rolling(7, min_periods=1).mean()
    df["media_movil_30d"] = df[columna].rolling(30, min_periods=1).mean()
    df["std_movil_7d"] = df[columna].rolling(7, min_periods=2).std()

    # Rezagos (lags), típicos para modelado de series
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df[columna].shift(lag)

    # Componente estacional simple: promedio histórico por día de la semana
    df["dia_semana"] = pd.to_datetime(df["fecha"]).dt.day_name()
    estacional = df.groupby("dia_semana")[columna].transform("mean")
    df["componente_estacional_dow"] = estacional

    # Tendencia lineal simple (regresión sobre el índice temporal)
    x = np.arange(len(df))
    if len(df) > 2:
        pendiente, intercepto = np.polyfit(x, df[columna].ffill().bfill(), 1)
    else:
        pendiente, intercepto = 0.0, df[columna].mean()
    df["tendencia_lineal"] = intercepto + pendiente * x
    df.attrs["pendiente_tendencia"] = float(pendiente)

    return df


def detectar_cambios_de_regimen(serie: pd.DataFrame, columna: str = "score_promedio", ventana: int = 14, z_umbral: float = 1.5) -> pd.DataFrame:
    """
    Detección simple de cambio de régimen: compara la media móvil reciente
    contra la media móvil histórica usando un z-score. No requiere librerías
    externas de detección de changepoints, mantiene el método explicable.
    """
    df = serie.copy()
    media_global = df[columna].mean()
    std_global = df[columna].std() or 1e-6
    df["media_movil_ventana"] = df[columna].rolling(ventana, min_periods=3).mean()
    df["z_score_regimen"] = (df["media_movil_ventana"] - media_global) / std_global
    df["cambio_de_regimen"] = df["z_score_regimen"].abs() > z_umbral
    return df


def resumen_series(df_auditorias: pd.DataFrame) -> dict:
    """Genera el resumen textual que se muestra en el módulo de Series de Tiempo."""
    serie = serie_diaria(df_auditorias)
    feat = agregar_features_temporales(serie)
    pendiente = feat.attrs.get("pendiente_tendencia", 0.0)
    tendencia_txt = "mejorando" if pendiente > 0 else "empeorando" if pendiente < 0 else "estable"

    peor_dow = feat.groupby("dia_semana")["score_promedio"].mean().idxmin()
    mejor_dow = feat.groupby("dia_semana")["score_promedio"].mean().idxmax()

    return {
        "pendiente_diaria": pendiente,
        "tendencia_texto": tendencia_txt,
        "peor_dia_semana": peor_dow,
        "mejor_dia_semana": mejor_dow,
        "serie": feat,
    }
