"""
Bloque C — Estadística.

Responde preguntas de negocio concretas, no ejecuta pruebas "porque sí":

  P1: ¿El turno Nocturno tiene un score de cumplimiento significativamente
      distinto al resto de los turnos?  -> prueba de hipótesis (t de Welch)
  P2: ¿Existe correlación entre la antigüedad del operador y su score de
      cumplimiento?                     -> correlación de Pearson/Spearman
  P3: ¿Cuál es el intervalo de confianza al 95% del score de cumplimiento
      promedio por línea?                -> IC bootstrap / t-Student
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def descriptivos_por_grupo(df: pd.DataFrame, columna: str, grupo: str) -> pd.DataFrame:
    """Media, mediana, desviación estándar y varianza de `columna` por `grupo`."""
    agg = df.groupby(grupo, observed=True)[columna].agg(
        n="count", media="mean", mediana="median", desv_std="std", varianza="var"
    ).round(4)
    return agg.reset_index().sort_values("media")


def intervalo_confianza_media(muestra: np.ndarray, confianza: float = 0.95) -> tuple[float, float, float]:
    """IC para la media usando distribución t de Student. Devuelve (media, lim_inf, lim_sup)."""
    muestra = np.asarray(muestra, dtype=float)
    n = len(muestra)
    media = muestra.mean()
    if n < 2:
        return media, media, media
    sem = stats.sem(muestra)
    margen = sem * stats.t.ppf((1 + confianza) / 2, n - 1)
    return media, media - margen, media + margen


def ic_por_linea(df: pd.DataFrame, columna: str = "score_cumplimiento", confianza: float = 0.95) -> pd.DataFrame:
    filas = []
    for linea, grupo in df.groupby("linea", observed=True):
        media, lo, hi = intervalo_confianza_media(grupo[columna].values, confianza)
        filas.append({"linea": linea, "n": len(grupo), "media": media, "ic_inf": lo, "ic_sup": hi})
    return pd.DataFrame(filas).sort_values("media")


def prueba_hipotesis_turno_nocturno(df: pd.DataFrame, columna: str = "score_cumplimiento") -> dict:
    """
    H0: el score de cumplimiento del turno Nocturno tiene la misma media que
        el de los demás turnos.
    H1: la media del turno Nocturno es distinta (menor, en la hipótesis de
        negocio: menor supervisión durante la noche).

    Se usa la prueba t de Welch (no asume varianzas iguales) de dos colas,
    y se reporta también el resultado one-sided para la hipótesis dirigida.
    """
    nocturno = df.loc[df["turno"] == "Nocturno", columna].dropna().values
    resto = df.loc[df["turno"] != "Nocturno", columna].dropna().values

    t_stat, p_two_sided = stats.ttest_ind(nocturno, resto, equal_var=False)
    # one-sided: H1 = nocturno < resto
    p_one_sided = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

    alfa = 0.05
    resultado = {
        "hipotesis_nula": "La media de score del turno Nocturno es igual a la del resto de turnos.",
        "hipotesis_alterna": "La media de score del turno Nocturno es MENOR a la del resto de turnos.",
        "media_nocturno": float(np.mean(nocturno)),
        "media_resto": float(np.mean(resto)),
        "t_stat": float(t_stat),
        "p_valor_dos_colas": float(p_two_sided),
        "p_valor_una_cola": float(p_one_sided),
        "alfa": alfa,
        "rechaza_h0": bool(p_one_sided < alfa),
    }
    if resultado["rechaza_h0"]:
        resultado["interpretacion"] = (
            f"Con p={p_one_sided:.4f} < {alfa}, se rechaza H0: hay evidencia estadística de que "
            "el turno Nocturno tiene un score de cumplimiento promedio menor. Se recomienda "
            "reforzar supervisión/auditoría en ese turno."
        )
    else:
        resultado["interpretacion"] = (
            f"Con p={p_one_sided:.4f} >= {alfa}, no hay evidencia suficiente para afirmar que el "
            "turno Nocturno tenga un desempeño distinto al resto."
        )
    return resultado


def correlacion_antiguedad_score(df: pd.DataFrame) -> dict:
    """Correlación entre antigüedad del operador (meses) y score de cumplimiento."""
    sub = df[["antiguedad_meses", "score_cumplimiento"]].dropna()
    r_pearson, p_pearson = stats.pearsonr(sub["antiguedad_meses"], sub["score_cumplimiento"])
    r_spearman, p_spearman = stats.spearmanr(sub["antiguedad_meses"], sub["score_cumplimiento"])
    return {
        "pregunta": "¿La antigüedad del operador se relaciona con su score de cumplimiento?",
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "n": int(len(sub)),
        "interpretacion": (
            "Correlación positiva" if r_pearson > 0.1 else
            "Correlación negativa" if r_pearson < -0.1 else
            "Correlación débil / prácticamente nula"
        ) + f" (r={r_pearson:.3f}, p={p_pearson:.4f}).",
    }


def matriz_correlacion(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    columnas = columnas or ["score_cumplimiento", "antiguedad_meses", "n_desviaciones", "medicion_vernier_mm"]
    return df[columnas].corr(numeric_only=True).round(3)
