"""
Pipeline de Datos (Bloque A).

Responsable de: cargar las fuentes crudas, inspeccionar tipos, tratar
duplicados/valores atípicos y producir el "dataset final" que consumen
EDA, Estadística, Series de Tiempo, Feature Engineering y ML.

Toda la app llama a `cargar_dataset_final()` en vez de leer CSVs sueltos,
para que exista un único punto de verdad del pipeline (trazabilidad).
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

from src.config import (
    AUDITORIAS_CSV,
    OPERADORES_CSV,
    SENALES_CSV,
    PREGUNTA_IDS,
    PUNTAJE_RESPUESTA,
)


def _generar_si_no_existe():
    """Si no existen los CSV, dispara el generador sintético (primer arranque)."""
    if not (os.path.exists(AUDITORIAS_CSV) and os.path.exists(OPERADORES_CSV) and os.path.exists(SENALES_CSV)):
        from src.data.generate_synthetic_data import main as generar
        generar()


def cargar_crudos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga desde CSV (fuente de datos declarada) los tres data sources crudos."""
    _generar_si_no_existe()
    auditorias = pd.read_csv(AUDITORIAS_CSV)
    operadores = pd.read_csv(OPERADORES_CSV)
    senales = pd.read_csv(SENALES_CSV, parse_dates=["timestamp"])
    return auditorias, operadores, senales


def limpiar_auditorias(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos, duplicados, valores faltantes y atípicos sobre la tabla de auditorías."""
    df = df.copy()

    # --- Tipos ---
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for col in ["linea", "turno", "auditor", "operador_id"]:
        df[col] = df[col].astype("category")
    df["operador_fuera_de_matriz"] = df["operador_fuera_de_matriz"].astype(bool)

    # --- Duplicados ---
    antes = len(df)
    df = df.drop_duplicates(subset=["audit_id"])
    duplicados_removidos = antes - len(df)

    # --- Valores faltantes ---
    # Comentario vacío -> texto neutro explícito (no se descarta la fila)
    df["comentario_auditor"] = df["comentario_auditor"].fillna("Sin comentario registrado.")
    # Si faltara alguna respuesta puntual, se imputa como "No OK sin acción"
    # (regla de negocio conservadora: dato faltante en auditoría = riesgo, no se asume OK)
    filas_incompletas = df[PREGUNTA_IDS].isna().any(axis=1).sum()
    for p in PREGUNTA_IDS:
        df[p] = df[p].fillna("No OK sin acción")

    # --- Valores atípicos: medición dimensional ---
    # Tolerancia especificada en IT-VERNIER-002: 25.00 mm +/- 0.10 mm
    tol = 0.10
    atipicos = ((df["medicion_vernier_mm"] - df["nominal_mm"]).abs() > 3 * tol).sum()
    # No se eliminan (son evidencia real de falla de medición), se marcan
    df["medicion_atipica"] = (df["medicion_vernier_mm"] - df["nominal_mm"]).abs() > 3 * tol

    df.attrs["log_limpieza"] = {
        "duplicados_removidos": int(duplicados_removidos),
        "filas_con_respuesta_faltante_imputada": int(filas_incompletas),
        "mediciones_atipicas_marcadas": int(atipicos),
    }
    return df


def enriquecer_con_operadores(df_auditorias: pd.DataFrame, df_operadores: pd.DataFrame) -> pd.DataFrame:
    """Combina auditorías + catálogo de operadores (join, Bloque A)."""
    df_operadores = df_operadores.copy()
    df_operadores["operador_id"] = df_operadores["operador_id"].astype(str)
    df = df_auditorias.copy()
    df["operador_id"] = df["operador_id"].astype(str)
    df = df.merge(df_operadores, on="operador_id", how="left")
    return df


def agregar_score_y_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas base usadas en casi todos los módulos posteriores."""
    df = df.copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["dia_semana"] = df["fecha"].dt.day_name()
    df["semana"] = df["fecha"].dt.to_period("W").astype(str)
    df["fin_de_mes"] = df["fecha"].dt.day >= 25

    # Recalcular score de cumplimiento desde la tabla limpia (garantiza consistencia)
    df["score_cumplimiento"] = df[PREGUNTA_IDS].apply(
        lambda fila: np.mean([PUNTAJE_RESPUESTA[v] for v in fila]), axis=1
    )
    # Bandera de severidad máxima de la auditoría (para priorizar acciones)
    df["tiene_no_ok_sin_accion"] = (df[PREGUNTA_IDS] == "No OK sin acción").any(axis=1)
    df["n_desviaciones"] = (df[PREGUNTA_IDS] != "OK").sum(axis=1)
    return df


def cargar_dataset_final() -> pd.DataFrame:
    """Punto único de entrada: crudo -> limpio -> enriquecido -> features base."""
    auditorias, operadores, _senales = cargar_crudos()
    auditorias = limpiar_auditorias(auditorias)
    df = enriquecer_con_operadores(auditorias, operadores)
    df = agregar_score_y_derivadas(df)
    return df


def cargar_senales() -> pd.DataFrame:
    """Dataset de señal dimensional de alta frecuencia (Fourier/Wavelets)."""
    _generar_si_no_existe()
    df = pd.read_csv(SENALES_CSV, parse_dates=["timestamp", "fecha"])
    return df


if __name__ == "__main__":
    df = cargar_dataset_final()
    print(df.shape)
    print(df.attrs.get("log_limpieza"))
    print(df.head())
