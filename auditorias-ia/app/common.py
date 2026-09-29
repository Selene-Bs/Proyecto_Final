"""
Utilidades compartidas por todas las páginas de la app Streamlit.
Centraliza el `sys.path` y el cacheo de datos/modelos para que cada
página sea rápida después de la primera carga.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from src.preprocessing.pipeline_datos import cargar_dataset_final, cargar_senales  # noqa: E402


@st.cache_data(show_spinner="Cargando y limpiando el dataset de auditorías...")
def obtener_dataset():
    return cargar_dataset_final()


@st.cache_data(show_spinner="Cargando señales dimensionales de alta frecuencia...")
def obtener_senales():
    return cargar_senales()


@st.cache_resource(show_spinner="Entrenando modelos de Machine Learning (una sola vez por sesión)...")
def obtener_modelos_ml(_df):
    from src.ml.modelos import entrenar_y_comparar_modelos
    return entrenar_y_comparar_modelos(_df)


@st.cache_resource(show_spinner="Entrenando red neuronal (MLP)...")
def obtener_modelo_dl(_df):
    from src.deep_learning.red_neuronal import entrenar_mlp_vs_clasico
    return entrenar_mlp_vs_clasico(_df)


@st.cache_resource(show_spinner="Entrenando clasificador de texto (NLP)...")
def obtener_modelo_nlp(_df):
    from src.nlp.nlp_clasico import entrenar_clasificador_texto
    return entrenar_clasificador_texto(_df)


@st.cache_resource(show_spinner="Construyendo índice RAG sobre Instrucciones de Trabajo...")
def obtener_indice_rag():
    from src.rag.rag_instrucciones import construir_indice
    return construir_indice()


def inicializar_pagina(titulo: str, icono: str = "🏭"):
    st.set_page_config(page_title=titulo, page_icon=icono, layout="wide")
    st.title(f"{icono} {titulo}")
