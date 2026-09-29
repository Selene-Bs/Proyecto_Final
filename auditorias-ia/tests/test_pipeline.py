"""
Pruebas mínimas de humo (smoke tests) para garantizar reproducibilidad:
si esto pasa, el dataset y los módulos principales del pipeline funcionan
igual en cualquier máquina / en el entorno de despliegue.

Ejecutar:
    pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline_datos import cargar_dataset_final, cargar_senales
from src.config import PREGUNTA_IDS, RESPUESTAS


def test_dataset_no_vacio():
    df = cargar_dataset_final()
    assert len(df) > 0
    assert set(df["operador_id"].unique()) <= set(f"OP{i:03d}" for i in range(1, 41))


def test_respuestas_validas():
    df = cargar_dataset_final()
    for p in PREGUNTA_IDS:
        assert set(df[p].unique()) <= set(RESPUESTAS)


def test_score_en_rango():
    df = cargar_dataset_final()
    assert df["score_cumplimiento"].between(0, 1).all()


def test_senales_tienen_las_dos_lineas_instrumentadas():
    senales = cargar_senales()
    assert senales["linea"].nunique() == 2
    assert senales["desviacion_dimensional_mm"].notna().all()


def test_estadistica_prueba_hipotesis_corre_sin_error():
    from src.features.estadistica import prueba_hipotesis_turno_nocturno
    df = cargar_dataset_final()
    resultado = prueba_hipotesis_turno_nocturno(df)
    assert "p_valor_una_cola" in resultado


def test_agente_router_determinista():
    from src.agents.agente import Agente
    df = cargar_dataset_final()
    agente = Agente(df_auditorias=df)
    assert agente.enrutar("¿Cuánto es 2+2?") == "calculadora"
    assert agente.enrutar("¿Qué EPP debo usar?") == "consulta_rag"
