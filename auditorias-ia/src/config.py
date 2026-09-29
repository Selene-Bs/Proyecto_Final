"""
Configuración central del proyecto: catálogos de negocio.
Todo el sistema (generador de datos, EDA, ML, agente, RAG) importa desde aquí
para que las categorías sean consistentes en toda la aplicación.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Escala de respuesta de auditoría (variable ordinal de 3 niveles)
# ---------------------------------------------------------------------------
RESPUESTAS = ["OK", "No OK con acción", "No OK sin acción"]

# Puntaje numérico asociado a cada respuesta (para score de cumplimiento)
PUNTAJE_RESPUESTA = {
    "OK": 1.0,
    "No OK con acción": 0.5,
    "No OK sin acción": 0.0,
}

# Severidad para priorización del agente / reportes
SEVERIDAD_RESPUESTA = {
    "OK": "sin riesgo",
    "No OK con acción": "riesgo medio",
    "No OK sin acción": "riesgo alto",
}

# ---------------------------------------------------------------------------
# Preguntas de la cédula de auditoría de proceso
# ---------------------------------------------------------------------------
PREGUNTAS = [
    {
        "id": "P1_EPP",
        "texto": "¿El operador trae puesto su EPP completo?",
        "categoria": "Seguridad",
    },
    {
        "id": "P2_VERNIER",
        "texto": "¿El operador utiliza el vernier para medir la pieza correctamente?",
        "categoria": "Calidad",
    },
    {
        "id": "P3_IT",
        "texto": "¿El operador sigue su instrucción de trabajo (IT)?",
        "categoria": "Proceso",
    },
    {
        "id": "P4_POLIVALENCIA",
        "texto": "¿El operador se encuentra dentro de su matriz de polivalencia?",
        "categoria": "Personal",
    },
    {
        "id": "P5_5S",
        "texto": "¿El área de trabajo está limpia, ordenada e identificada (5S)?",
        "categoria": "Proceso",
    },
    {
        "id": "P6_REGISTRO",
        "texto": "¿El operador registra correctamente los datos de calidad en el formato?",
        "categoria": "Calidad",
    },
    {
        "id": "P7_TIEMPO_CICLO",
        "texto": "¿El operador respeta el tiempo de ciclo estándar de la operación?",
        "categoria": "Proceso",
    },
    {
        "id": "P8_NO_CONFORME",
        "texto": "¿El operador identifica y segrega correctamente el producto no conforme?",
        "categoria": "Calidad",
    },
]

PREGUNTA_IDS = [p["id"] for p in PREGUNTAS]
PREGUNTA_TEXTO = {p["id"]: p["texto"] for p in PREGUNTAS}

# ---------------------------------------------------------------------------
# Catálogos operativos de planta
# ---------------------------------------------------------------------------
LINEAS = ["Línea 1", "Línea 2", "Línea 3", "Línea 4"]
TURNOS = ["Matutino", "Vespertino", "Nocturno"]
AUDITORES = ["J. Ramírez", "A. Torres", "M. Hernández", "L. Cruz"]

N_OPERADORES = 40

# Rutas de datos
DATA_DIR = "data"
AUDITORIAS_CSV = f"{DATA_DIR}/auditorias.csv"
OPERADORES_CSV = f"{DATA_DIR}/operadores.csv"
SENALES_CSV = f"{DATA_DIR}/senales_dimensionales.csv"
DOCS_DIR = "docs/instrucciones_trabajo"
