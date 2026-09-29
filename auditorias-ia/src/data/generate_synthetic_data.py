"""
Generador de datos sintéticos para el Sistema Inteligente de Auditorías de Proceso.

Produce tres fuentes de datos que alimentan TODOS los bloques del proyecto:

1. data/operadores.csv
   Catálogo de operadores con su línea "home" y su matriz de polivalencia
   (líneas en las que están certificados). Es la fuente de verdad contra la
   que se evalúa la pregunta P4 (matriz de polivalencia).

2. data/auditorias.csv
   Tabla principal: una fila por auditoría de proceso realizada en piso,
   con la respuesta a las 8 preguntas de la cédula, un comentario libre del
   auditor cuando hay una desviación, y una medición dimensional (vernier)
   ligada a la pregunta P2.
   Incluye tendencias, estacionalidad (fin de mes, turno nocturno) y un
   efecto "operador fuera de su matriz de polivalencia" para que el
   análisis posterior (EDA, estadística, ML) tenga señal real que
   recuperar, no ruido puro.

3. data/senales_dimensionales.csv
   Señal de proceso de alta frecuencia (24 lecturas/día) de desviación
   dimensional de pieza por línea, con un ciclo semanal inyectado y ráfagas
   de anomalía. Es la fuente para los bloques de Fourier (O) y Wavelets (P).

Ejecutar:
    python -m src.data.generate_synthetic_data
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (
    LINEAS,
    TURNOS,
    AUDITORES,
    PREGUNTA_IDS,
    N_OPERADORES,
    DATA_DIR,
    AUDITORIAS_CSV,
    OPERADORES_CSV,
    SENALES_CSV,
    DOCS_DIR,
)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Comentarios de auditor por tipo de desviación (para el módulo de NLP)
# ---------------------------------------------------------------------------
COMENTARIOS_POR_PREGUNTA = {
    "P1_EPP": [
        "Operador sin lentes de seguridad al momento de la auditoría.",
        "Falta guantes de protección, se corrigió en el momento.",
        "Tapones auditivos no colocados correctamente.",
        "Bata y calzado de seguridad correctos, pero sin cofia.",
    ],
    "P2_VERNIER": [
        "Operador no verifica cero del vernier antes de medir.",
        "Técnica de medición incorrecta, genera lecturas variables.",
        "Vernier fuera de calibración vigente, se retiró de línea.",
        "No registra la medición en el formato de control.",
    ],
    "P3_IT": [
        "Operador realiza secuencia distinta a la instrucción de trabajo vigente.",
        "IT desactualizada en el puesto, no coincide con el proceso actual.",
        "Omite paso de verificación intermedio marcado en la IT.",
        "No tiene la IT disponible en el puesto de trabajo.",
    ],
    "P4_POLIVALENCIA": [
        "Operador cubre la estación sin estar certificado en la matriz de polivalencia.",
        "Certificación vigente pero próxima a vencer, requiere recertificación.",
        "Operador de otra línea cubre ausentismo sin entrenamiento formal.",
    ],
    "P5_5S": [
        "Herramientas fuera de su lugar identificado.",
        "Acumulación de material en pasillo de evacuación.",
        "Contenedores de scrap sin identificar correctamente.",
    ],
    "P6_REGISTRO": [
        "Registro de calidad incompleto en el turno.",
        "Datos capturados con letra ilegible / con tachaduras.",
        "No se registró la hora de la medición.",
    ],
    "P7_TIEMPO_CICLO": [
        "Tiempo de ciclo por encima del estándar, genera cuello de botella.",
        "Operador se adelanta al tiempo estándar, riesgo de omitir pasos.",
    ],
    "P8_NO_CONFORME": [
        "Pieza no conforme detectada sin segregar en área roja.",
        "Etiqueta de no conformidad mal llenada.",
        "Producto sospechoso mezclado con producto bueno.",
    ],
}

COMENTARIOS_OK = [
    "Sin observaciones, cumple con el estándar.",
    "Operación conforme al estándar de la operación.",
    "Todo en orden durante la auditoría.",
]


def generar_operadores() -> pd.DataFrame:
    """Crea el catálogo de operadores y su matriz de polivalencia."""
    filas = []
    for i in range(1, N_OPERADORES + 1):
        operador_id = f"OP{i:03d}"
        linea_home = RNG.choice(LINEAS)
        # Cada operador domina 1 a 3 líneas (matriz de polivalencia real)
        n_poli = RNG.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        otras = [l for l in LINEAS if l != linea_home]
        extra = list(RNG.choice(otras, size=min(n_poli - 1, len(otras)), replace=False)) if n_poli > 1 else []
        matriz = sorted(set([linea_home] + extra))
        antiguedad_meses = int(RNG.integers(1, 96))
        # Operadores más nuevos tienden a tener más desviaciones (usado como
        # variable oculta consistente para que ML pueda encontrar señal real)
        propension_riesgo = float(np.clip(RNG.normal(loc=0.5 - antiguedad_meses / 200, scale=0.12), 0.02, 0.9))
        filas.append(
            {
                "operador_id": operador_id,
                "linea_home": linea_home,
                "matriz_polivalencia": ";".join(matriz),
                "antiguedad_meses": antiguedad_meses,
                "propension_riesgo": round(propension_riesgo, 3),
            }
        )
    return pd.DataFrame(filas)


def _elige_respuesta(prob_ok: float, prob_con_accion_dado_no_ok: float = 0.6) -> str:
    """Elige OK / No OK con acción / No OK sin acción dada una probabilidad de OK."""
    if RNG.random() < prob_ok:
        return "OK"
    return "No OK con acción" if RNG.random() < prob_con_accion_dado_no_ok else "No OK sin acción"


def generar_auditorias(df_operadores: pd.DataFrame, n_dias: int = 400, fecha_inicio: str = "2025-06-01") -> pd.DataFrame:
    """Genera la tabla principal de auditorías con estacionalidad y señal realista."""
    fechas = pd.date_range(fecha_inicio, periods=n_dias, freq="D")
    filas = []
    audit_id = 1

    operadores_por_linea = {
        linea: df_operadores[df_operadores["matriz_polivalencia"].str.contains(linea)]["operador_id"].tolist()
        for linea in LINEAS
    }

    for fecha in fechas:
        # No se audita todas las líneas/turnos todos los días (realista)
        n_auditorias_dia = RNG.integers(2, 6)
        dia_del_mes = fecha.day
        fin_de_mes = dia_del_mes >= 25  # más presión de producción -> más desviaciones
        dias_desde_inicio = (fecha - fechas[0]).days
        tendencia_mejora = dias_desde_inicio / n_dias  # 0 -> 1, la planta va mejorando (capacitación)

        for _ in range(n_auditorias_dia):
            linea = RNG.choice(LINEAS)
            turno = RNG.choice(TURNOS, p=[0.4, 0.35, 0.25])
            auditor = RNG.choice(AUDITORES)

            candidatos = operadores_por_linea[linea] or df_operadores["operador_id"].tolist()
            # 12% de las veces se asigna a un operador que NO pertenece a la matriz de esa línea
            fuera_de_matriz = RNG.random() < 0.12
            if fuera_de_matriz:
                otros = df_operadores[~df_operadores["operador_id"].isin(candidatos)]["operador_id"].tolist()
                operador_id = RNG.choice(otros) if otros else RNG.choice(candidatos)
            else:
                operador_id = RNG.choice(candidatos)

            fila_operador = df_operadores.loc[df_operadores["operador_id"] == operador_id].iloc[0]
            propension = fila_operador["propension_riesgo"]

            # Probabilidad base de cumplimiento por pregunta, modulada por
            # turno nocturno, fin de mes, tendencia de mejora y riesgo del operador
            ajuste = 0.0
            ajuste -= 0.10 if turno == "Nocturno" else 0.0
            ajuste -= 0.08 if fin_de_mes else 0.0
            ajuste += 0.10 * tendencia_mejora
            ajuste -= 0.15 * propension

            respuestas = {}
            comentarios = []
            for p in PREGUNTA_IDS:
                base_ok = {
                    "P1_EPP": 0.90,
                    "P2_VERNIER": 0.85,
                    "P3_IT": 0.86,
                    "P4_POLIVALENCIA": 0.95,
                    "P5_5S": 0.88,
                    "P6_REGISTRO": 0.87,
                    "P7_TIEMPO_CICLO": 0.84,
                    "P8_NO_CONFORME": 0.90,
                }[p]

                prob_ok = base_ok + ajuste
                if p == "P4_POLIVALENCIA" and fuera_de_matriz:
                    prob_ok = 0.15  # casi seguro marca desviación si está fuera de matriz
                prob_ok = float(np.clip(prob_ok, 0.03, 0.99))

                resp = _elige_respuesta(prob_ok)
                respuestas[p] = resp
                if resp != "OK":
                    comentarios.append(RNG.choice(COMENTARIOS_POR_PREGUNTA[p]))

            comentario_final = " ".join(comentarios) if comentarios else RNG.choice(COMENTARIOS_OK)

            # Medición dimensional ligada a P2 (vernier): si la técnica de
            # medición es incorrecta, la medición se aleja más del nominal.
            nominal_mm = 25.00
            ruido_base = 0.02 if respuestas["P2_VERNIER"] == "OK" else 0.06
            medicion_mm = round(float(RNG.normal(nominal_mm, ruido_base)), 3)

            score = float(np.mean([
                {"OK": 1.0, "No OK con acción": 0.5, "No OK sin acción": 0.0}[respuestas[p]]
                for p in PREGUNTA_IDS
            ]))

            fila = {
                "audit_id": audit_id,
                "fecha": fecha.date().isoformat(),
                "linea": linea,
                "turno": turno,
                "auditor": auditor,
                "operador_id": operador_id,
                "operador_fuera_de_matriz": fuera_de_matriz,
                "medicion_vernier_mm": medicion_mm,
                "nominal_mm": nominal_mm,
                "comentario_auditor": comentario_final,
                "score_cumplimiento": round(score, 3),
            }
            fila.update(respuestas)
            filas.append(fila)
            audit_id += 1

    df = pd.DataFrame(filas)
    df = df.sort_values(["fecha", "linea", "turno"]).reset_index(drop=True)
    return df


def generar_senales_dimensionales(n_dias: int = 120, fecha_inicio: str = "2025-10-01") -> pd.DataFrame:
    """
    Señal de alta frecuencia (24 muestras/día) de desviación dimensional de
    pieza por línea, pensada para el análisis de Fourier y Wavelets.

    Se inyecta:
      - un ciclo semanal (la desviación crece hacia el viernes por fatiga/ritmo
        de producción y baja el fin de semana),
      - un ciclo diario suave (arranque de turno),
      - ráfagas de anomalía (herramienta desgastada) detectables por Wavelets,
      - ruido gaussiano.
    """
    filas = []
    horas = np.arange(24)
    dias = pd.date_range(fecha_inicio, periods=n_dias, freq="D")

    for linea in LINEAS[:2]:  # dos líneas instrumentadas con sensor dimensional
        anomalia_dias = set(RNG.choice(np.arange(n_dias), size=max(3, n_dias // 25), replace=False).tolist())
        for i, dia in enumerate(dias):
            dow = dia.dayofweek  # 0=lunes ... 6=domingo
            ciclo_semanal = 0.015 * np.sin(2 * np.pi * dow / 7 - np.pi / 2) + 0.015
            for h in horas:
                ciclo_diario = 0.008 * np.sin(2 * np.pi * (h - 6) / 24)
                ruido = RNG.normal(0, 0.006)
                anomalia = 0.0
                if i in anomalia_dias and 10 <= h <= 14:
                    anomalia = 0.05 * np.exp(-((h - 12) ** 2) / 2.0)  # ráfaga localizada
                desviacion_mm = ciclo_semanal + ciclo_diario + ruido + anomalia
                filas.append(
                    {
                        "linea": linea,
                        "fecha": dia.date().isoformat(),
                        "hora": int(h),
                        "timestamp": pd.Timestamp(dia) + pd.Timedelta(hours=int(h)),
                        "desviacion_dimensional_mm": round(float(desviacion_mm), 5),
                        "anomalia_inyectada": bool(anomalia > 0.005),
                    }
                )
    return pd.DataFrame(filas)


INSTRUCCIONES_TRABAJO = {
    "IT-EPP-001.txt": """INSTRUCCIÓN DE TRABAJO IT-EPP-001
Título: Uso correcto del Equipo de Protección Personal (EPP)

1. Alcance: Aplica a todo operador dentro del área de producción.
2. EPP obligatorio por estación: lentes de seguridad, tapones auditivos,
   guantes de nitrilo, calzado de seguridad y cofia en áreas de ensamble fino.
3. El operador debe inspeccionar su EPP al inicio de turno; cualquier EPP
   dañado debe reportarse al líder de línea antes de iniciar operación.
4. No está permitido operar maquinaria sin el EPP completo, sin excepción.
5. El auditor de proceso verifica el cumplimiento de EPP como parte de la
   cédula de auditoría de piso (pregunta P1).
""",
    "IT-VERNIER-002.txt": """INSTRUCCIÓN DE TRABAJO IT-VERNIER-002
Título: Medición dimensional de pieza con vernier

1. Verificar que el vernier tenga calibración vigente (etiqueta verde).
2. Antes de medir, verificar el cero del instrumento sobre superficie limpia.
3. Sostener la pieza firmemente y realizar la medición perpendicular a la
   superficie de referencia, sin forzar las quijadas del vernier.
4. Registrar la medición en el formato de control dimensional de la
   estación, incluyendo hora y turno.
5. Si la medición está fuera de tolerancia (nominal 25.00 mm ± 0.10 mm),
   segregar la pieza como no conforme y notificar al supervisor de calidad.
6. La pregunta P2 de la auditoría de proceso valida el cumplimiento de esta IT.
""",
    "IT-PROCESO-003.txt": """INSTRUCCIÓN DE TRABAJO IT-PROCESO-003
Título: Secuencia estándar de operación

1. Cada estación cuenta con una instrucción de trabajo visible con la
   secuencia de pasos, tiempos de ciclo estándar y puntos de verificación.
2. El operador debe seguir la secuencia documentada; cualquier cambio de
   método debe pasar por control de cambios de ingeniería de proceso.
3. El tiempo de ciclo estándar no debe excederse por más de 10% de forma
   sostenida; desviaciones repetidas se escalan a mejora continua.
4. Los puntos de verificación de calidad marcados en la IT (checkpoints)
   no pueden omitirse bajo ninguna circunstancia.
""",
    "IT-POLIVALENCIA-004.txt": """INSTRUCCIÓN DE TRABAJO IT-POLIVALENCIA-004
Título: Matriz de polivalencia y asignación de personal

1. La matriz de polivalencia define en qué líneas/estaciones está
   certificado cada operador, con fecha de certificación y vigencia.
2. Un operador solo puede cubrir una estación si aparece certificado en la
   matriz vigente para esa línea.
3. En caso de ausentismo, Recursos Humanos y el líder de línea deben
   verificar la matriz antes de reasignar personal.
4. Cubrir una estación sin certificación vigente se considera una
   desviación crítica de proceso (riesgo alto) y debe registrarse como
   "No OK sin acción" si no se corrige de inmediato con un operador
   certificado o un entrenamiento supervisado in situ.
""",
    "IT-5S-005.txt": """INSTRUCCIÓN DE TRABAJO IT-5S-005
Título: Orden, limpieza e identificación de estación (5S)

1. Clasificar: solo herramientas y materiales necesarios en la estación.
2. Ordenar: cada herramienta tiene un lugar identificado (sombra/etiqueta).
3. Limpiar: la estación se limpia al inicio y fin de cada turno.
4. Estandarizar: los contenedores de producto bueno, scrap y no conforme
   deben estar rotulados e identificados visualmente en todo momento.
5. Disciplina: el cumplimiento de 5S se verifica en cada auditoría de
   proceso (pregunta P5) y forma parte del indicador de piso limpio.
""",
    "IT-NOCONFORME-006.txt": """INSTRUCCIÓN DE TRABAJO IT-NOCONFORME-006
Título: Manejo de producto no conforme

1. Toda pieza que no cumpla especificación dimensional o visual debe
   segregarse de inmediato en el área identificada como "Producto No
   Conforme" (contenedor rojo).
2. Debe llenarse la etiqueta de no conformidad con: fecha, línea, turno,
   operador, defecto encontrado y cantidad.
3. Está prohibido mezclar producto no conforme con producto bueno, aún de
   forma temporal.
4. El supervisor de calidad decide la disposición final (retrabajo,
   scrap o uso bajo concesión).
""",
}


def guardar_documentos_it(base_dir: str = DOCS_DIR) -> None:
    os.makedirs(base_dir, exist_ok=True)
    for nombre, contenido in INSTRUCCIONES_TRABAJO.items():
        with open(os.path.join(base_dir, nombre), "w", encoding="utf-8") as f:
            f.write(contenido.strip() + "\n")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generando catálogo de operadores...")
    df_operadores = generar_operadores()
    df_operadores.to_csv(OPERADORES_CSV, index=False)
    print(f"  -> {OPERADORES_CSV} ({len(df_operadores)} operadores)")

    print("Generando auditorías de proceso...")
    df_auditorias = generar_auditorias(df_operadores)
    df_auditorias.to_csv(AUDITORIAS_CSV, index=False)
    print(f"  -> {AUDITORIAS_CSV} ({len(df_auditorias)} auditorías)")

    print("Generando señales dimensionales (Fourier/Wavelets)...")
    df_senales = generar_senales_dimensionales()
    df_senales.to_csv(SENALES_CSV, index=False)
    print(f"  -> {SENALES_CSV} ({len(df_senales)} lecturas)")

    print("Generando instrucciones de trabajo (RAG)...")
    guardar_documentos_it()
    print(f"  -> {DOCS_DIR}/ ({len(INSTRUCCIONES_TRABAJO)} documentos)")

    print("\nListo. Dataset sintético generado con éxito.")


if __name__ == "__main__":
    main()
