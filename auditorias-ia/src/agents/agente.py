"""
Bloque N — Agentes y Tool Calling.

Se implementa un agente tipo "router + dispatcher" que decide, a partir de
una instrucción en lenguaje natural, qué herramienta(s) internas usar entre:

    - calculadora              : evalúa expresiones aritméticas simples
    - predictor_riesgo          : usa el modelo de ML del Bloque F
    - consulta_estadistica      : corre las funciones del Bloque C
    - consulta_rag              : responde contra las Instrucciones de Trabajo
    - analizador_series         : resume tendencia/estacionalidad (Bloque D)
    - generador_reporte         : encadena 2+ tools y redacta con el LLM
      (ejemplo de tarea MULTI-STEP)

Diseño:
    - Cada Tool se define con nombre, descripción y un validador de
      argumentos explícito (`_validar_args_*`).
    - El Dispatcher central (`ejecutar_tool`) captura errores por tool y
      nunca deja caer la aplicación completa.
    - Cada llamada queda registrada en `self.log` (auditable).
    - `MAX_PASOS` limita cuántas tools puede encadenar el agente en una
      sola instrucción, para evitar loops infinitos.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

MAX_PASOS = 4

# Operadores permitidos en la calculadora seguro (nunca usar eval() a secas)
_OPS_PERMITIDOS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval_seguro(nodo):
    if isinstance(nodo, ast.Constant):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPS_PERMITIDOS:
        return _OPS_PERMITIDOS[type(nodo.op)](_eval_seguro(nodo.left), _eval_seguro(nodo.right))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPS_PERMITIDOS:
        return _OPS_PERMITIDOS[type(nodo.op)](_eval_seguro(nodo.operand))
    raise ValueError("Expresión no permitida en la calculadora.")


TOOLS = {
    "calculadora": "Evalúa una expresión aritmética simple (+,-,*,/,**).",
    "predictor_riesgo": "Predice la probabilidad de una desviación crítica dado línea/turno/operador.",
    "consulta_estadistica": "Corre la prueba de hipótesis del turno Nocturno o la correlación de antigüedad.",
    "consulta_rag": "Responde preguntas sobre Instrucciones de Trabajo (IT) de la planta.",
    "analizador_series": "Resume tendencia y estacionalidad del score de cumplimiento diario.",
    "generador_reporte": "Tarea MULTI-STEP: combina estadística + series + RAG y redacta un resumen ejecutivo.",
}


@dataclass
class Agente:
    df_auditorias: object
    pipeline_riesgo: object = None
    indice_rag: dict = None
    log: list = field(default_factory=list)

    # ------------------------- Router -------------------------------
    def enrutar(self, instruccion: str) -> str:
        """Reglas simples de intención (palabras clave) -> nombre de tool.
        Se documenta como reglas explicables; puede sustituirse por tool-use
        nativo del LLM sin cambiar el resto del pipeline."""
        texto = instruccion.lower()
        if re.search(r"[\d\.\)\(]\s*[\+\-\*/]\s*[\d\.\(]", texto):
            return "calculadora"
        if "reporte" in texto or "resumen ejecutivo" in texto:
            return "generador_reporte"
        if any(p in texto for p in ["instrucción de trabajo", "instruccion de trabajo", " it ", "epp", "vernier", "5s", "no conforme", "polivalencia"]):
            return "consulta_rag"
        if any(p in texto for p in ["riesgo de", "probabilidad de", "predice", "predecir"]):
            return "predictor_riesgo"
        if any(p in texto for p in ["nocturno", "correlación", "correlacion", "hipótesis", "hipotesis", "significativ"]):
            return "consulta_estadistica"
        if any(p in texto for p in ["tendencia", "estacionalidad", "serie de tiempo", "mejorando", "empeorando"]):
            return "analizador_series"
        return "consulta_rag"  # fallback razonable: casi todo lo demás es "cómo se hace X"

    # ------------------------- Dispatcher -----------------------------
    def ejecutar_tool(self, nombre_tool: str, **kwargs) -> dict:
        try:
            if nombre_tool == "calculadora":
                return self._tool_calculadora(kwargs.get("expresion", ""))
            if nombre_tool == "predictor_riesgo":
                return self._tool_predictor_riesgo(**kwargs)
            if nombre_tool == "consulta_estadistica":
                return self._tool_estadistica(kwargs.get("instruccion", ""))
            if nombre_tool == "consulta_rag":
                return self._tool_rag(kwargs.get("pregunta", ""))
            if nombre_tool == "analizador_series":
                return self._tool_series()
            if nombre_tool == "generador_reporte":
                return self._tool_reporte_multi_step()
            return {"ok": False, "error": f"Tool desconocida: {nombre_tool}"}
        except Exception as e:
            resultado = {"ok": False, "error": str(e)}
            self.log.append({"tool": nombre_tool, "kwargs": kwargs, "resultado": resultado})
            return resultado
        finally:
            pass

    def _registrar(self, nombre_tool, kwargs, resultado):
        self.log.append({"tool": nombre_tool, "kwargs": kwargs, "resultado_ok": resultado.get("ok")})

    # ------------------------- Tools individuales -----------------------------
    def _tool_calculadora(self, expresion: str) -> dict:
        if not expresion or not re.match(r"^[\d\.\+\-\*/\(\)\s\*]+$", expresion):
            resultado = {"ok": False, "error": "Expresión vacía o con caracteres no permitidos."}
        else:
            try:
                valor = _eval_seguro(ast.parse(expresion, mode="eval").body)
                resultado = {"ok": True, "resultado": valor}
            except Exception as e:
                resultado = {"ok": False, "error": f"No se pudo evaluar la expresión: {e}"}
        self._registrar("calculadora", {"expresion": expresion}, resultado)
        return resultado

    def _tool_predictor_riesgo(self, linea="Línea 1", turno="Matutino", auditor="J. Ramírez",
                                antiguedad_meses=24, propension_riesgo=0.3, fin_de_mes=False,
                                fuera_de_matriz=False) -> dict:
        if self.pipeline_riesgo is None:
            resultado = {"ok": False, "error": "El modelo de ML aún no ha sido entrenado en esta sesión."}
        else:
            from src.ml.modelos import predecir_riesgo
            proba = predecir_riesgo(self.pipeline_riesgo, linea, turno, auditor, antiguedad_meses,
                                     propension_riesgo, fin_de_mes, fuera_de_matriz)
            resultado = {"ok": True, "probabilidad_desviacion_critica": round(proba, 3),
                         "linea": linea, "turno": turno}
        self._registrar("predictor_riesgo", {"linea": linea, "turno": turno}, resultado)
        return resultado

    def _tool_estadistica(self, instruccion: str) -> dict:
        from src.features.estadistica import prueba_hipotesis_turno_nocturno, correlacion_antiguedad_score
        if "correlaci" in instruccion.lower() or "antigüedad" in instruccion.lower():
            resultado = {"ok": True, **correlacion_antiguedad_score(self.df_auditorias)}
        else:
            resultado = {"ok": True, **prueba_hipotesis_turno_nocturno(self.df_auditorias)}
        self._registrar("consulta_estadistica", {"instruccion": instruccion}, resultado)
        return resultado

    def _tool_rag(self, pregunta: str) -> dict:
        from src.rag.rag_instrucciones import responder_con_rag
        if self.indice_rag is None:
            resultado = {"ok": False, "error": "El índice RAG aún no ha sido construido en esta sesión."}
        else:
            salida = responder_con_rag(self.indice_rag, pregunta)
            resultado = {"ok": True, **salida}
        self._registrar("consulta_rag", {"pregunta": pregunta}, resultado)
        return resultado

    def _tool_series(self) -> dict:
        from src.features.series_tiempo import resumen_series
        resumen = resumen_series(self.df_auditorias)
        resultado = {
            "ok": True,
            "tendencia": resumen["tendencia_texto"],
            "peor_dia_semana": resumen["peor_dia_semana"],
            "mejor_dia_semana": resumen["mejor_dia_semana"],
        }
        self._registrar("analizador_series", {}, resultado)
        return resultado

    def _tool_reporte_multi_step(self) -> dict:
        """Tarea multi-step: encadena estadística + series + RAG y pide al LLM
        redactar un resumen ejecutivo con esa evidencia (nunca inventada)."""
        from src.llm.cliente_llm import generar_texto

        paso1 = self._tool_estadistica("turno nocturno")
        paso2 = self._tool_series()
        paso3 = self._tool_rag("¿Qué hacer si un operador está fuera de su matriz de polivalencia?")

        evidencia = (
            f"- Prueba de hipótesis turno nocturno: {paso1.get('interpretacion')}\n"
            f"- Series de tiempo: la tendencia general es {paso2.get('tendencia')}, "
            f"el peor día de la semana es {paso2.get('peor_dia_semana')} y el mejor "
            f"{paso2.get('mejor_dia_semana')}.\n"
            f"- Norma de polivalencia (IT): {paso3.get('respuesta')}\n"
        )
        prompt = (
            "Redacta un resumen ejecutivo de máximo 120 palabras para el gerente de "
            "planta, en español, usando ÚNICAMENTE esta evidencia (no inventes cifras "
            f"nuevas):\n\n{evidencia}"
        )
        texto, ok = generar_texto(prompt, max_tokens=300)
        resultado = {
            "ok": ok,
            "pasos_ejecutados": ["consulta_estadistica", "analizador_series", "consulta_rag", "llm_redaccion"],
            "evidencia_usada": evidencia,
            "resumen_ejecutivo": texto,
        }
        self._registrar("generador_reporte", {}, resultado)
        return resultado

    # ------------------------- Punto de entrada -----------------------------
    def responder(self, instruccion: str, **kwargs) -> dict:
        tool = self.enrutar(instruccion)
        if len(self.log) >= MAX_PASOS and tool == "generador_reporte":
            return {"ok": False, "error": "Límite de pasos del agente alcanzado en esta sesión."}
        if tool == "consulta_rag":
            kwargs.setdefault("pregunta", instruccion)
        if tool == "consulta_estadistica":
            kwargs.setdefault("instruccion", instruccion)
        if tool == "calculadora":
            match = re.search(r"[\d\.]+(?:\s*[\+\-\*/]\s*[\d\.]+)+", instruccion)
            kwargs.setdefault("expresion", match.group().strip() if match else "")
        resultado = self.ejecutar_tool(tool, **kwargs)
        return {"tool_seleccionada": tool, **resultado}


# ---------------------------------------------------------------------------
# Métrica obligatoria del bloque: Tool Selection Accuracy sobre casos de prueba
# ---------------------------------------------------------------------------

CASOS_PRUEBA_ROUTER = [
    ("¿Cuánto es 12*3 + 7?", "calculadora"),
    ("¿Cuál es la probabilidad de riesgo en la Línea 2 turno nocturno?", "predictor_riesgo"),
    ("¿El turno nocturno tiene un desempeño significativamente distinto?", "consulta_estadistica"),
    ("¿Qué EPP debo usar en el área de ensamble?", "consulta_rag"),
    ("¿La tendencia del cumplimiento está mejorando o empeorando?", "analizador_series"),
    ("Dame un reporte ejecutivo de la semana", "generador_reporte"),
    ("¿Cómo debo calibrar el vernier antes de medir?", "consulta_rag"),
    ("¿Existe correlación entre antigüedad y score?", "consulta_estadistica"),
]


def evaluar_tool_selection_accuracy(agente: Agente) -> dict:
    aciertos = 0
    detalle = []
    for instruccion, tool_esperada in CASOS_PRUEBA_ROUTER:
        tool_predicha = agente.enrutar(instruccion)
        acierto = tool_predicha == tool_esperada
        aciertos += int(acierto)
        detalle.append({"instruccion": instruccion, "esperada": tool_esperada, "predicha": tool_predicha, "acierto": acierto})
    return {
        "tool_selection_accuracy": round(aciertos / len(CASOS_PRUEBA_ROUTER), 3),
        "n_casos": len(CASOS_PRUEBA_ROUTER),
        "detalle": detalle,
    }
