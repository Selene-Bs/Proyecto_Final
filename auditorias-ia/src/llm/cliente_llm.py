"""
Bloque L — LLM.

Wrapper delgado sobre la API de Anthropic (Claude) para tareas reales de la
aplicación: resumen ejecutivo de auditorías, explicación de resultados de
ML/estadística en lenguaje de piso de planta, y generación de reportes.

Control de calidad exigido por el bloque:
  - prompt: se centraliza en `_PROMPT_SISTEMA` y en las plantillas por tarea,
  - formato de respuesta: se le pide explícitamente texto plano o JSON según
    la tarea, y se valida el JSON antes de usarlo,
  - alucinaciones: el prompt obliga al modelo a basarse SOLO en los datos
    que se le entregan en el mensaje (nunca a inventar cifras),
  - errores: cualquier fallo de red/API se captura y se informa en la UI en
    vez de tronar la aplicación,
  - validación de salida: `generar_json` reintenta una vez si el JSON viene
    mal formado.
"""

from __future__ import annotations

import json
import os

MODELO_LLM = "claude-sonnet-5"  # ver docs.claude.com/en/docs/about-claude/models para el ID vigente

_PROMPT_SISTEMA = (
    "Eres un asistente experto en calidad y manufactura que ayuda a "
    "interpretar resultados de auditorías de proceso en una línea de "
    "producción. Responde siempre en español, de forma breve, concreta y "
    "orientada a acción. Básate ÚNICAMENTE en los datos y el contexto que "
    "se te proporcionan en el mensaje; si no hay evidencia suficiente para "
    "responder algo, dilo explícitamente en vez de inventar cifras o "
    "hechos."
)


def _cliente_disponible():
    """Crea el cliente de Anthropic solo si hay una API key configurada."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "No se encontró ANTHROPIC_API_KEY en las variables de entorno / secretos de la app."
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key), None
    except Exception as e:  # pragma: no cover
        return None, f"No se pudo inicializar el cliente de Anthropic: {e}"


def generar_texto(prompt_usuario: str, max_tokens: int = 600) -> tuple[str, bool]:
    """Devuelve (texto, ok). Si ok=False, `texto` contiene el mensaje de error para mostrar en la UI."""
    cliente, error = _cliente_disponible()
    if cliente is None:
        return error, False
    try:
        respuesta = cliente.messages.create(
            model=MODELO_LLM,
            max_tokens=max_tokens,
            system=_PROMPT_SISTEMA,
            messages=[{"role": "user", "content": prompt_usuario}],
        )
        texto = "".join(bloque.text for bloque in respuesta.content if bloque.type == "text")
        return texto.strip(), True
    except Exception as e:
        return f"Error al llamar al LLM: {e}", False


def generar_json(prompt_usuario: str, max_tokens: int = 600, reintentos: int = 1) -> tuple[dict | None, bool, str]:
    """
    Pide una respuesta SOLO JSON y la valida. Devuelve (dict_o_None, ok, mensaje).
    Reintenta una vez pidiendo explícitamente corregir el formato si falla el parseo.
    """
    instruccion_formato = (
        "\n\nResponde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, "
        "sin explicación y sin backticks de markdown."
    )
    prompt_actual = prompt_usuario + instruccion_formato
    ultimo_texto = ""
    for intento in range(reintentos + 1):
        texto, ok = generar_texto(prompt_actual, max_tokens=max_tokens)
        if not ok:
            return None, False, texto
        ultimo_texto = texto
        limpio = texto.strip().strip("`")
        if limpio.startswith("json"):
            limpio = limpio[4:].strip()
        try:
            return json.loads(limpio), True, "OK"
        except json.JSONDecodeError:
            prompt_actual = (
                prompt_usuario
                + "\n\nTu respuesta anterior no era JSON válido. Corrige y responde "
                "SOLO con JSON válido, sin texto extra:\n" + ultimo_texto
            )
    return None, False, "El modelo no devolvió JSON válido tras reintentar."
