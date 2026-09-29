"""
Bloque M — RAG (Retrieval-Augmented Generation).

Base de conocimiento: las Instrucciones de Trabajo (IT) de la planta
(docs/instrucciones_trabajo/*.txt). Pipeline completo:

  1. documentos            -> se leen los .txt de IT
  2. chunking               -> se parte cada IT en fragmentos por párrafo/numeral
  3. embeddings              -> Sentence-Transformers (o fallback TF-IDF+SVD)
  4. índice / vector store   -> matriz en memoria (numpy), suficiente para
                                el tamaño de la base documental de la planta
  5. consulta + Top-k        -> similitud coseno contra la pregunta del usuario
  6. construcción de contexto-> se concatenan los chunks recuperados con su fuente
  7. respuesta con LLM        -> Claude responde SOLO con ese contexto
  8. fuentes / metadatos      -> se listan el archivo y el fragmento usados
  9. comportamiento sin evidencia -> si la similitud máxima es baja, el
     sistema responde que no hay evidencia suficiente en vez de inventar.

Evaluación del retriever (obligatoria por el bloque): se construye un
pequeño conjunto de prueba pregunta -> documento_esperado y se mide
Recall@k y MRR (ver `evaluar_retriever`).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

from src.config import DOCS_DIR
from src.nlp.embeddings import obtener_embeddings
from src.llm.cliente_llm import generar_texto

UMBRAL_SIN_EVIDENCIA = 0.18


def cargar_documentos(base_dir: str = DOCS_DIR) -> list[dict]:
    rutas = sorted(glob.glob(os.path.join(base_dir, "*.txt")))
    documentos = []
    for ruta in rutas:
        with open(ruta, "r", encoding="utf-8") as f:
            documentos.append({"archivo": os.path.basename(ruta), "texto": f.read()})
    return documentos


def chunkear_documento(texto: str, min_len: int = 40) -> list[str]:
    """Chunking por numeral/párrafo (los IT ya están redactados como listas numeradas)."""
    partes = re.split(r"\n(?=\d+\.\s)|\n\n", texto)
    chunks = [p.strip().replace("\n", " ") for p in partes if len(p.strip()) >= min_len]
    return chunks


def construir_indice(base_dir: str = DOCS_DIR) -> dict:
    """
    Construye el índice documental. Cuando hay un modelo Sentence-Transformers
    disponible, los embeddings de los chunks se calculan una sola vez aquí
    (espacio vectorial estable entre consultas). Cuando se usa el fallback
    TF-IDF+SVD, el ajuste (fit) depende del corpus, así que cada consulta
    reajusta el espacio junto con la consulta (ver `recuperar_top_k`) para
    garantizar que documentos y consulta vivan en el MISMO espacio vectorial.
    """
    documentos = cargar_documentos(base_dir)
    chunks, fuentes = [], []
    for doc in documentos:
        for chunk in chunkear_documento(doc["texto"]):
            chunks.append(chunk)
            fuentes.append(doc["archivo"])

    embeddings, metodo = obtener_embeddings(chunks)
    return {"chunks": chunks, "fuentes": fuentes, "embeddings": embeddings, "metodo_embeddings": metodo}


def recuperar_top_k(indice: dict, consulta: str, k: int = 3) -> list[dict]:
    chunks = indice["chunks"]
    # Recalcular embeddings de chunks + consulta EN CONJUNTO garantiza que
    # ambos queden en el mismo espacio vectorial, sea cual sea el método
    # (Sentence-Transformers real o el fallback TF-IDF+SVD, que se reajusta
    # por corpus). Con la base documental de este proyecto (unas pocas
    # decenas de chunks) el costo es despreciable.
    embeddings, metodo = obtener_embeddings(chunks + [consulta])
    emb_docs, emb_query = embeddings[:-1], embeddings[-1]
    indice["metodo_embeddings"] = metodo

    normas = np.linalg.norm(emb_docs, axis=1) * np.linalg.norm(emb_query)
    normas[normas == 0] = 1e-9
    sims = (emb_docs @ emb_query) / normas

    orden = np.argsort(sims)[::-1][:k]
    return [
        {"chunk": chunks[i], "fuente": indice["fuentes"][i], "similitud": float(sims[i])}
        for i in orden
    ]


def responder_con_rag(indice: dict, pregunta: str, k: int = 3) -> dict:
    resultados = recuperar_top_k(indice, pregunta, k=k)
    similitud_max = max((r["similitud"] for r in resultados), default=0.0)

    if similitud_max < UMBRAL_SIN_EVIDENCIA:
        return {
            "respuesta": (
                "No encontré evidencia suficiente en las Instrucciones de Trabajo vigentes "
                "para responder esta pregunta con confianza. Verifica con el líder de proceso "
                "o revisa si falta documentar este tema."
            ),
            "fuentes": [],
            "contexto_usado": "",
            "similitud_max": similitud_max,
            "llm_ok": True,
        }

    contexto = "\n\n".join(f"[Fuente: {r['fuente']}]\n{r['chunk']}" for r in resultados)
    prompt = (
        "Usa ÚNICAMENTE el siguiente contexto extraído de las Instrucciones de Trabajo (IT) "
        "de la planta para responder la pregunta del usuario. Si el contexto no alcanza para "
        "responder, dilo explícitamente. Cita el nombre del archivo IT relevante en tu respuesta.\n\n"
        f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"
    )
    respuesta_llm, ok = generar_texto(prompt, max_tokens=400)

    return {
        "respuesta": respuesta_llm,
        "fuentes": sorted(set(r["fuente"] for r in resultados)),
        "contexto_usado": contexto,
        "similitud_max": similitud_max,
        "llm_ok": ok,
    }


# ---------------------------------------------------------------------------
# Evaluación del Retriever: Recall@k y MRR sobre un conjunto de prueba fijo
# ---------------------------------------------------------------------------

CONJUNTO_PRUEBA = [
    {"pregunta": "¿Qué EPP es obligatorio usar en producción?", "documento_esperado": "IT-EPP-001.txt"},
    {"pregunta": "¿Cómo debo calibrar y verificar el cero del vernier?", "documento_esperado": "IT-VERNIER-002.txt"},
    {"pregunta": "¿Qué tolerancia dimensional tiene la pieza?", "documento_esperado": "IT-VERNIER-002.txt"},
    {"pregunta": "¿Qué pasa si un operador cubre una línea sin estar certificado?", "documento_esperado": "IT-POLIVALENCIA-004.txt"},
    {"pregunta": "¿Cómo se debe rotular el producto no conforme?", "documento_esperado": "IT-NOCONFORME-006.txt"},
    {"pregunta": "¿Qué significa la 'S' de estandarizar en 5S?", "documento_esperado": "IT-5S-005.txt"},
    {"pregunta": "¿Puedo cambiar la secuencia de pasos de mi estación por mi cuenta?", "documento_esperado": "IT-PROCESO-003.txt"},
]


def evaluar_retriever(indice: dict, k: int = 3) -> dict:
    aciertos_recall = 0
    reciprocal_ranks = []

    detalle = []
    for caso in CONJUNTO_PRUEBA:
        recuperados = recuperar_top_k(indice, caso["pregunta"], k=k)
        fuentes_recuperadas = [r["fuente"] for r in recuperados]
        acierto = caso["documento_esperado"] in fuentes_recuperadas
        aciertos_recall += int(acierto)

        rr = 0.0
        if caso["documento_esperado"] in fuentes_recuperadas:
            rango = fuentes_recuperadas.index(caso["documento_esperado"]) + 1
            rr = 1.0 / rango
        reciprocal_ranks.append(rr)

        detalle.append({
            "pregunta": caso["pregunta"],
            "esperado": caso["documento_esperado"],
            "recuperado_top1": fuentes_recuperadas[0] if fuentes_recuperadas else None,
            "acierto_en_top_k": acierto,
        })

    recall_at_k = aciertos_recall / len(CONJUNTO_PRUEBA)
    mrr = float(np.mean(reciprocal_ranks))

    return {
        "k": k,
        "n_preguntas_prueba": len(CONJUNTO_PRUEBA),
        f"recall_at_{k}": round(recall_at_k, 3),
        "mrr": round(mrr, 3),
        "detalle": detalle,
    }
