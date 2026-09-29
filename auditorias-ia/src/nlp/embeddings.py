"""
Bloque K — Embeddings y Transformers.

Usa Sentence Transformers (modelo multilingüe pequeño) para representar
los comentarios de auditoría como vectores densos y hacer búsqueda
semántica, comparándolo explícitamente contra TF-IDF (representación
clásica dispersa) para la misma consulta.

Si el modelo no se puede descargar (por ejemplo, sin acceso a internet en
el entorno de despliegue), se degrada de forma explícita a un embedding
basado en TF-IDF + SVD truncado, dejando claro en la interfaz cuál
representación se está usando realmente (nunca falla en silencio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

MODELO_ST = "paraphrase-multilingual-MiniLM-L12-v2"

_modelo_cache = {"modelo": None, "disponible": None}


def _cargar_sentence_transformer():
    if _modelo_cache["disponible"] is not None:
        return _modelo_cache["modelo"]
    try:
        from sentence_transformers import SentenceTransformer
        _modelo_cache["modelo"] = SentenceTransformer(MODELO_ST)
        _modelo_cache["disponible"] = True
    except Exception:
        _modelo_cache["modelo"] = None
        _modelo_cache["disponible"] = False
    return _modelo_cache["modelo"]


def obtener_embeddings(textos: list[str]) -> tuple[np.ndarray, str]:
    """Devuelve (matriz_embeddings, nombre_metodo_usado)."""
    modelo = _cargar_sentence_transformer()
    if modelo is not None:
        vectores = modelo.encode(textos, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectores), f"Sentence-Transformers ({MODELO_ST})"

    # Fallback: TF-IDF + SVD como pseudo-embedding denso
    vectorizador = TfidfVectorizer(max_features=500)
    X = vectorizador.fit_transform(textos)
    n_componentes = min(50, X.shape[1] - 1, X.shape[0] - 1) or 1
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    vectores = svd.fit_transform(X)
    norma = np.linalg.norm(vectores, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    vectores = vectores / norma
    return vectores, "TF-IDF + SVD (fallback, sin acceso a modelo Transformer)"


def busqueda_semantica(df: pd.DataFrame, consulta: str, top_k: int = 5) -> tuple[pd.DataFrame, str]:
    """Busca los comentarios de auditoría más similares semánticamente a `consulta`."""
    textos = df["comentario_auditor"].astype(str).tolist()
    embeddings, metodo = obtener_embeddings(textos + [consulta])
    emb_docs, emb_query = embeddings[:-1], embeddings[-1:]

    sims = cosine_similarity(emb_query, emb_docs).flatten()
    resultado = df.copy()
    resultado["similitud_semantica"] = sims
    resultado = resultado.sort_values("similitud_semantica", ascending=False).head(top_k)
    return resultado[["audit_id", "fecha", "linea", "turno", "comentario_auditor", "similitud_semantica"]], metodo


def comparar_tfidf_vs_embeddings(df: pd.DataFrame, consulta: str, top_k: int = 5) -> dict:
    """Compara explícitamente el ranking de TF-IDF clásico vs. embeddings densos."""
    from src.nlp.nlp_clasico import limpiar_texto

    textos = df["comentario_auditor"].astype(str).tolist()
    textos_limpios = [limpiar_texto(t) for t in textos] + [limpiar_texto(consulta)]

    vectorizador = TfidfVectorizer(max_features=300)
    X = vectorizador.fit_transform(textos_limpios)
    sims_tfidf = cosine_similarity(X[-1], X[:-1]).flatten()

    resultado_tfidf = df.copy()
    resultado_tfidf["similitud_tfidf"] = sims_tfidf
    resultado_tfidf = resultado_tfidf.sort_values("similitud_tfidf", ascending=False).head(top_k)

    resultado_embeddings, metodo = busqueda_semantica(df, consulta, top_k=top_k)

    return {
        "tfidf": resultado_tfidf[["audit_id", "comentario_auditor", "similitud_tfidf"]],
        "embeddings": resultado_embeddings,
        "metodo_embeddings": metodo,
    }
