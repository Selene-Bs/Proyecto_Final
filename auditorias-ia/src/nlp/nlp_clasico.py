"""
Bloque J — NLP clásico.

Fuente textual: el campo `comentario_auditor` (texto libre que el auditor
escribe cuando encuentra una desviación). Se aplican:

  - limpieza de texto y tokenización simple (sin dependencias de descarga
    en runtime, para que funcione igual en local y en la nube),
  - TF-IDF,
  - clasificación de texto: predecir la severidad de la desviación
    ("sin riesgo" / "riesgo medio" / "riesgo alto") a partir SOLO del
    comentario escrito por el auditor,
  - extracción de palabras clave por severidad,
  - similitud textual entre comentarios (para agrupar quejas parecidas).
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS_ES = {
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "unos", "unas",
    "que", "no", "se", "su", "sus", "al", "con", "por", "para", "es", "lo", "le", "les",
    "correctamente", "correcta", "correcto", "sin", "durante", "muy", "más", "o",
}

TOKEN_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower()
    tokens = TOKEN_RE.findall(texto)
    tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]
    return " ".join(tokens)


def preparar_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["comentario_limpio"] = df["comentario_auditor"].apply(limpiar_texto)
    # etiqueta de severidad: la peor respuesta de la auditoría
    orden_severidad = {"OK": 0, "No OK con acción": 1, "No OK sin acción": 2}
    from src.config import PREGUNTA_IDS
    peor = df[PREGUNTA_IDS].apply(lambda fila: max(orden_severidad[v] for v in fila), axis=1)
    df["severidad"] = peor.map({0: "sin riesgo", 1: "riesgo medio", 2: "riesgo alto"})
    return df


def entrenar_clasificador_texto(df: pd.DataFrame, random_state: int = 42) -> dict:
    df = preparar_corpus(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df["comentario_limpio"], df["severidad"], test_size=0.25,
        random_state=random_state, stratify=df["severidad"],
    )

    vectorizador = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_train_tfidf = vectorizador.fit_transform(X_train)
    X_test_tfidf = vectorizador.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    metricas = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),
        "reporte": classification_report(y_test, y_pred, zero_division=0),
    }

    # Palabras clave por clase: mayor coeficiente en la regresión logística
    palabras_clave = {}
    vocab = np.array(vectorizador.get_feature_names_out())
    for i, clase in enumerate(clf.classes_):
        top_idx = np.argsort(clf.coef_[i])[-8:][::-1]
        palabras_clave[clase] = vocab[top_idx].tolist()

    return {
        "vectorizador": vectorizador,
        "modelo": clf,
        "metricas": metricas,
        "palabras_clave": palabras_clave,
        "df_corpus": df,
    }


def comentarios_mas_similares(df: pd.DataFrame, indice_referencia: int, top_k: int = 5) -> pd.DataFrame:
    """Similitud textual (TF-IDF + coseno) entre comentarios de auditoría."""
    df = preparar_corpus(df).reset_index(drop=True)
    vectorizador = TfidfVectorizer(max_features=300)
    X = vectorizador.fit_transform(df["comentario_limpio"])
    sims = cosine_similarity(X[indice_referencia], X).flatten()
    df["similitud"] = sims
    resultado = df.sort_values("similitud", ascending=False).iloc[1: top_k + 1]
    return resultado[["audit_id", "linea", "turno", "comentario_auditor", "severidad", "similitud"]]
