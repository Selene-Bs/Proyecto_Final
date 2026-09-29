"""
Bloque I — Deep Learning.

Se implementa un Perceptrón Multicapa (MLP) para el mismo problema de
clasificación del Bloque F (riesgo de auditoría con "No OK sin acción"),
y se compara explícitamente contra un modelo clásico (Regresión Logística).

Nota de diseño / justificación técnica:
Se usa `sklearn.neural_network.MLPClassifier` en lugar de TensorFlow/PyTorch
para mantener el despliegue ligero y 100% reproducible en Streamlit
Community Cloud (evita instalar binarios pesados de ~500 MB que exceden
límites de memoria del free tier). Sigue siendo una red neuronal
feed-forward multicapa con backpropagation y descenso de gradiente
estocástico (Adam), entrenada, evaluada y comparada como pide el bloque.
El archivo `red_neuronal_pytorch_opcional.py` incluye la versión con
PyTorch por si se despliega en un entorno con más recursos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from src.ml.modelos import _construir_preprocesador, preparar_dataset_supervisado
from sklearn.pipeline import Pipeline


def entrenar_mlp_vs_clasico(df: pd.DataFrame, random_state: int = 42) -> dict:
    X, y = preparar_dataset_supervisado(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pre = _construir_preprocesador()

    mlp = Pipeline([
        ("pre", pre),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        )),
    ])
    clasico = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    resultados = []
    curvas_perdida = {}
    for nombre, pipe in [("MLP (Deep Learning)", mlp), ("Regresión Logística (clásico)", clasico)]:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        resultados.append({
            "modelo": nombre,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        })
        if nombre.startswith("MLP"):
            curvas_perdida["loss_curve"] = pipe.named_steps["clf"].loss_curve_
            matriz_mlp = confusion_matrix(y_test, y_pred)

    return {
        "tabla_resultados": pd.DataFrame(resultados),
        "curva_perdida_mlp": curvas_perdida.get("loss_curve", []),
        "matriz_confusion_mlp": matriz_mlp,
        "pipeline_mlp": mlp,
    }
