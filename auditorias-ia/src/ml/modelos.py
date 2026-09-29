"""
Bloque F — ML supervisado, Bloque G — ML no supervisado,
Bloque H — Pipelines, validación cruzada y ensembles.

Problema supervisado elegido (justificación de negocio):
    Clasificación binaria -> ¿la auditoría terminará con al menos una
    respuesta "No OK sin acción" (riesgo alto)?  Esto permite priorizar
    auditorías/operadores de alto riesgo ANTES de que ocurra el hallazgo,
    usando variables que se conocen de antemano (línea, turno, operador,
    antigüedad, fin de mes, si está fuera de su matriz de polivalencia).

Problema no supervisado elegido:
    Clustering de operadores según su patrón de desviaciones por categoría
    de pregunta (Seguridad / Calidad / Proceso / Personal), + PCA para
    visualizar la estructura latente en 2D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import PREGUNTAS, PREGUNTA_IDS

FEATURES_CATEGORICAS = ["linea", "turno", "auditor"]
FEATURES_NUMERICAS = ["antiguedad_meses", "propension_riesgo", "fin_de_mes", "operador_fuera_de_matriz"]
TARGET = "tiene_no_ok_sin_accion"


def _construir_preprocesador() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAS),
            ("num", StandardScaler(), FEATURES_NUMERICAS),
        ]
    )


def preparar_dataset_supervisado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["fin_de_mes"] = df["fin_de_mes"].astype(int)
    df["operador_fuera_de_matriz"] = df["operador_fuera_de_matriz"].astype(int)
    X = df[FEATURES_CATEGORICAS + FEATURES_NUMERICAS]
    y = df[TARGET].astype(int)
    return X, y


def entrenar_y_comparar_modelos(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> dict:
    """
    Bloque H aplicado: separación train/test estratificada, baseline,
    >=2 modelos comparados, validación cruzada y un ensemble (Voting),
    evitando data leakage (el preprocesador se ajusta SOLO con train,
    dentro del Pipeline).
    """
    X, y = preparar_dataset_supervisado(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pre = _construir_preprocesador()

    modelos = {
        "Baseline (mayoría)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    resultados = []
    pipelines_entrenados = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for nombre, modelo in modelos.items():
        pipe = Pipeline([("pre", pre), ("clf", modelo)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1")

        resultados.append(
            {
                "modelo": nombre,
                "accuracy": round(accuracy_score(y_test, y_pred), 3),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
                "roc_auc": round(roc_auc_score(y_test, y_proba), 3) if y_proba is not None and len(set(y_test)) > 1 else np.nan,
                "f1_cv_media": round(cv_scores.mean(), 3),
                "f1_cv_std": round(cv_scores.std(), 3),
            }
        )
        pipelines_entrenados[nombre] = pipe

    # Ensemble: Voting entre los dos mejores modelos "reales" (no baseline)
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state, class_weight="balanced")),
            ("gb", GradientBoostingClassifier(random_state=random_state)),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ],
        voting="soft",
    )
    pipe_ensemble = Pipeline([("pre", pre), ("clf", ensemble)])
    pipe_ensemble.fit(X_train, y_train)
    y_pred_ens = pipe_ensemble.predict(X_test)
    y_proba_ens = pipe_ensemble.predict_proba(X_test)[:, 1]
    resultados.append(
        {
            "modelo": "Ensemble (Voting RF+GB+LR)",
            "accuracy": round(accuracy_score(y_test, y_pred_ens), 3),
            "precision": round(precision_score(y_test, y_pred_ens, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred_ens, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred_ens), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba_ens), 3),
            "f1_cv_media": np.nan,
            "f1_cv_std": np.nan,
        }
    )
    pipelines_entrenados["Ensemble (Voting RF+GB+LR)"] = pipe_ensemble

    mejor_nombre = max(
        [r for r in resultados if r["modelo"] != "Baseline (mayoría)"], key=lambda r: r["f1"]
    )["modelo"]

    matriz_confusion = confusion_matrix(y_test, pipelines_entrenados[mejor_nombre].predict(X_test))
    reporte_txt = classification_report(y_test, pipelines_entrenados[mejor_nombre].predict(X_test), zero_division=0)

    # Importancia de variables (si el mejor modelo la soporta)
    importancias = None
    mejor_pipe = pipelines_entrenados[mejor_nombre]
    clf_final = mejor_pipe.named_steps["clf"]
    if hasattr(clf_final, "feature_importances_"):
        nombres_ohe = mejor_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(FEATURES_CATEGORICAS)
        nombres = list(nombres_ohe) + FEATURES_NUMERICAS
        importancias = pd.DataFrame(
            {"variable": nombres, "importancia": clf_final.feature_importances_}
        ).sort_values("importancia", ascending=False).head(12)

    return {
        "tabla_resultados": pd.DataFrame(resultados),
        "mejor_modelo": mejor_nombre,
        "matriz_confusion": matriz_confusion,
        "reporte_texto": reporte_txt,
        "importancias": importancias,
        "pipelines": pipelines_entrenados,
        "X_test": X_test,
        "y_test": y_test,
    }


def predecir_riesgo(pipeline, linea, turno, auditor, antiguedad_meses, propension_riesgo, fin_de_mes, fuera_de_matriz) -> float:
    """Usado por el módulo de Agentes como Tool 'predictor_riesgo'."""
    fila = pd.DataFrame([{
        "linea": linea,
        "turno": turno,
        "auditor": auditor,
        "antiguedad_meses": antiguedad_meses,
        "propension_riesgo": propension_riesgo,
        "fin_de_mes": int(fin_de_mes),
        "operador_fuera_de_matriz": int(fuera_de_matriz),
    }])
    proba = pipeline.predict_proba(fila)[0, 1]
    return float(proba)


# ---------------------------------------------------------------------------
# Bloque G — No supervisado: clustering de operadores + PCA
# ---------------------------------------------------------------------------

def construir_perfil_operadores(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de desviación por categoría de pregunta, por operador."""
    categorias = {}
    for p in PREGUNTAS:
        categorias.setdefault(p["categoria"], []).append(p["id"])

    perfil = df.groupby("operador_id", observed=True).agg(
        n_auditorias=("audit_id", "count"),
        antiguedad_meses=("antiguedad_meses", "first"),
        score_promedio=("score_cumplimiento", "mean"),
    )

    for categoria, preguntas_cat in categorias.items():
        tasa_desviacion = df.groupby("operador_id", observed=True).apply(
            lambda g: (g[preguntas_cat] != "OK").mean().mean()
        )
        perfil[f"tasa_desviacion_{categoria}"] = tasa_desviacion

    return perfil.reset_index()


def clustering_operadores(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> dict:
    perfil = construir_perfil_operadores(df)
    cols_features = [c for c in perfil.columns if c.startswith("tasa_desviacion_")] + ["score_promedio"]
    X = perfil[cols_features].fillna(0)

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    perfil["cluster"] = kmeans.fit_predict(X_esc)

    pca = PCA(n_components=2, random_state=random_state)
    comp = pca.fit_transform(X_esc)
    perfil["pca_1"] = comp[:, 0]
    perfil["pca_2"] = comp[:, 1]

    varianza_explicada = pca.explained_variance_ratio_.sum()

    resumen_clusters = perfil.groupby("cluster")[cols_features].mean().round(3)

    return {
        "perfil": perfil,
        "resumen_clusters": resumen_clusters,
        "varianza_explicada_pca": float(varianza_explicada),
        "features_usadas": cols_features,
    }
