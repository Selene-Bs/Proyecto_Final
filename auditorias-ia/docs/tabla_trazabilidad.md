# Tabla de trazabilidad — Diplomado de Python y Análisis de Datos

| Tema del diplomado | Dónde se aplicó (módulo/archivo) | Evidencia / resultado |
|---|---|---|
| CSV / Pandas (Bloque A) | `src/preprocessing/pipeline_datos.py`, `src/data/generate_synthetic_data.py` | Dataset procesado de 1,415 auditorías; log de limpieza (duplicados, imputaciones, atípicos) visible en la página "Pipeline y EDA" |
| EDA (Bloque B) | `app/pages/1_Pipeline_y_EDA.py` | Gráficas de distribución, boxplots por línea/turno, dispersión antigüedad vs. score, hallazgos escritos |
| Estadística (Bloque C) | `src/features/estadistica.py`, `app/pages/2_Estadistica.py` | Prueba t de Welch (turno nocturno, p<0.05, H0 rechazada), correlación Pearson/Spearman, IC 95% por línea |
| Series temporales (Bloque D) | `src/features/series_tiempo.py`, `app/pages/3_Series_Temporales.py` | Tendencia lineal, estacionalidad por día de semana, detección de cambio de régimen (z-score) |
| Feature Engineering (Bloque E) | `src/features/series_tiempo.py` | Medias móviles 7/30 días, lags (1,7,14), componente estacional día-de-semana |
| ML supervisado (Bloque F) | `src/ml/modelos.py`, `app/pages/4_Machine_Learning.py` | 4 modelos comparados + baseline; métricas Accuracy/Precision/Recall/F1/ROC-AUC en tabla |
| ML no supervisado (Bloque G) | `src/ml/modelos.py` (`clustering_operadores`) | K-Means + PCA 2D; perfil de riesgo por cluster de operadores |
| Pipelines/validación/ensembles (Bloque H) | `src/ml/modelos.py` | `sklearn.Pipeline` (sin data leakage), `StratifiedKFold` 5-fold, `VotingClassifier` |
| Deep Learning (Bloque I) | `src/deep_learning/red_neuronal.py`, `app/pages/5_Deep_Learning.py` | MLP (32,16) vs. Regresión Logística, curva de pérdida, matriz de confusión |
| NLP clásico (Bloque J) | `src/nlp/nlp_clasico.py`, `app/pages/6_NLP_y_Embeddings.py` | TF-IDF + clasificador de severidad (Accuracy ≈0.73), palabras clave por clase |
| Embeddings/Transformers (Bloque K) | `src/nlp/embeddings.py` | Búsqueda semántica con Sentence-Transformers vs. TF-IDF, comparación lado a lado |
| LLM (Bloque L) | `src/llm/cliente_llm.py` | Wrapper con control de prompt/formato/errores; usado en RAG y en el agente |
| RAG (Bloque M) | `src/rag/rag_instrucciones.py`, `app/pages/7_RAG_Instrucciones.py` | Pipeline completo con fuentes citadas; **Recall@3 = 1.0, MRR = 1.0** sobre 7 preguntas de prueba |
| Agentes / Tool Calling (Bloque N) | `src/agents/agente.py`, `app/pages/8_Agente_IA.py` | 6 tools, dispatcher con manejo de errores, tarea multi-step (`generador_reporte`); **Tool Selection Accuracy = 100%** |
| Fourier (Bloque O) | `src/signals/fourier.py`, `app/pages/9_Fourier_y_Wavelets.py` | FFT sobre señal dimensional; ciclo semanal (~7 días) identificado e interpretado |
| Wavelets (Bloque P) | `src/signals/wavelets.py` | DWT `db4`, energía por nivel, denoising, detección de anomalías validada contra ráfagas simuladas |

## Notas de evaluación
- Todas las métricas de esta tabla se recalculan en vivo al correr la aplicación (no son
  valores fijos "quemados" en el código); pueden variar ligeramente entre corridas porque
  el split de train/test usa una semilla fija pero el tamaño de muestra es limitado.
- El detalle completo de cada prueba (hipótesis, datos usados, interpretación) está
  documentado como docstring en el módulo correspondiente de `src/`.
