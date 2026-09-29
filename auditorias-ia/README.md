# 🏭 Sistema Inteligente de Auditorías de Proceso

**Diplomado de Python y Análisis de Datos — Universidad Marista**
Proyecto Final: solución de datos e inteligencia artificial desplegada en la nube.

---

## 1. Nombre del proyecto
**Sistema Inteligente de Auditorías de Proceso en Línea de Producción.**

## 2. Integrantes
_(completar con los integrantes del equipo)_
- Nombre 1
- Nombre 2
- Nombre 3

## 3. Problema
Las auditorías de proceso en piso de producción (EPP, uso correcto del vernier, apego a
instrucción de trabajo, matriz de polivalencia, 5S, registro de calidad, tiempo de ciclo,
producto no conforme) se capturan hoy en papel o en hojas de cálculo sueltas. No existe una
vista integrada que permita: detectar patrones de riesgo por línea/turno/operador, anticipar
auditorías con hallazgos críticos, ni consultar de forma natural los procedimientos vigentes.

## 4. Objetivo
### Objetivo general
Construir y desplegar en la nube una aplicación que digitalice la auditoría de proceso y
aplique análisis estadístico, Machine Learning, Deep Learning, NLP, LLM/RAG y análisis de
señales (Fourier/Wavelets) para convertir esas auditorías en decisiones de mejora continua.

### Objetivos específicos
1. Digitalizar la cédula de auditoría (captura + almacenamiento reproducible).
2. Cuantificar con estadística inferencial si turno, línea u operador afectan el cumplimiento.
3. Predecir el riesgo de una auditoría con hallazgo crítico antes de que ocurra.
4. Segmentar operadores por patrón de desviación para dirigir capacitación.
5. Clasificar y buscar semánticamente los comentarios libres del auditor.
6. Responder preguntas de piso sobre Instrucciones de Trabajo con un asistente RAG.
7. Analizar la señal dimensional de proceso en frecuencia y multiresolución para detectar
   periodicidades y anomalías (desgaste de herramienta).

## 5. Arquitectura

```
┌─────────────────────────────┐
│   Streamlit (Frontend/UI)   │  app/Home.py + app/pages/*.py
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│                        src/ (lógica de negocio)                │
│  preprocessing/  → Pipeline de datos (Bloque A)                │
│  features/       → Estadística (C), Series de tiempo (D/E)     │
│  ml/             → Supervisado / no supervisado (F/G/H)        │
│  deep_learning/  → MLP (I)                                     │
│  nlp/            → NLP clásico (J), Embeddings (K)              │
│  llm/            → Cliente Claude API (L)                       │
│  rag/            → RAG sobre Instrucciones de Trabajo (M)       │
│  agents/         → Router + Tool Calling (N)                    │
│  signals/        → Fourier (O), Wavelets (P)                    │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────┐
│  data/*.csv + docs/*.txt      │  Fuente de datos (sintética, reproducible)
└────────────────────────────────┘
```

La aplicación es modular: cada bloque del diplomado vive en su propio módulo de `src/` y se
consume desde una página independiente de Streamlit, sin acoplarse entre sí.

## 6. Stack tecnológico
- **Lenguaje:** Python 3.12
- **Frontend/App:** Streamlit + Plotly
- **Datos:** Pandas, NumPy
- **Estadística:** SciPy
- **ML/DL clásico:** scikit-learn (incluye `MLPClassifier` como red neuronal)
- **Señales:** NumPy (FFT), PyWavelets (DWT)
- **NLP/Embeddings:** scikit-learn (TF-IDF) + Sentence-Transformers (con *fallback* automático)
- **LLM:** Claude API (Anthropic) vía SDK `anthropic`
- **Despliegue:** Streamlit Community Cloud
- **Pruebas:** pytest

## 7. Fuente de datos
Dataset **sintético pero estadísticamente realista**, generado por
`src/data/generate_synthetic_data.py` (semilla fija `numpy.random.default_rng(42)` para
reproducibilidad total). Genera:
- `data/auditorias.csv` — 1,400+ auditorías de proceso (8 preguntas, 3 respuestas posibles).
- `data/operadores.csv` — 40 operadores con matriz de polivalencia y antigüedad.
- `data/senales_dimensionales.csv` — 5,760 lecturas de señal dimensional de alta frecuencia.
- `docs/instrucciones_trabajo/*.txt` — 6 Instrucciones de Trabajo (base documental del RAG).

> Para usar datos reales de planta, sustituye el generador por tu propia fuente (ERP/MES/Excel)
> manteniendo el mismo esquema de columnas; el resto del pipeline no requiere cambios.

## 8. Módulos de la aplicación
| Página | Bloque(s) del diplomado |
|---|---|
| 📊 Pipeline y EDA | A, B |
| 📈 Estadística | C |
| ⏱️ Series de Tiempo | D, E |
| 🤖 Machine Learning | F, G, H |
| 🧠 Deep Learning | I |
| 💬 NLP y Embeddings | J, K |
| 📄 RAG - Instrucciones | M |
| 🕹️ Agente IA | L, N |
| 🌊 Fourier y Wavelets | O, P |
| 📝 Registrar Auditoría | A (captura) |

## 9. Modelos
- **Supervisado:** Regresión Logística, Random Forest, Gradient Boosting, Ensemble Voting
  (clasificación binaria: riesgo de hallazgo crítico).
- **No supervisado:** K-Means + PCA (segmentación de operadores).
- **Deep Learning:** `MLPClassifier` (32,16), ReLU, Adam.
- **NLP:** TF-IDF + Regresión Logística (severidad desde texto).
- **Embeddings:** Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2` (fallback
  TF-IDF+SVD si no hay acceso al modelo en el entorno de despliegue).
- **LLM:** Claude (Anthropic API) para redacción de resúmenes y respuestas RAG.

## 10. Métricas (resultados de referencia con el dataset sintético)
| Métrica | Valor de referencia |
|---|---|
| F1 mejor modelo supervisado (test) | ~0.55–0.65 según el split |
| Accuracy clasificador NLP de severidad | ~0.73 |
| Recall@3 del Retriever RAG | **1.0** (7/7 preguntas de prueba) |
| MRR del Retriever RAG | **1.0** |
| Tool Selection Accuracy del agente | **100%** (8/8 casos de prueba) |
| Anomalías detectadas por Wavelets | Coinciden con las ráfagas simuladas |

Estas cifras se recalculan en vivo cada vez que corre la app (no están hardcodeadas).

## 11. Fourier / Wavelets
- **Fourier:** FFT sobre la señal dimensional (24 muestras/día); se identifica un ciclo
  ≈7 días (ritmo semanal de producción) y un ciclo ≈1 día (arranque de turno).
- **Wavelets:** DWT (familia `db4`, 4 niveles) para energía por nivel, denoising por
  umbralización suave y detección de anomalías por energía local del detalle fino,
  validada contra las ráfagas de "desgaste de herramienta" inyectadas en la simulación.

## 12. LLM / RAG / Agentes
- **LLM (Bloque L):** wrapper en `src/llm/cliente_llm.py`; controla prompt, formato
  (texto/JSON validado), manejo de errores y evita alucinaciones (solo responde con datos
  entregados en el mensaje).
- **RAG (Bloque M):** pipeline completo documentos → chunking → embeddings → índice →
  Top-k → contexto → respuesta con fuentes citadas → manejo explícito de "sin evidencia".
- **Agentes (Bloque N):** router + dispatcher con 6 tools (`calculadora`,
  `predictor_riesgo`, `consulta_estadistica`, `consulta_rag`, `analizador_series`,
  `generador_reporte` como tarea *multi-step*), logging, límite de pasos y manejo de errores.

## 13. Instrucciones de desarrollo (local)

```bash
# 1. Clonar el repositorio y crear entorno virtual
git clone <url-del-repo>
cd auditorias-ia
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Generar el dataset sintético (se genera automáticamente al primer uso,
#    pero puedes forzarlo manualmente):
python -m src.data.generate_synthetic_data

# 4. Ejecutar pruebas
pytest tests/ -v

# 5. Levantar la app
streamlit run app/Home.py
```

## 14. Variables de entorno necesarias
| Variable | Requerida | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | Opcional* | Habilita las respuestas del LLM (RAG y reporte del agente). |

\* Sin esta variable, la app funciona igual (EDA, estadística, ML, DL, NLP, Fourier,
Wavelets, routing del agente), solo la redacción final con el LLM muestra un mensaje de
error controlado en vez de fallar.

En Streamlit Community Cloud se configura en **Settings → Secrets**:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

## 15. Enlace a la aplicación desplegada
`[PENDIENTE — pegar aquí la URL de Streamlit Community Cloud tras el despliegue]`

## 16. Capturas
`[PENDIENTE — agregar capturas de pantalla de cada módulo en docs/assets/]`

## 17. Limitaciones
- El dataset es sintético; los efectos (turno nocturno, fin de mes, matriz de polivalencia)
  fueron inyectados deliberadamente para que el análisis tenga señal real que recuperar.
- El modelo de embeddings semántico puede degradarse a TF-IDF+SVD si el entorno de
  despliegue no tiene acceso de red al repositorio de modelos.
- El LLM requiere una API key propia (no incluida) para funcionar en producción.
- El agente usa un router basado en reglas explicables (no tool-use nativo del LLM) para
  mantener el sistema 100% auditable y determinista.

## 18. Trabajo futuro
- Conectar a una base de datos remota (Postgres/Supabase) en vez de CSV para persistencia real.
- Sustituir el router de reglas por tool-use nativo de la API de Anthropic.
- Añadir forecasting supervisado del score diario usando los lags ya calculados.
- Agregar autenticación de usuarios (auditor/gerente) y control de versiones de las IT.

---

## Estructura del repositorio
```
auditorias-ia/
├── app/
│   ├── Home.py
│   ├── common.py
│   └── pages/
├── src/
│   ├── config.py
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── ml/
│   ├── deep_learning/
│   ├── nlp/
│   ├── llm/
│   ├── rag/
│   ├── agents/
│   └── signals/
├── data/                       # se genera en el primer arranque
├── docs/
│   ├── instrucciones_trabajo/  # base documental del RAG
│   ├── reporte_tecnico.pdf
│   └── tabla_trazabilidad.md
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Tabla de trazabilidad
Ver `docs/tabla_trazabilidad.md` para el detalle completo tema del diplomado → módulo →
evidencia, exigido en la sección 9 del documento del proyecto final.

## Despliegue en Streamlit Community Cloud
1. Sube este repositorio a GitHub (sin subir `ANTHROPIC_API_KEY` en texto plano).
2. Entra a https://share.streamlit.io y conecta el repositorio.
3. Archivo principal: `app/Home.py`.
4. En **Settings → Secrets**, agrega `ANTHROPIC_API_KEY` si quieres habilitar el LLM.
5. Deploy. La primera carga generará automáticamente el dataset sintético.
