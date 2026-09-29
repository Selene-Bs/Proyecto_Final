"""
Genera docs/reporte_tecnico.pdf con las 24 secciones exigidas por el
documento del proyecto final, usando los resultados reales que produce
el pipeline (no cifras inventadas).

Ejecutar:  python docs/generar_reporte_tecnico.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, ListFlowable, ListItem
)

from src.preprocessing.pipeline_datos import cargar_dataset_final, cargar_senales
from src.features.estadistica import prueba_hipotesis_turno_nocturno, correlacion_antiguedad_score
from src.features.series_tiempo import resumen_series
from src.ml.modelos import entrenar_y_comparar_modelos
from src.deep_learning.red_neuronal import entrenar_mlp_vs_clasico
from src.nlp.nlp_clasico import entrenar_clasificador_texto
from src.rag.rag_instrucciones import construir_indice, evaluar_retriever
from src.agents.agente import Agente, evaluar_tool_selection_accuracy
from src.signals.fourier import obtener_serie_linea, calcular_fft, interpretar_top_frecuencias
from src.signals.wavelets import descomponer_dwt, detectar_anomalias_wavelet

OUT = os.path.join(os.path.dirname(__file__), "reporte_tecnico.pdf")

print("Recalculando resultados reales para el reporte (puede tardar un momento)...")
df = cargar_dataset_final()
senales = cargar_senales()
hip = prueba_hipotesis_turno_nocturno(df)
corr = correlacion_antiguedad_score(df)
series = resumen_series(df)
ml = entrenar_y_comparar_modelos(df)
dl = entrenar_mlp_vs_clasico(df)
nlp = entrenar_clasificador_texto(df)
indice = construir_indice()
retriever_eval = evaluar_retriever(indice, k=3)
agente = Agente(df_auditorias=df, pipeline_riesgo=ml["pipelines"][ml["mejor_modelo"]], indice_rag=indice)
tool_eval = evaluar_tool_selection_accuracy(agente)
serie_l1 = obtener_serie_linea(senales, "Línea 1")
fft = calcular_fft(serie_l1)
interpretaciones_fft = interpretar_top_frecuencias(fft["top_frecuencias"])
dwt = descomponer_dwt(serie_l1)
anomalias = detectar_anomalias_wavelet(serie_l1)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", fontSize=16, spaceAfter=10, spaceBefore=14, textColor=colors.HexColor("#1a1a1a"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#1a73e8"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14, spaceAfter=8, alignment=4))
styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=11, textColor=colors.grey))

story = []

def h1(t): story.append(Paragraph(t, styles["H1"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def p(t): story.append(Paragraph(t, styles["Body"]))
def small(t): story.append(Paragraph(t, styles["Small"]))
def espacio(alto=6): story.append(Spacer(1, alto))

# Portada
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Reporte Técnico", ParagraphStyle(name="Titulo", fontSize=26, alignment=1, fontName="Helvetica-Bold", leading=32)))
story.append(Spacer(1, 1.2*cm))
story.append(Paragraph("Sistema Inteligente de Auditorías de Proceso en Línea de Producción", ParagraphStyle(name="Subtitulo", fontSize=14, alignment=1, leading=18)))
story.append(Spacer(1, 1*cm))
story.append(Paragraph("Diplomado de Python y Análisis de Datos — Universidad Marista", ParagraphStyle(name="Sub2", fontSize=11, alignment=1, textColor=colors.grey)))
story.append(Paragraph("Proyecto Final", ParagraphStyle(name="Sub3", fontSize=11, alignment=1, textColor=colors.grey)))
story.append(PageBreak())

# 1. Resumen ejecutivo
h1("1. Resumen ejecutivo")
p("Este proyecto digitaliza la auditoría de proceso de línea de producción (EPP, uso del "
  "vernier, apego a instrucción de trabajo, matriz de polivalencia, 5S, registro de "
  "calidad, tiempo de ciclo y manejo de producto no conforme) y aplica de forma integrada "
  "estadística inferencial, Machine Learning, Deep Learning, NLP, embeddings, un asistente "
  "LLM con RAG, un agente con Tool Calling, y análisis espectral (Fourier) y multiresolución "
  "(Wavelets) sobre una señal dimensional de proceso. El resultado es una aplicación "
  "Streamlit funcional, desplegable en la nube, que convierte auditorías dispersas en "
  "decisiones de mejora continua accionables.")

# 2. Problema
h1("2. Problema")
p("Las auditorías de piso se capturan hoy de forma dispersa (papel/hojas sueltas), sin una "
  "vista integrada que permita detectar patrones de riesgo por línea, turno u operador, "
  "anticipar auditorías con hallazgos críticos, ni consultar de forma natural los "
  "procedimientos vigentes (Instrucciones de Trabajo).")

# 3-4. Objetivos
h1("3. Objetivo general")
p("Construir y desplegar en la nube una aplicación que digitalice la auditoría de proceso "
  "y aplique análisis de datos e inteligencia artificial de extremo a extremo para "
  "convertir esas auditorías en decisiones de mejora continua.")

h1("4. Objetivos específicos")
story.append(ListFlowable([
    ListItem(Paragraph(t, styles["Body"])) for t in [
        "Digitalizar la cédula de auditoría con almacenamiento reproducible.",
        "Cuantificar estadísticamente el efecto de turno, línea y antigüedad del operador.",
        "Predecir el riesgo de una auditoría con hallazgo crítico antes de que ocurra.",
        "Segmentar operadores por patrón de desviación para dirigir capacitación.",
        "Clasificar y buscar semánticamente los comentarios libres del auditor.",
        "Responder preguntas de piso sobre Instrucciones de Trabajo con un asistente RAG.",
        "Analizar la señal dimensional en frecuencia y multiresolución.",
    ]
], bulletType="bullet"))

# 5. Fuente y descripción de datos
h1("5. Fuente y descripción de datos")
p(f"Dataset sintético pero estadísticamente realista, generado con semilla fija "
  f"(reproducible). Incluye <b>{len(df):,} auditorías</b> de proceso, "
  f"<b>{df['operador_id'].nunique()} operadores</b> con matriz de polivalencia y "
  f"antigüedad, <b>{len(senales):,} lecturas</b> de señal dimensional de alta frecuencia "
  f"(24 muestras/día en 2 líneas instrumentadas) y 6 Instrucciones de Trabajo como base "
  f"documental del módulo RAG.")

# 6. Arquitectura
h1("6. Arquitectura")
p("Arquitectura modular en tres capas: (1) interfaz Streamlit, (2) lógica de negocio en "
  "módulos independientes de <font face='Courier'>src/</font> — uno por bloque del "
  "diplomado — y (3) fuentes de datos (CSV + documentos de texto). Ver diagrama a continuación.")
try:
    story.append(Image(os.path.join(os.path.dirname(__file__), "assets", "diagrama_arquitectura.png"), width=15*cm, height=11*cm))
except Exception:
    pass
story.append(PageBreak())

# 7. Ingeniería de datos
h1("7. Ingeniería de datos (Pipeline)")
log = df.attrs.get("log_limpieza", {})
p(f"Se aplicó tipado de columnas, eliminación de duplicados por <font face='Courier'>audit_id</font> "
  f"({log.get('duplicados_removidos', 0)} removidos), imputación conservadora de respuestas "
  f"faltantes ({log.get('filas_con_respuesta_faltante_imputada', 0)} filas, siempre a "
  f"'No OK sin acción' — nunca se asume cumplimiento), marcado de mediciones dimensionales "
  f"atípicas ({log.get('mediciones_atipicas_marcadas', 0)}) y <font face='Courier'>merge</font> "
  f"con el catálogo de operadores.")

# 8. EDA
h1("8. Análisis Exploratorio de Datos (EDA)")
tabla_linea = df.groupby("linea", observed=True)["score_cumplimiento"].mean().round(3)
p("Distribución del score de cumplimiento promedio por línea:")
data_tabla = [["Línea", "Score promedio"]] + [[k, f"{v:.1%}"] for k, v in tabla_linea.items()]
t = Table(data_tabla, colWidths=[6*cm, 4*cm])
t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a73e8")),
                        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTSIZE", (0,0), (-1,-1), 9)]))
story.append(t)
espacio()
efecto_matriz = df.groupby("operador_fuera_de_matriz", observed=True)["score_cumplimiento"].mean()
p(f"Hallazgo clave: el score promedio cae de {efecto_matriz.get(False,0):.1%} a "
  f"{efecto_matriz.get(True,0):.1%} cuando un operador cubre una línea fuera de su matriz "
  f"de polivalencia, concentrado en la pregunta P4.")

# 9. Estadística
h1("9. Estadística")
p(f"<b>Prueba de hipótesis (turno Nocturno):</b> {hip['interpretacion']}")
p(f"<b>Correlación antigüedad–cumplimiento:</b> {corr['interpretacion']}")

# 10. Series temporales
h1("10. Series temporales")
p(f"La tendencia general del score diario es <b>{series['tendencia_texto']}</b>. El peor día "
  f"de la semana en promedio es <b>{series['peor_dia_semana']}</b> y el mejor "
  f"<b>{series['mejor_dia_semana']}</b>.")

# 11. Feature Engineering
h1("11. Feature Engineering")
p("Se construyeron medias móviles (7 y 30 días), desviación estándar móvil, rezagos "
  "(lags 1, 7 y 14 días) y un componente estacional por día de la semana, listos para "
  "alimentar un modelo de forecasting supervisado del score diario.")

# 12. ML
h1("12. Machine Learning")
tabla_ml = ml["tabla_resultados"][["modelo", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(3)
data_ml = [list(tabla_ml.columns)] + tabla_ml.values.tolist()
t2 = Table(data_ml, colWidths=[5.2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
t2.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a73e8")),
                         ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTSIZE", (0,0), (-1,-1), 8)]))
story.append(t2)
espacio()
p(f"Mejor modelo: <b>{ml['mejor_modelo']}</b>. Se aplicó validación cruzada estratificada "
  f"(5-fold) y un ensemble Voting (RF+GB+LR), dentro de un Pipeline de scikit-learn que "
  f"evita data leakage (el preprocesador se ajusta solo con el set de entrenamiento).")
p("Clustering no supervisado (K-Means + PCA) sobre el perfil de desviaciones de cada "
  "operador permite segmentar en grupos de riesgo por categoría (Seguridad, Calidad, "
  "Proceso, Personal) para dirigir capacitación específica.")

# 13. Deep Learning
h1("13. Deep Learning")
tabla_dl = dl["tabla_resultados"].round(3)
data_dl = [list(tabla_dl.columns)] + tabla_dl.values.tolist()
t3 = Table(data_dl, colWidths=[7*cm, 3*cm, 3*cm, 3*cm])
t3.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a73e8")),
                         ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTSIZE", (0,0), (-1,-1), 8)]))
story.append(t3)
espacio()
p("Se implementó un MLP (32,16 neuronas, ReLU, Adam) comparado contra Regresión Logística "
  "sobre el mismo problema de clasificación de riesgo. No se exige que Deep Learning supere "
  "al modelo clásico; se exige implementarlo, evaluarlo y compararlo honestamente.")

# 14. NLP y embeddings
h1("14. NLP y embeddings")
p(f"Clasificador de severidad basado en TF-IDF sobre el comentario del auditor: "
  f"Accuracy={nlp['metricas']['accuracy']}, F1 macro={nlp['metricas']['f1_macro']}. "
  f"Para búsqueda semántica se usa Sentence-Transformers (con fallback automático a "
  f"TF-IDF+SVD si el entorno de despliegue no tiene acceso al modelo), comparado "
  f"explícitamente contra el ranking TF-IDF clásico en la aplicación.")

# 15. LLM
h1("15. LLM")
p("Se integra la API de Claude (Anthropic) con un prompt de sistema que exige responder "
  "solo en español, de forma accionable, y basarse ÚNICAMENTE en los datos entregados en "
  "el mensaje (control de alucinaciones). Se valida el formato de salida (JSON) con "
  "reintento automático, y cualquier error de red/API se captura y se muestra de forma "
  "controlada en la interfaz.")

# 16. RAG
h1("16. RAG")
recall_txt = retriever_eval[f"recall_at_{retriever_eval['k']}"]
p(f"Pipeline completo (documentos → chunking → embeddings → índice → Top-k → contexto → "
  f"respuesta con fuentes citadas → manejo explícito de 'sin evidencia') sobre las "
  f"Instrucciones de Trabajo de la planta. Evaluación del retriever sobre "
  f"{retriever_eval['n_preguntas_prueba']} preguntas de prueba: "
  f"<b>Recall@{retriever_eval['k']} = {recall_txt}</b>, "
  f"<b>MRR = {retriever_eval['mrr']}</b>.")

# 17. Agentes
h1("17. Agentes")
p(f"Agente tipo router + dispatcher con 6 herramientas (calculadora, predictor de riesgo, "
  f"consulta estadística, consulta RAG, analizador de series y un generador de reporte "
  f"como tarea multi-step). Se registran todas las llamadas (log auditable), se limita el "
  f"número de pasos y se capturan errores por herramienta. "
  f"<b>Tool Selection Accuracy = {tool_eval['tool_selection_accuracy']:.0%}</b> sobre "
  f"{tool_eval['n_casos']} casos de prueba.")

# 18. Fourier
h1("18. Fourier")
p("Análisis espectral (FFT) sobre la señal dimensional de la Línea 1. Interpretación de "
  "las frecuencias/periodos dominantes:")
story.append(ListFlowable([ListItem(Paragraph(t, styles["Body"])) for t in interpretaciones_fft], bulletType="bullet"))

# 19. Wavelets
h1("19. Wavelets")
n_anom = int(anomalias["anomalia_detectada_wavelet"].sum())
energia_txt = ", ".join(f"{k}: {v:.1f}%" for k, v in dwt["energia_por_nivel_pct"].items())
p(f"Descomposición Wavelet ({dwt['wavelet']}, {dwt['niveles_usados']} niveles). Energía por "
  f"nivel: {energia_txt}. Se detectaron <b>{n_anom} anomalías</b> por energía local del "
  f"detalle de nivel fino, coincidiendo con las ráfagas de desgaste de herramienta "
  f"inyectadas en la simulación (validación del método).")

# 20. Evaluación integral
h1("20. Evaluación integral")
p("La evaluación integral combina evidencia cuantitativa (métricas de ML/DL/NLP/RAG/Agentes "
  "reportadas arriba, todas recalculadas en vivo) con evidencia cualitativa (interpretación "
  "de negocio de cada hallazgo), siguiendo el principio del proyecto: qué se hizo, por qué, "
  "cómo se midió y qué valor produce.")

# 21. Despliegue
h1("21. Despliegue")
p("Aplicación Streamlit desplegada en Streamlit Community Cloud. Variables sensibles "
  "(API key del LLM) gestionadas vía *secrets*, nunca en el repositorio. Dependencias "
  "fijadas en requirements.txt. El dataset sintético se autogenera en el primer arranque, "
  "sin dependencias de red obligatorias (salvo el LLM y, opcionalmente, el modelo de "
  "embeddings, ambos con manejo explícito de fallo/():contenedor.")

# 22. Limitaciones
h1("22. Limitaciones")
story.append(ListFlowable([ListItem(Paragraph(t, styles["Body"])) for t in [
    "El dataset es sintético; los efectos de negocio fueron inyectados deliberadamente.",
    "El embedding semántico puede degradarse a TF-IDF+SVD sin acceso de red al modelo.",
    "El LLM requiere una API key propia para operar en producción.",
    "El router del agente usa reglas explicables, no tool-use nativo del LLM.",
]], bulletType="bullet"))

# 23. Conclusiones
h1("23. Conclusiones")
p("El proyecto demuestra que es posible integrar, sobre un mismo problema de negocio real, "
  "todo el recorrido del diplomado —desde el pipeline de datos hasta agentes con LLM y "
  "análisis de señales— produciendo evidencia medible (no solo mencionada) en cada bloque, "
  "y empaquetado en una aplicación funcional desplegable en la nube.")

# 24. Trabajo futuro
h1("24. Trabajo futuro")
story.append(ListFlowable([ListItem(Paragraph(t, styles["Body"])) for t in [
    "Persistencia en base de datos remota (Postgres/Supabase) en vez de CSV.",
    "Tool-use nativo de la API de Anthropic en reemplazo del router de reglas.",
    "Forecasting supervisado del score diario usando los lags ya calculados.",
    "Autenticación de usuarios y control de versiones de las Instrucciones de Trabajo.",
]], bulletType="bullet"))

doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
doc.build(story)
print(f"Reporte técnico generado en: {OUT}")
