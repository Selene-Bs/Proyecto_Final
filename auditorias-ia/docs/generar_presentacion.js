const pptxgen = require("pptxgenjs");

const AZUL = "1A73E8";
const VERDE = "188038";
const GRIS_OSCURO = "202124";
const GRIS = "5F6368";
const ROJO = "C5221F";
const FONDO = "FFFFFF";

function nuevaPres() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in
  return pres;
}

function slideBase(pres) {
  const s = pres.addSlide();
  s.background = { color: FONDO };
  return s;
}

function titulo(s, texto, kicker) {
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.6, y: 0.4, w: 12, h: 0.4, fontSize: 12, color: AZUL,
      bold: true, charSpacing: 1, isTextBox: true, margin: 0,
    });
  }
  s.addText(texto, {
    x: 0.6, y: kicker ? 0.75 : 0.5, w: 12, h: 0.9, fontSize: 30, bold: true,
    color: GRIS_OSCURO, isTextBox: true, margin: 0,
  });
}

function bullets(s, items, opts = {}) {
  const arr = items.map((t, i) => ({
    text: t,
    options: { bullet: { code: "2022", indent: 18 }, breakLine: i < items.length - 1, color: GRIS_OSCURO },
  }));
  s.addText(arr, Object.assign({
    x: 0.7, y: 1.9, w: 11.8, h: 4.8, fontSize: 16, valign: "top",
    paraSpaceAfter: 10, isTextBox: true, margin: 0,
  }, opts));
}

const pres = nuevaPres();

// ---------------------------------------------------------------- Slide 1
{
  const s = slideBase(pres);
  s.addText("🏭", { x: 5.9, y: 1.5, w: 1.5, h: 1.2, fontSize: 54, align: "center", isTextBox: true, margin: 0 });
  s.addText("Sistema Inteligente de\nAuditorías de Proceso", {
    x: 1.5, y: 2.7, w: 10.3, h: 1.6, fontSize: 34, bold: true, align: "center",
    color: GRIS_OSCURO, isTextBox: true, margin: 0,
  });
  s.addText("Diplomado de Python y Análisis de Datos — Universidad Marista  |  Proyecto Final", {
    x: 1.5, y: 4.3, w: 10.3, h: 0.5, fontSize: 15, align: "center", color: GRIS, isTextBox: true, margin: 0,
  });
}

// ---------------------------------------------------------------- Slide 2: Problema real
{
  const s = slideBase(pres);
  titulo(s, "El problema real en piso de producción", "Problema");
  bullets(s, [
    "Las auditorías de proceso (EPP, uso del vernier, instrucción de trabajo, matriz de polivalencia, 5S, calidad, tiempo de ciclo, producto no conforme) se capturan en papel u hojas sueltas.",
    "No existe una vista integrada para detectar patrones de riesgo por línea, turno u operador.",
    "No hay forma de anticipar auditorías con hallazgos críticos antes de que ocurran.",
    "Los procedimientos vigentes (Instrucciones de Trabajo) no son consultables de forma natural desde piso.",
  ]);
}

// ---------------------------------------------------------------- Slide 3: Usuario
{
  const s = slideBase(pres);
  titulo(s, "¿Quién usa el sistema?", "Usuario");
  const cols = [
    ["Auditor de calidad", "Captura la cédula de auditoría en piso y consulta desviaciones históricas."],
    ["Líder de línea / Supervisor", "Prioriza capacitación y acciones correctivas por operador o estación."],
    ["Gerente de planta", "Recibe resúmenes ejecutivos y tendencias para decisiones de mejora continua."],
  ];
  cols.forEach((c, i) => {
    const x = 0.7 + i * 4.05;
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.0, w: 3.75, h: 3.6, rectRadius: 0.12, fill: { color: "F1F5FE" }, line: { color: "FFFFFF" } });
    s.addText(c[0], { x: x + 0.25, y: 2.3, w: 3.25, h: 0.7, fontSize: 17, bold: true, color: AZUL, isTextBox: true, margin: 0 });
    s.addText(c[1], { x: x + 0.25, y: 3.0, w: 3.25, h: 2.3, fontSize: 13.5, color: GRIS_OSCURO, isTextBox: true, margin: 0, valign: "top" });
  });
}

// ---------------------------------------------------------------- Slide 4: Arquitectura
{
  const s = slideBase(pres);
  titulo(s, "Arquitectura modular", "Arquitectura");
  s.addImage({ path: "docs/assets/diagrama_arquitectura.png", x: 2.6, y: 1.7, w: 8.0, h: 5.7 });
}

// ---------------------------------------------------------------- Slide 5: Datos
{
  const s = slideBase(pres);
  titulo(s, "Datos: sintéticos, pero con señal real", "Datos");
  bullets(s, [
    "1,415 auditorías de proceso generadas con semilla fija (100% reproducibles).",
    "40 operadores con matriz de polivalencia y antigüedad simulada.",
    "5,760 lecturas de señal dimensional de alta frecuencia (24 muestras/día, 2 líneas).",
    "6 Instrucciones de Trabajo como base documental del asistente RAG.",
    "Efectos de negocio inyectados a propósito: turno nocturno, fin de mes, operador fuera de su matriz de polivalencia, tendencia de mejora en el tiempo.",
  ]);
}

// ---------------------------------------------------------------- Slide 6: Hallazgos (EDA + Estadística)
{
  const s = slideBase(pres);
  titulo(s, "Hallazgos con EDA y Estadística", "Hallazgos");
  bullets(s, [
    "El turno Nocturno tiene un score de cumplimiento significativamente MENOR al resto (prueba t de Welch, p < 0.05).",
    "Cubrir una línea fuera de la matriz de polivalencia se asocia con una caída marcada del cumplimiento (pregunta P4).",
    "Existe correlación positiva débil entre la antigüedad del operador y su score de cumplimiento.",
    "La tendencia general del cumplimiento diario va MEJORANDO en el periodo analizado.",
  ]);
}

// ---------------------------------------------------------------- Slide 7: Modelo (ML + DL)
{
  const s = slideBase(pres);
  titulo(s, "Modelos: Machine Learning y Deep Learning", "Modelo");
  bullets(s, [
    "Clasificación supervisada: ¿la auditoría tendrá un hallazgo crítico ('No OK sin acción')?",
    "4 modelos comparados + ensemble Voting (RF + Gradient Boosting + Regresión Logística), con validación cruzada 5-fold.",
    "Clustering (K-Means + PCA) segmenta operadores por patrón de desviación para dirigir capacitación.",
    "Red neuronal MLP (Deep Learning) evaluada y comparada honestamente contra el modelo clásico.",
  ]);
}

// ---------------------------------------------------------------- Slide 8: IA generativa (NLP, LLM, RAG, Agentes)
{
  const s = slideBase(pres);
  titulo(s, "Inteligencia artificial generativa", "IA generativa");
  bullets(s, [
    "NLP clásico: clasifica la severidad de la auditoría a partir del comentario del auditor (TF-IDF).",
    "Embeddings: búsqueda semántica de comentarios, comparada contra TF-IDF clásico.",
    "RAG: asistente que responde sobre Instrucciones de Trabajo citando la fuente exacta.",
    "Agente con Tool Calling: enruta la instrucción del usuario entre 6 herramientas, con una tarea multi-step de reporte ejecutivo.",
  ]);
}

// ---------------------------------------------------------------- Slide 9: Señales (Fourier + Wavelets)
{
  const s = slideBase(pres);
  titulo(s, "Series y señales: Fourier y Wavelets", "Señales");
  bullets(s, [
    "FFT sobre la señal dimensional: se identifica un ciclo semanal (~7 días) ligado al ritmo de producción.",
    "Descomposición Wavelet (DWT, db4): energía concentrada en la tendencia, con picos en el detalle fino.",
    "Detección de anomalías por energía Wavelet: coincide con las ráfagas de desgaste de herramienta simuladas.",
    "Denoising por umbralización suave para limpiar la señal antes de analizarla.",
  ]);
}

// ---------------------------------------------------------------- Slide 10: Métricas
{
  const s = slideBase(pres);
  titulo(s, "Resultados medibles, no solo mencionados", "Métricas");
  const metricas = [
    ["Recall@3 del Retriever (RAG)", "1.0"],
    ["MRR del Retriever (RAG)", "1.0"],
    ["Tool Selection Accuracy (Agente)", "100%"],
    ["Accuracy clasificador NLP", "≈0.73"],
    ["Anomalías detectadas (Wavelets)", "Validadas contra simulación"],
  ];
  let y = 1.9;
  metricas.forEach(([nombre, valor]) => {
    s.addText(nombre, { x: 0.7, y, w: 8.0, h: 0.6, fontSize: 15, color: GRIS_OSCURO, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(valor, { x: 9.0, y, w: 3.5, h: 0.6, fontSize: 17, bold: true, color: VERDE, isTextBox: true, margin: 0, valign: "middle" });
    y += 0.85;
  });
}

// ---------------------------------------------------------------- Slide 11: Conclusiones
{
  const s = slideBase(pres);
  titulo(s, "Conclusiones", "Cierre");
  bullets(s, [
    "El proyecto integra TODO el recorrido del diplomado sobre un problema real de manufactura, no como demostraciones aisladas.",
    "Cada técnica está conectada al flujo general, justificada y medida — cumpliendo la regla de integración real del proyecto final.",
    "La aplicación es funcional, reproducible y desplegable en la nube (Streamlit Community Cloud).",
    "Trabajo futuro: base de datos remota, tool-use nativo del LLM, forecasting supervisado del score diario.",
  ]);
}

pres.writeFile({ fileName: "docs/presentacion_final.pptx" }).then(() => {
  console.log("Presentación generada: docs/presentacion_final.pptx");
});
