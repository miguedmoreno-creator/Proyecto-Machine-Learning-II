# 🚍 SentiTransMilenio | MLOps & Advanced Predictive Intelligence Suite
## Análisis Predictivo de Sentimiento Ciudadano sobre TransMi[App] mediante Modelos Avanzados de Ensamble e Interpretabilidad SHAP

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E.svg)](https://scikit-learn.org/)
[![Vercel Ready](https://img.shields.io/badge/vercel-deployed-black.svg?logo=vercel)](https://vercel.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boost-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen.svg)](https://shap.readthedocs.io/)

---

### 📌 Información Institucional
* **Asignatura:** Machine Learning II (2026-2)
* **Programa Académico:** Pregrado en Ciencia de Datos
* **Institución:** Universidad Externado de Colombia
* **Docente Titular:** Wilmer Pineda (`wilmer.pineda@uexternado.edu.co`)
* **Fase:** Entrega Final (Fase 3: Consolidación, Despliegue y Comunicación)
* **Artículo de Referencia:** Tiwari, D., Nagpal, B., Bhati, B. S., Mishra, A., & Kumar, M. (2023). A systematic review of social network sentiment analysis with comparative study of ensemble-based techniques. *Artificial Intelligence Review, 56*, 13407–13461. https://doi.org/10.1007/s10462-023-10472-w

---

## 🗺️ Mapa de Entregables Oficiales

| Entregable Oficial | Ubicación en el Repositorio | Descripción / Estado |
|---|---|---|
| **Informe Técnico Final (Fase 3)** | [`docs/Entrega_Final.md`](docs/Entrega_Final.md) | Informe formal estructurado según las directrices de la Sección 8 y la rúbrica (máximo 5 páginas de cuerpo). |
| **Plataforma Web para Vercel** | [`vercel_app/index.html`](vercel_app/index.html) | Aplicación web interactiva client-side (SPA) con motor de inferencia en tiempo real, inspector SHAP y simulación. |
| **Configuración de Vercel** | [`vercel_app/vercel.json`](vercel_app/vercel.json) | Manifiesto de despliegue estático con cabeceras de seguridad y optimización de caché. |
| **Pipeline Experimental Completo** | [`src/run_final_world_class_pipeline.py`](src/run_final_world_class_pipeline.py) | Script autónomo y reproducible: 5-Fold CV, 11 modelos, calibración Brier, optimización de umbral y suite SHAP. |
| **Generador de la Web App** | [`src/build_vercel_presentation.py`](src/build_vercel_presentation.py) | Compilador que embebe el payload de datos y pesos del modelo dentro del HTML estático. |
| **Dataset Oficial Curado** | [`dataset/Datos obtenidos.csv`](dataset/Datos obtenidos.csv) | $N = 6.089$ registros reales extraídos de Google Play Store para `com.nexura.transmilenio`. |
| **Payload Maestro de Datos** | [`dataset/data_payload_master.json`](dataset/data_payload_master.json) | Objeto JSON con tablas de métricas, curvas ROC/PR, puntos de calibración y coeficientes léxicos. |
| **Gráficas de Publicación** | [`graphs/`](graphs/) | Figuras en alta resolución ($300\text{ dpi}$): ROC, PR, Calibración Brier, Costo vs Umbral, SHAP y Estrés DGP. |
| **Entregas Anteriores** | [`docs/Primera_Entrega (1).docx`](docs/Primera_Entrega%20(1).docx) y [`docs/Segunda_Entrega.pdf`](docs/Segunda_Entrega.pdf) | Trazabilidad histórica de las Fases 1 y 2. |

---

## 🔬 Resumen Científico y Hallazgos del Benchmark

Se contrastaron **11 arquitecturas predictivas** divididas en 4 familias:

```
                            +-------------------------------------------------------+
                            |     11 Modelos Evaluados en SentiTransMilenio         |
                            +-------------------------------------------------------+
                               |              |                 |                 |
                   [ Líneas Base ML1 ]   [ Bagging ML2 ]   [ Boosting ML2 ]   [ Stacking ML2 ]
                   - Dummy (Mayoría)     - Random Forest   - AdaBoost         - Stacking
                   - Multinomial NB      - Extra Trees     - Gradient Boost     (SVC + CatB
                   - Regresión Log.      - Bagging Lin.SVC - XGBoost            + RF con Meta-
                                                           - CatBoost           Clasificador)
```

### Resultados Destacados en Conjunto de Prueba Independiente (Test 20%, $N=1.218$):
1. **Confirmación Empírica del Paper:** Revalidando los hallazgos de Tiwari et al. (2023) sobre *Uber Reviews* (desbalance 82.7%), **Bagging Linear SVC** superó a todas las familias en balance operativo:
   * **Weighted F1-Score:** $90.16\%$
   * **ROC-AUC Test:** $0.9288$ (con $0.9299$ en el meta-ensamble Stacking)
   * **Recall de Quejas Ciudadanas (Clase 0):** **$97.45\%$** (solo 25 quejas pasadas por alto de 981)
   * **Calibración de Probabilidades:** Brier Score mínimo de **$0.0712$**
   * **Latencia de Inferencia:** **0.06 ms** por reseña (apto para servir miles de consultas por segundo).
2. **Optimización de Umbral de Costo ($p^*$):**
   * Operar con el umbral ingenuo $p=0.50$ en fuerte desbalance ($80.5\%$ quejas) es un error metodológico.
   * Bajo la función de pérdida asimétrica $C(p) = 20 \cdot FP + 5 \cdot FN$, el umbral óptimo encontrado fue **$p^* = 0.64$**, reduciendo el costo operacional de $940$ a **$800$ unidades** (**$14.9\%$ de ahorro** en Bagging SVC y hasta **$70.9\%$** en CatBoost).
3. **Resiliencia ante Pruebas de Estrés Sintéticas (DGP OOD):**
   * Validando la simulación fundamentada exigida en la rúbrica, el modelo fue sometido a 3 escenarios sintéticos extremos (jerga bogotana, ataques de negación y crisis de desbalance al 95%).
   * Mantuvo un **$100.00\%$ de Recall de quejas** en los tres escenarios.

---

## 🚀 Despliegue en Vercel

La carpeta [`vercel_app/`](vercel_app/) contiene una aplicación web completa y autónoma (SPA) que no requiere servidor backend en ejecución.

### Despliegue en 1 Clic con Vercel CLI:
```bash
cd vercel_app
vercel --prod
```

### Despliegue desde GitHub:
1. Conecta este repositorio en tu cuenta de [Vercel](https://vercel.com/).
2. Configura el **Root Directory** como `vercel_app`.
3. Haz clic en **Deploy**. ¡Listo en menos de 30 segundos!

### Prueba Local Inmediata:
Abre directamente [`vercel_app/index.html`](vercel_app/index.html) en cualquier navegador (Google Chrome, Edge, Safari, Firefox). Funciona **100% offline** gracias a su motor client-side vectorizado.

---

## 🛠️ Reproducibilidad del Pipeline de Entrenamiento

Para reproducir todo el benchmark desde cero y regenerar los gráficos:

```bash
# 1. Instalar dependencias
pip install scikit-learn xgboost catboost shap matplotlib seaborn pandas numpy

# 2. Ejecutar el pipeline de modelamiento
python src/run_final_world_class_pipeline.py

# 3. Compilar la aplicación interactiva para Vercel
python src/build_vercel_presentation.py
```

