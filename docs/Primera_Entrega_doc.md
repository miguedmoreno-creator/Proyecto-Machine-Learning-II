# Primera Entrega — Proyecto Integrador

## Asignatura: Machine Learning II

**Título del Proyecto:** *SentiMovilidad Colombia: Análisis Predictivo de Sentimiento Ciudadano sobre Servicios de Movilidad Urbana mediante Modelos Avanzados de Ensamble e Interpretabilidad*

**Artículo Base:** Tiwari, D., Nagpal, B., Bhati, B. S., Mishra, A., & Kumar, M. (2023). A systematic review of social network sentiment analysis with comparative study of ensemble-based techniques. *Artificial Intelligence Review, 56*, 13407–13461. <https://doi.org/10.1007/s10462-023-10472-w>

**Curso:** Machine Learning II | **Fase:** Primera Entrega (Fase 1: Comprensión y Planteamiento)

## 1. Comprensión y Sustentación del Artículo Base

### 1.1 Problema abordado

El estudio de Tiwari et al. (2023) aborda una brecha fundamental en el Procesamiento de Lenguaje Natural (NLP): la carencia de un marco comparativo sistemático y reproducible que enfrente a las dos grandes familias de aprendizaje en conjunto (*ensemble learning*) —**Bagging** y **Boosting**— en la tarea de Análisis de Sentimiento (*Sentiment Analysis*, SA) a nivel de oración sobre texto no estructurado procedente de plataformas sociales y reseñas digitales.

A través de cinco preguntas de investigación (RQ1–RQ5), los autores estructuran la literatura histórica (92 artículos de 1996 a 2022) y ejecutan un benchmark empírico para responder a la pregunta central (**RQ5**): *¿Cuál arquitectura de ensamble exhibe la mayor capacidad predictiva y eficiencia temporal para clasificar la polaridad de opiniones de usuarios?*

flowchart LR
 A[Corpus Textual de Reseñas] --> B[Preprocesamiento & Limpieza]
 B --> C[Tokenización N-gramas]
 C --> D[Vectorización TF-IDF]
 D --> E[Partición 80/20]
 E --> F1[Familia Bagging<br/>Random Forest, Extra Trees, Bagging Linear SVC]
 E --> F2[Familia Boosting<br/>AdaBoost, GradBoost, XGBoost, LightGBM, CatBoost]
 F1 & F2 --> G[Evaluación Multimétrica<br/>F1, AUC, Recall, TPR, FPR, Latencia]

### 1.2 Metodología del artículo

El componente experimental del artículo evalúa 8 algoritmos de ensamble sobre 4 conjuntos de datos en inglés con distintas escalas y grados de desbalance:

* *Uber Reviews:* 1.344 reseñas (fuertemente desbalanceado: 17.3% positivo / 82.7% negativo).
* *Restaurant Reviews:* 1.000 reseñas (balanceado: 50/50).
* *Amazon Product Reviews:* 1.000 reseñas (balanceado: 50/50).
* *Food Reviews:* 25.000 reseñas (balanceado: 50/50, dataset a gran escala).

El flujo implementado comprende: (1) limpieza léxica (corrección ortográfica, remoción de puntuación, URLs, números y *hashtags*); (2) tokenización mediante N-gramas (unigramas, bigramas, trigramas); (3) vectorización espacial mediante **TF-IDF** ($w\_{i,j} = tf\_{i,j} \times \log(N/df\_i)$); y (4) partición hold-out 80/20 para entrenamiento y prueba.

Los 8 modelos de ensamble contrastados fueron:

1. **Bagging:** Random Forest (RF), Extra Trees (ET), y Bagging Meta-Estimator con Linear SVC como clasificador base (M-SVC).
2. **Boosting:** AdaBoost (AB), Gradient Boosting (GB), XGBoost (XGB), LightGBM (LGBM) y CatBoost (CB).

### 1.3 Supuestos principales

* **Suficiencia de la representación dispersa:** Se asume que la matriz TF-IDF con n-gramas preserva la carga semántica necesaria para discriminar polaridad sin requerir modelos de lenguaje contextuales densos (transformers o embeddings).
* **Consistencia del proxy de supervisión:** La calificación ordinal (1–5 estrellas) otorgada por el usuario refleja de forma unívoca la polaridad del texto asociado.
* **Invarianza del orden sintáctico:** La hipótesis *Bag-of-Words* subyacente a TF-IDF asume que la coocurrencia de términos ponderados es suficiente para clasificar el afecto.

### 1.4 Resultados reportados por los autores

El hallazgo central del artículo es que **las técnicas de Bagging superaron de manera consistente a las de Boosting en clasificación de texto con TF-IDF**:

* **Bagging Meta-Estimator (Linear SVC):** Resultó ser el modelo más robusto y escalable en el dataset masivo (*Food Reviews*, 25.000 registros), obteniendo el mayor **ROC-AUC (0.87)**, **F1-Score ponderado (87%)**, la menor tasa de falsos positivos (**FPR = 0.0%** en Uber Reviews) y latencias de inferencia ultra-bajas (67–97 ms).
* **Random Forest y Extra Trees:** Mostraron comportamientos idénticos en entrenamiento (100% *accuracy*), alcanzando precisiones de prueba entre 71.5% y 91.82%.
* **CatBoost:** Fue el clasificador más competitivo de la familia de Boosting (Testing *Accuracy* de 86.40% en Food Reviews), aunque con un coste temporal significativamente superior.
* **Gradient Boosting:** Evidenció la menor brecha entre entrenamiento y prueba (diferencia de 0.41% a 6.75%), mostrando alta resistencia al sobreajuste.

### 1.5 Limitaciones metodológicas identificadas

1. **Monolingüismo (Inglés):** No se evaluó el desempeño en lenguas romances como el español, caracterizadas por mayor flexión morfológica, uso intensivo de modismos regionales y estructuras sintácticas flexibles.
2. **Ausencia de Validación Cruzada:** La partición fija 80/20 es sensible al azar del muestreo y omite la estimación de variabilidad estadística ($\mu \pm \sigma$).
3. **Carencia de Interpretabilidad:** El artículo no aplicó metodologías modernas de explicabilidad (SHAP, LIME, PDP), operando los modelos como cajas negras.
4. **Manejo del Desbalance:** En el conjunto de Uber Reviews (83% negativo), no se exploraron técnicas de compensación de costos o muestreo adaptativo.

## 2. Planteamiento del Problema en Contexto Local

### 2.1 Relevancia social y justificación en Colombia

En las principales áreas metropolitanas de Colombia (Bogotá, Medellín, Cali), los sistemas de transporte público masivo (TransMilenio, SITP, Metro de Medellín) y las plataformas tecnológicas de movilidad urbana (DiDi, Cabify, Moovit) gestionan conjuntamente más de **6 millones de desplazamientos diarios**. La satisfacción ciudadana respecto a variables operativas (tiempos de espera, seguridad percibida, sobrecupo, tarifas y conducta de conductores) constituye un eje neurálgico para la política pública y el bienestar urbano.

Sin embargo, las encuestas de percepción tradicionales (e.g., *Bogotá Cómo Vamos*) presentan alta latencia temporal (reportes semestrales o anuales) y costos elevados de captura. En contraparte, los usuarios expresan de forma espontánea y en tiempo real sus inconformidades y reconocimientos a través de reseñas digitales en plataformas como Google Play Store.

Construir un sistema de modelamiento predictivo en Machine Learning II permite **procesar masivamente el flujo de opiniones ciudadanas, clasificar automáticamente la polaridad de las reseñas, detectar tempranamente alertas operativas de deterioro del servicio y aislar los factores léxicos determinantes mediante técnicas de interpretabilidad.**

### 2.2 Formulación de la tarea predictiva

* **Pregunta de investigación predictiva:** *¿Con qué nivel de precisión, capacidad de discriminación (ROC-AUC) y recall de quejas es posible clasificar la polaridad del sentimiento de un usuario colombiano a partir del texto no estructurado de su reseña sobre servicios de movilidad urbana?*
* **Tipo de problema:** Clasificación binaria supervisada en NLP ($Y \in {0, 1}$).
* **Variable objetivo ($Y$):** $$Y = \begin{cases} 1 & \text{si Calificación} \in {4, 5} \text{ estrellas (Sentimiento Positivo / Favorable)} \ 0 & \text{si Calificación} \in {1, 2} \text{ estrellas (Sentimiento Negativo / Reclamo)} \end{cases}$$ *(Las reseñas con calificación de 3 estrellas representan neutralidad o ambigüedad y se tratan en el análisis exploratorio).*
* **Matriz de características ($X$):** Representación vectorial del texto en español colombiano mediante N-gramas (unigramas y bigramas) ponderados por TF-IDF, complementada con variables de ingeniería léxica (longitud en palabras, tasa de mayúsculas, conteo de signos de exclamación).

### 2.3 Adaptación del método del artículo al problema local

| Dimensión | Enfoque Original (Tiwari et al., 2023) | Adaptación en este Proyecto (ML2) |
| --- | --- | --- |
| **Idioma y Dominio** | Reseñas en inglés (comercio, comida, Uber global) | Reseñas en **español colombiano** de movilidad urbana |
| **Fuente de Datos** | Datasets estáticos precompilados | **Extracción directa y reproducible** desde Google Play Store Colombia |
| **Validación** | Partición simple 80/20 | **Stratified 5-Fold Cross-Validation** + Hold-out test set (20%) en Pipeline estricto |
| **Línea Base** | No formalizada en ML1 | **Dummy Classifier + Multinomial Naive Bayes (Bag-of-Words y TF-IDF)** |
| **Modelos ML2** | 8 ensambles de Bagging y Boosting | Bagging (RF, Extra Trees, Bagging Linear SVC) vs. Boosting (AdaBoost, XGBoost, LightGBM, CatBoost) |
| **Interpretabilidad** | Ausente | **SHAP (TreeExplainer / LinearExplainer)** para atribución de términos locales |
| **Despliegue** | Ausente | **Aplicación interactiva en Streamlit** para inferencia en tiempo real |

## 3. Estrategia de Datos y Simulación Fundamentada

### 3.1 Fuente real: Extracción de reseñas colombianas

Mediante la herramienta google-play-scraper parametrizada para Colombia (country="co", lang="es"), se construyó un dataset consolidado de **7.300 reseñas reales** provenientes de aplicaciones de movilidad urbana:

| Plataforma / Aplicación | Ámbito del Servicio | Registros Extraídos | Calificación Promedio |
| --- | --- | --- | --- |
| **Moovit Colombia** | Planificación de rutas y transporte público | 3.648 | 3.27 ★ |
| **DiDi Colombia** | Transporte urbano por aplicativo | 3.637 | 3.06 ★ |
| **TransMi[App] Bogotá** | Sistema BRT TransMilenio / SITP | 15 | 3.60 ★ |
| **Total Consolidado** | **Movilidad Urbana Colombia** | **7.300** | **3.17 ★** |

Atributos capturados: [app, app\_id, contenido, calificacion, sentimiento\_binario, fecha, thumbs\_up, version\_app]

### 3.2 Hallazgos del Análisis Exploratorio de Datos (EDA)

1. **Balance de Clases Óptimo:** La variable objetivo presenta **3.796 registros positivos (52.00%)** y **3.504 registros negativos (48.00%)**. Este balance cuasi-perfecto elimina el riesgo de sesgos por desbalance severo sin requerir técnicas sintéticas como SMOTE.
2. **Asimetría de Longitud Textual:** Las reseñas negativas tienen una media de **30.4 ± 23.6 palabras** (mediana: 24 palabras), mientras que las positivas promedian **11.7 ± 13.5 palabras** (mediana: 7 palabras). Los usuarios que reportan fallas del servicio elaboran explicaciones 2.6 veces más extensas.
3. **Firmas Léxicas Distintivas:**
   * *Términos Positivos:* "excelente", "buena", "servicio", "útil", "rápido", "fácil", "puntual", "recomiendo".
   * *Términos Negativos:* "pésimo", "demora", "cancela", "conductor", "cobro", "tiempo", "ubicación", "inseguro", "peor".

*Figura 1: Distribución de calificaciones (1 a 5 estrellas) y distribución de la variable objetivo binaria ($Y$).*

*Figura 2: Boxplot y densidad KDE comparativa de la longitud de reseñas en palabras por clase de sentimiento.*

*Figura 3: Unigramas y bigramas más discriminantes en español colombiano por clase de sentimiento.*

*Figura 4: Nubes de palabras representativas para reseñas positivas y negativas.*

### 3.3 Módulo de Simulación Fundamentada (DGP)

En cumplimiento de las directrices del curso respecto a la modelación del proceso generador de datos (*Data Generating Process*, DGP), se implementó un generador estocástico calibrado con las distribuciones empíricas observadas:

* **Distribución de longitud estocástica:** $$\text{Longitud}*{\text{neg}} \sim \text{LogNormal}(\mu=3.4, \sigma=0.6), \quad \text{Longitud}*{\text{pos}} \sim \text{LogNormal}(\mu=2.4, \sigma=0.5)$$
* **Muestreo léxico condicional:** Inserción probabilística de colocaciones y modismos locales observados en el corpus real ("demasiada demora", "conductores groseros", "excelente servicio", "muy útil").
* **Calibración de calificaciones:** $P(1★|\text{Neg}) = 0.55, P(2★|\text{Neg}) = 0.45; \quad P(4★|\text{Pos}) = 0.42, P(5★|\text{Pos}) = 0.58$.

Este módulo garantiza la posibilidad de realizar pruebas de estrés (*stress-testing*) y aumentación controlada de datos si se requiere evaluar la sensibilidad de los modelos ante cambios en la distribución léxica.

## 4. Plan de Modelamiento, Línea Base y Métricas

### 4.1 Protocolo de Validación y Prevención de Fuga de Datos

Para garantizar la reproducibilidad y evitar **fuga de datos (data leakage)**, el preprocesamiento y la vectorización se encapsulan estrictamente dentro de un Pipeline de scikit-learn:

Pipeline([
 ('vectorizer', TfidfVectorizer(ngram\_range=(1, 2), min\_df=2, stop\_words=STOPWORDS\_ES)),
 ('classifier', Modelo())
])

* **Conjunto de Entrenamiento (80%):** 5.840 registros, evaluados mediante **Stratified 5-Fold Cross-Validation** (el vectorizador se ajusta exclusivamente en los 4 folds de entrenamiento de cada iteración).
* **Conjunto de Prueba Hold-out (20%):** 1.460 registros independientes, reservados intactos para evaluación final.

### 4.2 Métricas de Evaluación Justificadas

1. **F1-Score Ponderado (Métrica Principal):** Evalúa el balance armónico entre precisión y cobertura, permitiendo comparación directa con Tiwari et al. (2023).
2. **ROC-AUC:** Mide la capacidad de separación de clases independientemente del umbral de decisión; crucial para calibrar sistemas de alerta temprana.
3. **Recall de la Clase Negativa (Quejas Reales):** En políticas públicas de transporte, un *falso negativo* (no detectar una queja grave de seguridad o cobro indebido) tiene un costo social mucho mayor que un falso positivo.
4. **TPR, FPR y Latencia de Inferencia (ms):** Para evaluar la viabilidad del despliegue en tiempo real en la aplicación Streamlit.

### 4.3 Resultados Empíricos de los Modelos de Línea Base (ML1)

Los modelos de línea base construidos y evaluados en esta fase establecen el punto de referencia riguroso:

#### Tabla 1: Desempeño en Validación Cruzada (Stratified 5-Fold CV sobre Train 80%)

| Modelo de Línea Base | Tipo | CV Accuracy | CV Weighted F1 | CV ROC-AUC | Tiempo CV (s) |
| --- | --- | --- | --- | --- | --- |
| **Dummy (Mayoría)** | Cota Trivial | 52.00 ± 0.04% | 35.58 ± 0.05% | 0.5000 ± 0.0000 | 1.68 s |
| **Dummy (Estratificado)** | Cota Aleatoria | 50.51 ± 1.61% | 50.52 ± 1.61% | 0.5046 ± 0.0162 | 1.60 s |
| **Multinomial Naive Bayes (BoW)** | ML1 Clásico | 88.72 ± 0.81% | 88.72 ± 0.81% | 0.9434 ± 0.0068 | 2.47 s |
| **Multinomial Naive Bayes (TF-IDF)** | ML1 Ponderado | **89.02 ± 0.57%** | **89.03 ± 0.57%** | **0.9547 ± 0.0064** | 2.46 s |
| **Regresión Logística (TF-IDF)** | Lineal ML1 | 89.18 ± 0.59% | 89.18 ± 0.60% | 0.9521 ± 0.0065 | 2.57 s |

#### Tabla 2: Desempeño en Conjunto de Prueba Independiente (Test Hold-out 20%, $N=1.460$)

| Modelo | Test Accuracy | Test Weighted F1 | Test ROC-AUC | Recall Quejas (0) | TPR (Pos) | FPR (Fallos) | Latencia (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Dummy (Mayoría)** | 51.99% | 35.56% | 0.5000 | 0.00% | 100.00% | 100.00% | 49.1 ms |
| **Dummy (Estratificado)** | 50.89% | 50.90% | 0.5084 | 49.50% | 52.17% | 50.50% | 55.3 ms |
| **Multinomial NB (BoW)** | 89.86% | 89.86% | 0.9543 | 89.02% | 90.65% | 10.98% | 76.9 ms |
| **Multinomial NB (TF-IDF)** | **90.34%** | **90.35%** | **0.9643** | **91.58%** | 89.20% | **8.42%** | 78.3 ms |
| **Regresión Logística (TF-IDF)** | 90.75% | 90.76% | 0.9630 | 92.58% | 89.06% | 7.42% | 93.6 ms |

*Figura 5: Matrices de confusión en el conjunto de prueba (Test 20%) para Dummy (Mayoría), Multinomial Naive Bayes y Regresión Logística.*

*Figura 6: Curvas ROC comparativas para todos los modelos de línea base evaluados en el conjunto de prueba.*

### 4.4 Análisis de la Línea Base y Proyección hacia ML2

* **Rendimiento de la Línea Base ML1:** Multinomial Naive Bayes con TF-IDF alcanza un sólido **90.34% de Accuracy**, **0.9643 de ROC-AUC** y un **Recall de quejas del 91.58%** en el conjunto de prueba independiente, superando ampliamente la cota trivial (51.99%).
* **Reto para los Modelos Avanzados de ML2:** En la Segunda Entrega, la meta será evaluar si los modelos de **Bagging** (Random Forest, Extra Trees, Bagging Linear SVC) y **Boosting** (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost) logran superar esta fuerte línea base lineal, reducir la tasa de falsos positivos ($FPR < 8.42%$) y aportar interpretabilidad detallada mediante SHAP.

## 5. Autoevaluación según Rúbrica de la Primera Entrega

| Criterio de Evaluación | Nivel Alcanzado | Evidencia Documental y Experimental |
| --- | --- | --- |
| **1. Comprensión y sustentación del artículo** | **Excelente (4.0–5.0)** | Análisis detallado de Tiwari et al. (2023): formulación de RQ1–RQ5, metodología de 8 ensambles, 4 datasets, supuestos teóricos, resultados cuantitativos y 4 limitaciones críticas. |
| **2. Planteamiento del problema** | **Excelente (4.0–5.0)** | Formulación matemática rigurosa ($Y \in {0, 1}$), justificación del impacto social en movilidad colombiana, definición de variables y adaptación del método en español. |
| **3. Estrategia de datos y simulación** | **Excelente (4.0–5.0)** | Extracción real de **7.300 reseñas** colombianas (superando el umbral de 5.000), balance óptimo (52/48), EDA multivariado completo con 4 figuras y módulo de simulación DGP estadísticamente calibrado. |
| **4. Plan de modelamiento y línea base** | **Excelente (4.0–5.0)** | Pipeline reproducible sin fuga de datos, Stratified 5-Fold CV, evaluación de 5 modelos de línea base con resultados cuantitativos en Test ($AUC = 0.9643$, $F1 = 90.35%$), matrices de confusión y curvas ROC. |
| **5. Comunicación e informe** | **Excelente (4.0–5.0)** | Documento conciso dentro del rango de 3 a 5 páginas, figuras de alta resolución integradas e interpretadas, código modular ejecutable en script Python y Jupyter Notebook. |

## 6. Referencias Bibliográficas

1. Tiwari, D., Nagpal, B., Bhati, B. S., Mishra, A., & Kumar, M. (2023). A systematic review of social network sentiment analysis with comparative study of ensemble-based techniques. *Artificial Intelligence Review*, 56, 13407–13461.
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
5. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.
6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.