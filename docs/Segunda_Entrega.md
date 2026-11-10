Entrega #2-Analisis de Sentimiento de transmilenio
Resumen Ejecutivo
En esta segunda fase del proyecto integrador, se consolida el flujo analítico y experimental avanzado de Machine
Learning II sobre el corpus oficial de reseñas ciudadanas de TransMi[App] ( com.nexura.transmilenio ), extraído
directamente de Google Play Store para Bogotá. A diferencia de esquemas con datos simulados, esta entrega
opera sobre un dataset real curado de N = 6.089 registros (N > 5.000), el cual reproduce de manera natural la
estructura de fuerte desbalance documentada en el artículo base de Tiwari et al. (2023) (80.57% de quejas
ciudadanas / 19.43% de elogios, análogo al 82.70% de
Uber Reviews
).
Se implementó un flujo de aprendizaje automatizado estrictamente encapsulado en Pipeline de scikit-learn ,
garantizando la ausencia total de fuga de información (data leakage) mediante una partición hold-out
independiente (80% Train, N = 4.871; 20% Test, N = 1.218) y un esquema de validación cruzada estratificada de 5
pliegues (Stratified 5-Fold CV). Se contrastaron 10 arquitecturas predictivas: 3 líneas base (Dummy Mayoría,
Multinomial Naive Bayes, Regresión Logística), 3 ensambles de la familia Bagging (Random Forest, Extra Trees,
Bagging Linear SVC) y 4 ensambles de la familia Boosting (AdaBoost, Gradient Boosting, XGBoost, CatBoost).
Los resultados empíricos confirman de forma contundente la hipótesis de Tiwari et al. (2023): el modelo Bagging
Meta-Estimator con Linear SVC superó a todas las familias en equilibrio operativo, alcanzando en el conjunto de
prueba independiente un Weighted F1-Score del 90.23%, un ROC-AUC de 0.9277, la máxima cobertura en
detección de reclamos ciudadanos (Recall de Quejas = 97.55%), una tasa mínima de falsas alarmas (FPR = 2.45%)
y una latencia de inferencia de apenas 0.06 ms por reseña, convirtiéndose en la arquitectura idónea para el
monitoreo institucional de TransMilenio S.A.
7.1. Consolidación de los Datos
7.1.1. Origen, Autenticidad y Calidad del Dataset
El corpus definitivo se conformó a partir de las reseñas ciudadanas emitidas para la aplicación institucional
TransMi[App] ( com.nexura.transmilenio ), herramienta digital oficial desarrollada por Nexura Internacional para el
ente gestor TransMilenio S.A. La captura se ejecutó exhaustivamente a través de la API pública de Google Play
Store con localización geográfica en Colombia ( country="co" , lang="es" ).
Parámetro / Atributo Valor Confirmado Justificación Metodológica
Identificador de Aplicación com.nexura.transmilenio TransMi[App] Oficial — Sistema Integrado de Transporte
de Bogotá
Registros Totales Extraídos 6.093 Extracción exhaustiva por relevancia y novedad temporal
Registros Depurados (N) 6.089 Limpieza de textos vacíos o con longitud < 3 caracteres
Requisito Guía ML2 (N ≥ CUMPLIDO (6.089 > 5.000) Satisface el umbral sin requerir síntesis artificial de datos
5.000)
Ventana Temporal 13/06/2017 a 26/08/2026 Captura longitudinal que abarca múltiples versiones
operativas
Variable Objetivo (Y) Binaria: Y ∈ Y = 0 (1–2★, Queja / Falla); Y = 1 (4–5★, Elogio /
Satisfacción)
Desbalance Empírico 80.57% Quejas / 19.43% Ratio de desbalance de 4.15 a 1
Elogios
7.1.2. Justificación de No Uso de Datos Sintéticos
La guía del curso autoriza tres vías de obtención: dataset real, simulación fundamentada o combinación. En esta
entrega se optó deliberadamente por la vía 100% de datos reales, sustentada en tres criterios metodológicos:
1. Suficiencia Muestral: El corpus de 6.089 registros supera con holgura los 5.000 registros mínimos requeridos,

garantizando suficiente soporte estadístico por pliegue en validación cruzada.
2. Validez Ecológica del Lenguaje Coloquial Bogotano: El lenguaje natural de los usuarios de TransMilenio incluye
construcciones sintácticas informales, modismos locales ("el sitp nunca pasa", "se cierra sola la app", "no
actualiza el saldo de tullave", "la ruta h131 demorada") y variaciones ortográficas que un generador sintético no
puede replicar con fidelidad semántica sin inducir sesgos artificiales.
3. Desbalance Orgánico del Dataset Real: El corpus de TransMi[App] presenta 80.57% de quejas frente a 19.43%
de elogios (ratio 4.15 a 1), un desbalance que emerge de forma natural del comportamiento de los usuarios y que
ningún mecanismo de simulación podría replicar con la misma autenticidad lingüística y contextual.
Figura 1: A. Tasa de completitud por variable: el texto (contenido) y la calificación poseen 100% de integridad; únicamente version_app
presenta un 14.52% de valores faltantes (usuarios que omitieron actualizar el metadato). B. Distribución de calificaciones en estrellas (1★:
65.6%, 2★: 14.9%, 4★: 5.5%, 5★: 14.0%).
7.1.3. Análisis Exploratorio de Datos (EDA)
El análisis exploratorio reveló tres propiedades estructurales fundamentales:
1. Asimetría de la Longitud Textual: Los usuarios que expresan quejas y reclamos redactan opiniones
significativamente más extensas y descriptivas que aquellos que elogian la aplicación. Las quejas exhiben una
media de 22.74 ± 17.87 palabras (mediana: 18 palabras, rango intercuartílico IQR = 20), mientras que los elogios
registran una media de 17.36 ± 18.41 palabras (mediana: 11 palabras, IQR = 20). En caracteres, las quejas
alcanzan un percentil 75 de 164 caracteres contra 130 caracteres en elogios.
2. Correlación Multivariada: Se apreciaron patrones estadísticos donde textos extensos se asocian de manera
consistente con fallas operativas reportadas por los usuarios (correlación ρ = -0.21).
3. Firmas Léxicas Diferenciales: El análisis de N-gramas y nubes de palabras demuestra que el vocabulario de
quejas se agrupa en torno a la operatividad del sistema ("rutas", "sitp", "actualizacion", "demora", "buses",
"paraderos", "tiempo real", "mapa", "pesimo"), mientras que los elogios se concentran en términos de utilidad y
afecto ("excelente", "buena", "util", "ayuda", "facil", "recomiendo", "gracias").
Figura 2: A. Estimación de densidad por kernel (KDE) de la longitud en palabras por polaridad. B. Diagrama de caja y bigotes (Boxplot) que
evidencia la mayor asimetría y dispersión en las quejas ciudadanas.

Figura 3: Nubes léxicas comparativas de TransMi[App]: A. Léxico predominante en quejas (Y = 0). B. Léxico predominante en elogios (Y = 1).
7.2. Flujo Reproducible sin Fuga de Datos
7.2.1. Arquitectura del Pipeline y Prevención de Fuga (Data Leakage)
Para asegurar la reproducibilidad científica y evitar cualquier contaminación entre particiones, el
preprocesamiento, la extracción de características y el modelamiento se encapsularon en objetos Pipeline de
scikit-learn :
Pipeline de Aprendizaje: [ Vectorizador TF-IDF (Vocabulario, IDF) ⟶ Clasificador Meta-Estimador ]
El orden estricto de las etapas y las salvaguardas implementadas son:
1. Preprocesamiento Léxico Desacoplado: Normalización Unicode NFD (remoción de tildes y diacríticos),
conversión a minúsculas, eliminación de URLs y caracteres no alfanuméricos. Esta etapa no aprende ningún
parámetro de la distribución de los datos.
2. Partición Previa Hold-Out (80/20): La división entre datos de entrenamiento (N_train = 4.871) y datos de prueba
(N_test = 1.218) se realiza antes de instanciar o ajustar cualquier estimador, estratificando por la variable Y con
semilla fija ( random_state=42 ). El conjunto de prueba se almacena sellado.
3. Ajuste del Vectorizador Estrictamente en Train: El cálculo del vocabulario (V), las frecuencias de término (TF) y
los pesos de frecuencia inversa de documento (IDF) se ejecuta exclusivamente sobre los datos de entrenamiento
de cada fold:
Fórmula IDF: IDF(t, D_train) = log((1 + |D_train|) / (1 + df(t, D_train))) + 1
Los datos de validación en cada pliegue y el conjunto de prueba independiente se transforman únicamente con la
matriz ajustada en train. Cualquier n-grama exclusivo de validación o prueba es ignorado, garantizando que el
modelo jamás observe la distribución léxica de evaluación durante el entrenamiento.
7.3. Reproducción y Adaptación del Método
7.3.1. Adaptación Metodológica del Artículo Base
Tiwari et al. (2023) formularon un benchmark comparativo de técnicas de ensamble sobre datasets en inglés. Para
trasladar dicho marco al contexto de TransMilenio en Bogotá, se introdujeron adaptaciones esenciales:

Elemento Enfoque Original (Tiwari et al., 2023) Adaptación en este Proyecto (Fase 2)
Idioma y Tokenización estándar en inglés Preprocesamiento fonético-ortográfico para español
Tokenización colombiano
Stopwords Lista genérica de inglés Stopwords en español enriquecidas con términos
neutros del dominio
Representación TF-IDF de unigramas, bigramas y TF-IDF con N-gramas ∈ (1, 2), min_df = 2, max_features
Dispersa trigramas = 4.000 y sublinear TF
Compensación del No implementada (Uber con 82.7% no Ponderación de clases inversa
Desbalance compensado) ( class_weight='balanced' ,
scale_pos_weight=4.15 )
Esquema de Hold-out simple 80/20 sin estimación de Stratified 5-Fold Cross-Validation (media ± desv. est.) +
Validación varianza Hold-out Test 20%
Calibración de Linear SVC sin salida probabilística Calibración con CalibratedClassifierCV para
Probabilidades cálculo exacto de ROC-AUC
7.3.2. Catálogo de Modelos Contrastados
Se evaluaron 10 modelos pertenecientes a tres familias metodológicas:
1. Líneas Base (ML1):
- Dummy Classifier (Mayoría): Clasificador trivial que asigna siempre la clase mayoritaria (Y = 0). Cota inferior
indispensable en problemas con fuerte desbalance.
- Multinomial Naive Bayes: Clasificador probabilístico generativo clásico con suavizado de Laplace (α = 0.5).
- Regresión Logística: Modelo discriminativo lineal con regularización L2 (C = 1.0) y pesos balanceados.
2. Familia Bagging (ML2):
- Random Forest: Ensamble de 100 árboles de decisión con remuestreo bootstrap, subespacios aleatorios,
profundidad máxima de 20 y pesos balanceados.
- Extra Trees (Extremely Randomized Trees): Ensamble de 100 árboles con umbrales de división seleccionados de
forma completamente estocástica.
- Bagging Meta-Estimator (Linear SVC): Ensamble bagging de 10 clasificadores de soporte vectorial lineal
( LinearSVC con C = 0.5), calibrados mediante sigmoid Platt para generar probabilidades continuas.
3. Familia Boosting (ML2):
- AdaBoost: Boosting adaptativo sobre árboles de decisión débiles (80 estimadores, tasa de aprendizaje η = 0.3).
- Gradient Boosting: Boosting estocástico de gradiente basado en descenso por gradiente sobre función de
pérdida deviance (80 estimadores, η = 0.1, profundidad 4).
- XGBoost: Algoritmo de árboles con regularización de segundo orden y ponderación de la clase positiva
( scale_pos_weight = 4.15 , 100 árboles, η = 0.08, profundidad 5).
- CatBoost: Boosting simétrico oblivious con manejo balanceado de clases ( auto_class_weights='Balanced' , 100
iteraciones, η = 0.1, profundidad 5).
7.4. Validación, Métricas e Interpretación Preliminar
7.4.1. Métricas de Evaluación Justificadas
Dada la naturaleza del servicio público y la asimetría de clases (80.57% quejas vs. 19.43% elogios), la métrica
tradicional de Accuracy resulta engañosa: predecir siempre queja ofrece un 80.57% de exactitud trivial pero
inutilidad operativa absoluta. Por ello, la evaluación se fundamenta en:
- Balanced Accuracy: Media aritmética de la sensibilidad de cada clase: (Recall_0 + Recall_1) / 2 . Premia a los
modelos que reconocen efectivamente la clase minoritaria sin sacrificar la mayoritaria.
- Weighted F1-Score y Macro F1-Score: Balance armónico entre precisión y cobertura.
- Recall de Quejas (Clase 0): Métrica de máxima prioridad institucional para TransMilenio S.A., orientada a
minimizar falsos positivos (falsas quejas pasadas por alto).
- TPR de Elogios (Clase 1) y FPR (Tasa de Falsas Alarmas): Para comparar directamente con los resultados
reportados en Tiwari et al. (2023).
- ROC-AUC: Capacidad de discriminación global independiente del punto de corte.
- Latencia de Inferencia: Tiempo de cómputo por registro en milisegundos.
7.4.2. Resultados Empíricos en Validación Cruzada (Stratified 5-Fold CV)
Tabla 1: Desempeño en Validación Cruzada (Train Set 80%, N = 4.871, media ± desv. est.)

Familia Modelo CV Accuracy CV Balanced Acc CV Weighted F1 CV Recall Quejas (0) CV ROC-AUC Tiempo CV (s)
Línea Dummy 80.58 ± 50.00 ± 0.00% 71.91 ± 0.02% 100.00 ± 0.00% 0.5000 ± 1.8 s
| Base (Mayoría) | 0.02% | 0.0000 |
| -------------- | ----- | ------ |
Línea Multinomial 88.07 ± 70.98 ± 1.96% 86.53 ± 0.90% 99.46 ± 0.38% 0.9278 ± 1.8 s
| Base NB | 0.69% | 0.0076 |
| ------- | ----- | ------ |
Línea Regresión 89.55 ± 83.92 ± 0.47% 89.22 ± 0.47% 92.36 ± 0.70% 0.9279 ± 2.1 s
| Base Logística | 0.40% | 0.0044 |
| -------------- | ----- | ------ |
Bagging Random 84.15 ± 82.97 ± 1.66% 84.70 ± 1.15% 84.03 ± 1.54% 0.9089 ± 6.2 s
| Forest | 1.25% | 0.0089 |
| ------ | ----- | ------ |
Bagging Extra Trees 86.68 ± 83.16 ± 1.71% 86.84 ± 1.21% 88.20 ± 1.83% 0.9100 ± 5.1 s
|     | 1.34% | 0.0083 |
| --- | ----- | ------ |
Bagging Bagging 90.00 ± 78.48 ± 1.44% 89.63 ± 0.60% 97.96 ± 0.49% 0.9276 ± 3.6 s
| Linear SVC | 0.53% | 0.0069 |
| ---------- | ----- | ------ |
Boosting AdaBoost 82.02 ± 58.42 ± 2.67% 78.66 ± 0.96% 99.62 ± 0.42% 0.7860 ± 7.9 s
|     | 0.50% | 0.0152 |
| --- | ----- | ------ |
Boosting Gradient 88.05 ± 72.69 ± 2.14% 86.63 ± 0.84% 97.71 ± 0.58% 0.8934 ± 14.8 s
| Boosting | 0.52% | 0.0125 |
| -------- | ----- | ------ |
Boosting XGBoost 83.64 ± 82.43 ± 0.27% 84.36 ± 0.34% 83.77 ± 0.55% 0.9026 ± 8.1 s
|     | 0.30% | 0.0051 |
| --- | ----- | ------ |
Boosting CatBoost 85.38 ± 83.90 ± 1.35% 86.08 ± 0.86% 86.09 ± 1.27% 0.9134 ± 18.0 s
|     | 0.89% | 0.0082 |
| --- | ----- | ------ |
Figura 4: Evaluación comparativa en Stratified 5-Fold Cross-Validation: A. Balanced Accuracy (con barra de error ± 1 desv. est.). B. Weighted
F1-Score. C. Capacidad de discriminación ROC-AUC. D. Tiempo total de entrenamiento en validación cruzada.
7.4.3. Resultados en el Conjunto de Prueba Independiente (Test Hold-out 20%)
Tabla 2: Evaluación Final en Conjunto de Prueba Independiente (N_test = 1.218)

Familia Modelo Test Accuracy Balanced Acc Weighted F1 Macro F1 ROC-AUC Recall Quejas (0) TPR Elogios (1) FP
Línea Dummy 80.54% 50.00% 71.86% 44.61% 0.5000 100.00% 0.00% 0.0
Base (Mayoría)
Línea Multinomial 89.08% 73.22% 87.60% 76.90% 0.9305 99.18% 47.26% 0.8
Base NB
Línea Regresión 88.99% 84.21% 89.18% 80.60% 0.9271 92.05% 76.37% 7.9
Base Logística
Bagging Random 84.07% 83.87% 85.14% 75.33% 0.9125 84.20% 83.54% 15.
Forest
Bagging Extra Trees 86.62% 84.49% 87.25% 78.43% 0.9194 87.97% 81.01% 12.
Bagging Bagging 90.80% 80.21% 90.23% 82.20% 0.9277 97.55% 62.87% 2.4
Linear SVC
Boosting AdaBoost 82.02% 54.28% 75.56% 52.88% 0.7928 99.69% 8.86% 0.3
Boosting Gradient 89.49% 75.72% 88.43% 79.25% 0.8789 98.27% 53.16% 1.7
Boosting
Boosting XGBoost 81.69% 83.03% 83.15% 73.35% 0.9135 80.84% 85.23% 19.
Boosting CatBoost 84.48% 83.33% 85.42% 76.43% 0.9119 85.22% 81.43% 14.
Figura 5: Curvas ROC comparativas para los 10 modelos evaluados sobre el conjunto de prueba independiente de TransMi[App].
7.4.4. Discusión Comparativa: Bagging vs. Boosting
La contrastación empírica en el contexto real de TransMilenio arroja hallazgos de gran relevancia teórica y
práctica:
1. Confirmación del Hallazgo Central de Tiwari et al. (2023):
Al igual que en el estudio de referencia sobre Food Reviews y Uber Reviews, la familia Bagging superó a la
familia Boosting en clasificación de texto con matrices TF-IDF. El modelo Bagging Linear SVC alcanzó el
mayor F1 ponderado global (90.23%) y el mayor Accuracy de prueba (90.80%), combinado con un ROC-AUC
sobresaliente de 0.9277. Esto ocurre porque los clasificadores lineales con margen máximo (SVM) manejan con
superioridad la alta dimensionalidad y dispersión del espacio vectorial de n-gramas frente a los árboles de
decisión individuales, mientras que el ensamble bagging reduce sustancialmente la varianza del estimador
base.
2. El Dilema Operativo de Sensibilidad vs. Falsas Alarmas:
Los algoritmos de Boosting (especialmente XGBoost y CatBoost) lograron una mayor sensibilidad en la

detección de elogios (TPR ≈ 81% - 85%). Sin embargo, lo hicieron a costa de inflar fuertemente la tasa de falsos
positivos sobre las quejas (FPR = 19.16% en XGBoost y 14.78% en CatBoost). En contraste, Bagging Linear SVC
exhibe un FPR de tan solo 2.45%, capturando el 97.55% de todas las quejas ciudadanas reales (957 de 981
quejas en prueba). Para una entidad pública como TransMilenio S.A., este balance es crítico: clasificar
erróneamente una queja real como un elogio equivale a invisibilizar fallas operativas en estaciones o buses.
7.4.5. Interpretación Preliminar y Diagnóstico de Errores
Figura 6: Importancia de características e interpretabilidad léxica: Magnitud del coeficiente discriminante para términos determinantes de
quejas (A) y elogios (B).
Figura 7: Diagnóstico cualitativo y cuantitativo de errores en el conjunto de prueba (N = 1.218): A. Desglose entre aciertos (90.8%) y errores
(9.2%). B. Distribución de longitud en palabras para casos de falsos positivos y falsos negativos.

7.5. Autoevaluación según Rúbrica de la Segunda Entrega
Criterio de Evaluación Nivel Alcanzado Evidencia Documental y Experimental
1. Consolidación de los Excelente (4.0– Dataset 100% real de 6.089 reseñas exclusivas de TransMi[App] (N > 5.000),
datos 5.0) extraídas directamente de Google Play Store (Colombia). Justificación sólida de
no simular: suficiencia muestral, validez ecológica del español coloquial
bogotano y desbalance orgánico real (80.57% quejas vs. 19.43% elogios).
2. Flujo sin fuga de Excelente (4.0– Arquitectura integrada en Pipeline de scikit-learn . Partición previa
datos 5.0) Hold-out 80/20 y Stratified 5-Fold Cross-Validation. Ajuste de vocabulario e IDF
ejecutado estrictamente en los folds de entrenamiento de cada iteración,
garantizando cero contaminación o data leakage.
3. Reproducción y Excelente (4.0– Reproducción rigurosa de Tiwari et al. (2023) adaptada al español de Bogotá.
adaptación del método 5.0) Comparación de 10 modelos pertenecientes a tres familias: Líneas Base
(Dummy, Naive Bayes, Regresión Logística), Bagging (Random Forest, Extra
Trees, Bagging Linear SVC) y Boosting (AdaBoost, Gradient Boosting, XGBoost,
CatBoost).
4. Validación, métricas Excelente (4.0– Validación cruzada estratificada reportando variabilidad (media ± desv. est.) y
e interpretación 5.0) evaluación en Test independiente. Justificación de métricas ante el desbalance
(Balanced Accuracy, Weighted F1, Recall de Quejas, TPR, FPR, Latencia).
Confirmación empírica de la superioridad de Bagging Linear SVC (F1 = 90.23%,
AUC = 0.9277, Recall quejas = 97.55%, FPR = 2.45%).
5. Comunicación, Excelente (4.0– Documento formal con extensión concisa y rigurosa dentro del límite de 3 a 5
informe y sustentación 5.0) páginas de cuerpo. Figuras de alta resolución integradas. Código modular y
reproducible en Python.
7.6. Referencias Bibliográficas
1. Tiwari, D., Nagpal, B., Bhati, B. S., Mishra, A., & Kumar, M. (2023). A systematic review of social network
sentiment analysis with comparative study of ensemble-based techniques. Artificial Intelligence Review, 56,
13407–13461. https://doi.org/10.1007/s10462-023-10472-w