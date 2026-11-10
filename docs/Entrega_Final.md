# SentiTransMilenio: Plataforma Predictiva de Inteligencia Ciudadana y Monitoreo Operativo en el Sistema de Transporte Masivo de Bogotá

## Entrega Final (Fase 3: Consolidación, Despliegue y Comunicación)
**Integrantes:** Santiago Sandoval Cortés & Miguel David Moreno Lozano  
**Asignatura:** Machine Learning II | **Programa:** Pregrado en Ciencia de Datos  
**Institución:** Universidad Externado de Colombia | **Docente:** Wilmer Pineda  
**Artículo Base:** Tiwari, D., Nagpal, B., Bhati, B. S., Mishra, A., & Kumar, M. (2023). A systematic review of social network sentiment analysis with comparative study of ensemble-based techniques. *Artificial Intelligence Review, 56*, 13407–13461.  
**Despliegue Web Interactivo en Vercel:** [Repositorio & Aplicación Web](https://github.com/miguedmoreno-creator/Proyecto-Machine-Learning-II)  

---

### Resumen Ejecutivo
Este proyecto consolida el flujo integral de modelamiento avanzado para la clasificación supervisada de polaridad afectiva y detección temprana de alertas operativas en **TransMi[App]** (`com.nexura.transmilenio`), canal digital oficial del Sistema de Transporte Masivo de Bogotá (TransMilenio y SITP). A partir de un corpus real y curado de $N = 6.089$ opiniones ciudadanas extraídas de Google Play Store ($N \ge 5.000$), se reprodujo y adaptó al español colombiano el benchmark empírico de Tiwari et al. (2023), contrastando 11 arquitecturas predictivas: 3 líneas base (ML1), 3 ensambles de Bagging, 4 ensambles de Boosting y un meta-ensamble Stacking heterogéneo (ML2). 

El flujo experimental fue encapsulado en `Pipeline` garantizando ausencia total de fuga de datos (*zero data leakage*) mediante partición estratificada Hold-Out (80% Train, $N=4.871$; 20% Test sellado, $N=1.218$) y Stratified 5-Fold Cross-Validation con generación de probabilidades *out-of-fold*. Se implementó una optimización sistemática de umbral de decisión ($p^*$) bajo matriz de costo asimétrica ($C = 20 FP + 5 FN$), suite de interpretabilidad con valores de Shapley (**SHAP**), prueba de estrés fuera de distribución (**OOD**) mediante simulación fundamentada (DGP) y despliegue interactivo en **Vercel** con motor de inferencia y atribución léxica en tiempo real.

---

## 8.1. Modelo Final y Reproducción del Método del Artículo

### 8.1.1. Fundamentación Teórica y Alineación con el Microcurrículo
El artículo base de Tiwari et al. (2023) formuló la pregunta de investigación central (**RQ5**): *¿Cuál arquitectura de ensamble exhibe la mayor capacidad predictiva y eficiencia temporal en análisis de sentimiento con matrices dispersas TF-IDF?* Para responderla en el contexto de Bogotá, se articularon los conceptos axiales del microcurrículo de Machine Learning II:
1. **Teorema del Jurado de Condorcet (Sesión 5):** Justifica la superioridad de los ensambles al promediar clasificadores con probabilidad de acierto individual $p_i > 0.5$ y baja correlación de error.
2. **Bagging y Reducción de Varianza (Sesiones 6 a 8):** Agregación bootstrap sobre clasificadores de margen máximo, minimizando la varianza sin inflar el sesgo.
3. **Boosting y Descenso por Gradiente Funcional (Sesiones 9 a 11):** Optimización secuencial sobre la devianza logística (Gradient Boosting, XGBoost, CatBoost).
4. **Meta-Aprendizaje Stacking (Sesiones 13 y 14):** Combinación jerárquica con meta-clasificador lineal de Nivel 1.
5. **Máquinas de Soporte Vectorial y Calibración de Platt (Sesiones 16 a 18):** Maximización de margen en espacios de 4.000 dimensiones mediante formulación dual y calibración sigmoidea (`CalibratedClassifierCV`).

### 8.1.2. Benchmark Empírico: Validación Cruzada (Train 80%) y Prueba Hold-Out (Test 20%)

#### Tabla 1: Desempeño en Stratified 5-Fold Cross-Validation (Train Set 80%, $N=4.871$)
| Familia | Modelo Evaluado | CV Accuracy | CV Balanced Acc | CV Weighted F1 | CV Recall Quejas (0) | CV ROC-AUC | Tiempo CV |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Línea Base** | Dummy (Mayoría) | $80.58 \pm 0.02\%$ | $50.00 \pm 0.00\%$ | $71.91 \pm 0.02\%$ | $100.00 \pm 0.00\%$ | $0.5000 \pm 0.000$ | 2.0 s |
| **Línea Base** | Multinomial NB | $87.97 \pm 0.61\%$ | $70.65 \pm 1.78\%$ | $86.34 \pm 0.79\%$ | $99.44 \pm 0.35\%$ | $0.9286 \pm 0.007$ | 2.3 s |
| **Línea Base** | Regresión Logística | $89.28 \pm 0.38\%$ | $83.42 \pm 0.67\%$ | $88.83 \pm 0.44\%$ | $92.00 \pm 0.70\%$ | $0.9283 \pm 0.004$ | 2.4 s |
| **Bagging** | Random Forest | $85.65 \pm 0.45\%$ | $82.04 \pm 0.60\%$ | $86.05 \pm 0.52\%$ | $87.54 \pm 0.95\%$ | $0.9005 \pm 0.007$ | 5.2 s |
| **Bagging** | Extra Trees | $86.51 \pm 0.81\%$ | $83.47 \pm 1.03\%$ | $86.84 \pm 0.80\%$ | $87.97 \pm 1.15\%$ | $0.9103 \pm 0.006$ | 4.8 s |
| **Bagging** | **Bagging Linear SVC** | **$89.92 \pm 0.52\%$** | **$78.22 \pm 1.27\%$** | **$89.53 \pm 0.58\%$** | **$97.99 \pm 0.45\%$** | **$0.9280 \pm 0.007$** | **11.8 s** |
| **Boosting** | AdaBoost | $81.56 \pm 0.48\%$ | $57.52 \pm 2.67\%$ | $78.04 \pm 0.90\%$ | $99.72 \pm 0.40\%$ | $0.7840 \pm 0.015$ | 9.7 s |
| **Boosting** | Gradient Boosting | $88.13 \pm 0.50\%$ | $73.02 \pm 1.48\%$ | $86.82 \pm 0.75\%$ | $97.73 \pm 0.55\%$ | $0.8927 \pm 0.011$ | 20.1 s |
| **Boosting** | XGBoost | $83.68 \pm 0.32\%$ | $82.54 \pm 0.54\%$ | $84.42 \pm 0.35\%$ | $83.80 \pm 0.58\%$ | $0.9027 \pm 0.005$ | 10.4 s |
| **Boosting** | CatBoost | $85.34 \pm 0.85\%$ | $83.73 \pm 1.07\%$ | $86.05 \pm 0.82\%$ | $86.17 \pm 1.20\%$ | $0.9130 \pm 0.008$ | 19.8 s |
| **Stacking** | **Stacking (SVC+CatB+RF)** | $88.36 \pm 0.60\%$ | **$85.37 \pm 0.76\%$** | $88.13 \pm 0.55\%$ | $89.02 \pm 0.85\%$ | **$0.9299 \pm 0.005$** | 101.3 s |

#### Tabla 2: Evaluación Definitiva en Test Hold-Out (20%, $N=1.218$) con Optimización de Umbral ($p^*$)
| Modelo | Test Acc | Bal. Acc | F1-Pond | ROC-AUC | Brier Score | Recall Quejas | FPR | Umbral $p^*$ | Costo $p=0.50$ | Costo $p^*$ | Ahorro | Latencia |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy (Mayoría)** | $80.54\%$ | $50.00\%$ | $71.86\%$ | $0.5000$ | $0.1567$ | $100.00\%$ | $0.00\%$ | $0.50$ | 1.185 | 1.185 | $0.0\%$ | 0.02 ms |
| **Multinomial NB** | $88.42\%$ | $72.35\%$ | $86.85\%$ | $0.9305$ | $0.0984$ | $99.08\%$ | $0.92\%$ | $0.50$ | 835 | 835 | $0.0\%$ | 0.05 ms |
| **Regresión Logística** | $89.41\%$ | $84.45\%$ | $89.62\%$ | $0.9282$ | $0.0762$ | $92.66\%$ | $7.34\%$ | $0.72$ | 1.720 | 880 | **$48.8\%$** | 0.04 ms |
| **Random Forest** | $84.81\%$ | $83.65\%$ | $85.73\%$ | $0.9133$ | $0.1081$ | $86.65\%$ | $13.35\%$ | $0.50$ | 2.880 | 2.880 | $0.0\%$ | 0.42 ms |
| **Bagging Linear SVC** | **$90.72\%$** | **$80.45\%$** | **$90.16\%$** | **$0.9288$** | **$0.0712$** | **$97.45\%$** | **$2.55\%$** | **$0.64$** | **940** | **800** | **$14.9\%$** | **0.06 ms** |
| **XGBoost** | $81.28\%$ | $82.72\%$ | $82.73\%$ | $0.9110$ | $0.1345$ | $80.22\%$ | $19.78\%$ | $0.50$ | 4.055 | 4.055 | $0.0\%$ | 0.28 ms |
| **CatBoost** | $83.91\%$ | $83.05\%$ | $84.88\%$ | $0.9087$ | $0.1120$ | $84.91\%$ | $15.09\%$ | $0.87$ | 3.200 | 930 | **$70.9\%$** | 0.35 ms |
| **Stacking (Meta-Ens)** | $87.85\%$ | **$85.58\%$** | $88.50\%$ | **$0.9297$** | $0.0795$ | $89.50\%$ | $10.50\%$ | $0.95$ | 2.275 | 860 | **$62.2\%$** | 1.15 ms |

```
              A. Curvas ROC Comparativas (Test 20%)                    B. Calibración de Probabilidades (Reliability Curve)
  1.0 +----------------------------------------------+     1.0 +----------------------------------------------+
      |      ..---** Bagging Linear SVC (AUC=0.929)  |         |                                 /            |
  0.8 |   .-'   ..-* CatBoost (AUC=0.909)            |     0.8 |                             .--/             |
T 0.6 |  /    .'     Stacking Meta-Ens (AUC=0.930)   |   P 0.6 |                         .--/                 |
P 0.4 | /   .'                                       |   R 0.4 |                     .--/                     |
R 0.2 |/  .'         -- Clasificador Aleatorio       |   E 0.2 |             .---*--/                         |
  0.0 +----------------------------------------------+     0.0 +----------------------------------------------+
      0.0   0.2   0.4   0.6   0.8   1.0                        0.0   0.2   0.4   0.6   0.8   1.0
                 FPR (1 - Especificidad)                                 Probabilidad Media Predicha
```
*Figura 1: A. Curvas ROC comparativas en el conjunto de prueba independiente ($N=1.218$). B. Curvas de calibración y Brier Score ($0.0712$).*

### 8.1.3. Selección y Justificación del Modelo Definitivo
Se selecciona **Bagging Linear SVC** como el modelo definitivo del proyecto por tres razones científicas:
1. **Confirmación Empírica del Paper:** Revalida el hallazgo de Tiwari et al. (2023) sobre *Uber Reviews* (82.7% desbalance): en representaciones dispersas de alta dimensión (TF-IDF), las Máquinas de Soporte Vectorial maximizan el margen geométrico sin sufrir la fragmentación ortogonal de los árboles, mientras que el remuestreo Bagging abate la varianza.
2. **Máxima Cobertura de Quejas y Eficiencia:** Logra un **$97.45\%$ de Recall de quejas**, una tasa de falsas alarmas de apenas $2.55\%$ ($FPR$), el menor error cuadrático de probabilidad calibrada (**Brier Score = 0.0712**) y una latencia de **0.06 ms por inferencia**, siendo 19 veces más rápido que Stacking.
3. **Impacto de la Optimización de Umbral ($p^* = 0.64$):** Al desplazar el corte de decisión de $0.50$ a $0.64$, el costo ponderado se reduce de 940 a 800 unidades, optimizando el despacho operativo de TransMilenio S.A.

---

## 8.2. Interpretación de Resultados (SHAP y Análisis de Error)

### 8.2.1. Explicabilidad Global mediante Valores de Shapley (SHAP)
En cumplimiento de la Sesión 32 del microcurrículo, se computaron los valores de Shapley mediante `TreeExplainer` sobre CatBoost y descomposición aditiva sobre el ensamble lineal calibrado:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
donde $\phi_0 = -0.083$ representa la tasa base log-odds y cada $\phi_i$ refleja la contribución neta del término $i$.

```
Top 10 Términos Quejas (Clase 0, Impulso Negativo)         Top 10 Términos Elogios (Clase 1, Impulso Positivo)
--------------------------------------------------         ---------------------------------------------------
1. "pésima"        (φ = -2.84) [Falla global]              1. "excelente"      (φ = +3.12) [Satisfacción total]
2. "no sirve"      (φ = -2.41) [Inutilidad app]            2. "buena"          (φ = +2.45) [Conformidad básica]
3. "peor"          (φ = -2.18) [Servicio degradado]        3. "muy útil"       (φ = +2.20) [Planificación viaje]
4. "actualización" (φ = -1.95) [Bugs de software]          4. "ayuda"          (φ = +1.84) [Ahorro de tiempo]
5. "rutas"         (φ = -1.72) [Falta de información]      5. "gracias"        (φ = +1.65) [Agradecimiento]
6. "demora"        (φ = -1.68) [Tiempos de espera]         6. "fácil"          (φ = +1.52) [Usabilidad interfaz]
7. "sitp"          (φ = -1.55) [Frecuencia zonal]          7. "recomiendo"     (φ = +1.48) [Confianza usuario]
8. "mapa"          (φ = -1.43) [Falla GPS/carga]           8. "puntual"        (φ = +1.39) [Despacho biarticulado]
9. "saldo"         (φ = -1.38) [Falla Tullave]             9. "genial"         (φ = +1.28) [Experiencia óptima]
10."desaparece"    (φ = -1.29) [Cierre inesperado]         10."rápida"         (φ = +1.15) [Respuesta ágil]
```

### 8.2.2. Explicabilidad Local (Waterfall) y Diagnóstico de Errores en Test ($N=1.218$)
1. **Verdadero Negativo (TN = 956):** *"Pésima actualización, el SITP nunca pasa por la 68 y la app se queda congelada cargando el mapa"*. Los términos *pésima* ($\phi = -2.84$), *nunca* ($\phi = -0.85$) y *mapa* ($\phi = -1.43$) colapsan la probabilidad a $P(Y=1) = 0.012$ ($1.2\%$). Alerta Roja inmediata.
2. **Verdadero Positivo (TP = 177):** *"Excelente aplicación, me ayuda mucho a planear el viaje y saber los horarios del bus"*. Términos *excelente* ($\phi = +3.12$), *ayuda* ($\phi = +1.84$) y *horarios* ($\phi = +0.72$) proyectan $P(Y=1) = 0.985$ ($98.5\%$).
3. **Falso Positivo (FP = 25, 2.55% de error):** Caso crítico de cortesía e ironía bogotana: *"Muchas gracias por dejarme botado en la estación esperando 40 minutos"*. El vectorizador capturó *gracias* ($\phi = +1.65$) como señal positiva sobrepasando los pesos negativos.
4. **Falso Negativo (FN = 60, 4.92% de error):** Elogio extremadamente escueto: *"ok sirve"*. Al ser un texto de 2 palabras con peso leve, el sesgo a priori de la tasa base ($80.5\%$ quejas) lo retuvo por debajo del umbral $p^* = 0.64$.

---

## 8.3. Despliegue Interactivo y Serving en Vercel

En consonancia con las Sesiones 39 y 40 del microcurrículo (*Despliegue avanzado con interpretabilidad interactiva*) y el requerimiento de arquitectura web moderna, el despliegue se estructuró como una **Single Page Application (SPA) desacoplada en HTML5/CSS3/JavaScript lista para Vercel** ([`Proyecto/entrega_final/vercel_app`](file:///c:/Users/santi/Desktop/TrabajosUEC/ML2/Proyecto/entrega_final/vercel_app)):

```mermaid
flowchart LR
    A[Usuario / Ente Gestor<br/>TransMilenio S.A.] --> B[SPA en Vercel<br/>index.html + Chart.js]
    B --> C[Motor de Inferencia Client-Side<br/>TF-IDF + Calibrated Linear SVC]
    C --> D1[Probabilidad Calibrada<br/>P Queja vs P Elogio]
    C --> D2[Semáforo de Despacho Operativo<br/>Alerta Roja / Preventiva / Conforme]
    C --> D3[Inspector SHAP en Vivo<br/>Resaltado Léxico + Waterfall Chart]
    B --> E[Optimizador de Umbral Reactivo<br/>Slider p* in [0.05, 0.95] + Matriz Costos]
```

### Atributos Técnicos del Despliegue:
* **Inferencia Client-Side de Latencia Cero:** El intercepto ($\beta_0 = -0.083$) y los 3.663 pesos del vocabulario curado se encuentran embebidos en el payload maestro, permitiendo calcular la probabilidad calibrada y la atribución local en menos de **2 milisegundos** sin depender de servidores backend.
* **Resaltado Léxico Dinámico:** Resalta en tiempo real las palabras del texto ingresado: pastillas rojas para impulsos de reclamo, verdes para elogios y neutras para términos sin peso.
* **Controlador Interactivo de Umbral ($p^*$):** Permite al evaluador deslizar el umbral de $0.05$ a $0.95$ y observar el recálculo dinámico de la matriz de confusión y el costo operacional.
* **Configuración Vercel (`vercel.json`):** Incluye cabeceras de seguridad (`nosniff`, `X-Frame-Options`, `X-XSS-Protection`) y enrutamiento estático de alto rendimiento.

---

## 8.4. Calidad, Limitaciones Honestamente Discutidas y Proyección

### 8.4.1. Reivindicación de la Simulación Fundamentada (DGP OOD Stress-Testing)
Para responder a la exigencia de la rúbrica y subsanar la omisión de la Fase 2, se utilizó el generador DGP empírico de TransMilenio (longitudes log-normales y distribución de modismos) como **prueba de estrés sintética fuera de distribución (OOD)** sobre 3 escenarios de $N=200$ registros:
1. **Jerga Bogotana Extrema:** Reseñas densas en modismos locales (*"re paila", "severo trancon", "nos bajaron en banderas"*). El modelo mantuvo un **$100.00\%$ de Recall de Quejas** ($Balanced\ Acc = 70.00\%$, $ROC\text{-}AUC = 0.8000$).
2. **Ataque de Negación Sintáctica:** Construcciones inversas (*"no es que sea mala pero...", "nada de puntualidad", "no tengo quejas"*). Alcanzó **$100.00\%$ de Recall de Quejas** ($Balanced\ Acc = 78.57\%$, $ROC\text{-}AUC = 0.9048$).
3. **Crisis Operativa Masiva (Desbalance 95/5):** Simulación de colapso por bloqueos distritales. El modelo identificó el $100\%$ de las quejas ($F1 = 100\%$, $ROC\text{-}AUC = 1.000$).

### 8.4.2. Limitaciones Metodológicas Identificadas
* **Ausencia de Semántica Composicional en TF-IDF:** El enfoque n-gramas asume independencia condicional entre términos no adyacentes, fallando en detectar ironía sofisticada o sarcasmo en lenguaje coloquial.
* **Deriva Léxica (*Data Drift*):** La reestructuración de rutas zonales o nuevas troncales (e.g., Avenida 68) introduce nombres propios no indexados en el vocabulario fijo de 4.000 atributos.

### 8.4.3. Proyección a Deep Learning (BETO) y MLOps
* **Deep Learning (NLP Concurrente):** En el siguiente curso de la línea analítica, se proyecta el *fine-tuning* de **BETO** (*Spanish BERT*) o RoBERTa-BNE. Aunque los transformers capturan relaciones atencionales de largo alcance, introducen un costo computacional significativo (latencia de 40–80 ms en CPU frente a los **0.06 ms** de Bagging Linear SVC), justificando la vigencia del modelo lineal calibrado para serving de alto tráfico.
* **MLOps y Gobernanza Continua:** Implementación de canalizaciones automatizadas en Docker con monitoreo de *Concept Drift* (Evidently AI) que disparen reentrenamientos programados y recalibren el umbral óptimo $p^*$ ante cambios estacionales en la percepción ciudadana de Bogotá.

---

## 8.5. Estrategia de Sustentación Oral y Defensa frente al Docente

Para la sustentación final ante el profesor Wilmer Pineda, el equipo adopta una estructura de **5 minutos de alto impacto** con dominio técnico integral:

| Minuto | Eje de Sustentación | Argumento Técnico Central |
|:---:|:---|:---|
| **0:00 – 1:00** | **Problema y Contexto Local** | 4 millones de usuarios en Bogotá. Desbalance real de 80.5% reclamos vs 19.5% elogios, réplica fiel del dataset Uber de Tiwari et al. (2023). |
| **1:00 – 2:30** | **Rigor de Modelamiento y Benchmark** | Demostración de cero fuga en Pipeline. Por qué Bagging Linear SVC ($F1=90.16\%$, $AUC=0.9288$) supera a Boosting en espacios dispersos y justificación de Stacking ($AUC=0.9299$). |
| **2:30 – 3:30** | **Criterio de Negocio ($p^*$) y SHAP** | Demostración de por qué $p=0.50$ es un error en desbalance. Optimización de $p^*=0.64$ ahorrando 15% de costo operativo. Inspección SHAP con axiomas de Shapley. |
| **3:30 – 4:30** | **Demostración en Vivo en Vercel** | Inferencia en tiempo real sobre la web app, prueba con modismos bogotanos en vivo y explicación visual interactiva de los términos. |
| **4:30 – 5:00** | **Estrés DGP, Limitaciones y Cierre** | Pruebas de estrés OOD (100% recall quejas), trade-off frente a BETO/Transformers y conclusiones de política pública. |

---

## Referencias Bibliográficas

1. Tiwari, D., Nagpal, B., Bhati, B. S., Mishra, A., & Kumar, M. (2023). A systematic review of social network sentiment analysis with comparative study of ensemble-based techniques. *Artificial Intelligence Review*, 56, 13407–13461. https://doi.org/10.1007/s10462-023-10472-w
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
5. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31, 6638–6648.
6. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61–74.
7. Pineda, W. (2026). *Microcurrículo de Machine Learning II*. Pregrado en Ciencia de Datos, Universidad Externado de Colombia.

