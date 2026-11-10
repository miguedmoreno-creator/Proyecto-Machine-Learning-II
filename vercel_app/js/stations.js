/**
 * ==============================================================================
 * CONTENIDO Y CONTROLADOR DE ESTACIONES (10-MIN PUNCHY DATA STORYTELLING)
 * Textos concisos para exposición de 10 minutos con gráficas 100% dinámicas
 * y tarjetas explicativas didácticas en lenguaje sencillo para cada sección.
 * ==============================================================================
 */

window.StationsModule = (function () {
    let activeModalIndex = 0;
    let visitedStations = new Set([0]);

    const STATIONS = [
        {
            id: 0,
            code: "PORTAL SALIDA",
            name: "Portal del Dato Crudo",
            subtitle: "El Problema & Tiwari et al. (2023)",
            x: 200,
            character: "Jefe de Patio S23",
            avatar: "👨‍✈️",
            dialog: "¡Bienvenidos a la Ruta S23! Cada día 4 millones de bogotanos viajan en TransMilenio. Analizamos 6.089 reseñas reales de TransMi[App] para predecir y alertar fallas operativas en 0.06 milisegundos."
        },
        {
            id: 1,
            code: "ESTACIÓN 1",
            name: "Estación Voz Bogotana",
            subtitle: "EDA & Asimetría de Sentimiento",
            x: 920,
            character: "Doña Martha (Usuaria SITP)",
            avatar: "👵",
            dialog: "Mijo, cuando el bus se demora 40 minutos uno escribe un testamento; cuando llega rápido, solo pone 'gracias'. ¡Por eso las quejas son 2.4 veces más largas que los elogios!"
        },
        {
            id: 2,
            code: "ESTACIÓN 2",
            name: "Estación Aduana Cero Fuga",
            subtitle: "Pipeline Hermético & 5-Fold CV",
            x: 1640,
            character: "Inspectora de Integridad",
            avatar: "🕵️‍♀️",
            dialog: "¡Alto ahí! En esta aduana no permitimos trampas ni fuga de información. Evaluamos 11 modelos con Validación Cruzada de 5 pliegues estrictamente sellada."
        },
        {
            id: 3,
            code: "ESTACIÓN 3",
            name: "Estación Torneo de Ensambles",
            subtitle: "Bagging vs Boosting vs Stacking",
            x: 2360,
            character: "Árbitro de Modelos",
            avatar: "⚖️",
            dialog: "En este cuadrilátero compiten 11 modelos. Revalidamos a Tiwari et al.: Bagging Linear SVC domina con 90.16% de F1 y el Brier Score más bajo (0.0712). ¡Las probabilidades son honestas!"
        },
        {
            id: 4,
            code: "ESTACIÓN 4",
            name: "Estación Dilema del Costo",
            subtitle: "Matriz Asimétrica & Umbral Óptimo",
            x: 3080,
            character: "Director de Operaciones",
            avatar: "👔",
            dialog: "Ignorar una queja real nos cuesta 4 veces más que revisar un elogio falso. Mueve el slider: al cambiar del ingenuo 0.50 al óptimo 0.64 ahorramos hasta un 70.9% en costos operativos."
        },
        {
            id: 5,
            code: "ESTACIÓN 5",
            name: "Estación Caja de Cristal",
            subtitle: "SHAP Global & Auditoría de Errores",
            x: 3800,
            character: "Lingüista Rolo",
            avatar: "🧠",
            dialog: "Descomponemos cada palabra con Valores de Shapley. Las quejas se delatan con 'pésima' y 'no sirve', pero descubrimos el talón de Aquiles de la IA: ¡el sarcasmo bogotano!"
        },
        {
            id: 6,
            code: "ESTACIÓN 6",
            name: "Laboratorio de Despacho",
            subtitle: "Simulador Interactivo en Vivo",
            x: 4520,
            character: "Operador de Centro de Control",
            avatar: "🎙️",
            dialog: "¡Prueba nuestro modelo en tiempo real! Escribe una reseña o presiona los botones de ejemplo para ver la predicción instantánea y la alerta en 0.06 ms."
        },
        {
            id: 7,
            code: "PORTAL LLEGADA",
            name: "Portal Inteligencia Ciudadana",
            subtitle: "Estrés DGP, Límites & BETO",
            x: 5240,
            character: "Arquitecto MLOps",
            avatar: "🚀",
            dialog: "¡Llegamos a la meta! El modelo resistió pruebas de estrés con datos simulados (100% recall). Conocemos nuestros límites BoW y trazamos la ruta hacia Transformers (BETO) y MLOps."
        }
    ];

    const OBSTACLES = [
        {
            id: 1, x: 580, yPct: 56, icon: "🕳️🚧", label: "Hueco del Ruido Léxico",
            title: "Hueco 1: Jerga, Emojis y Textos Cortos",
            problem: "Reseñas sin tildes, con faltas de ortografía ('pesima', 'actualisacion') y modismos que confunden a vectorizadores ingenuos.",
            solution: "Pipeline de limpieza con normalización Unicode NFD, remoción de puntuación y filtrado de reseñas vacías (<3 caracteres)."
        },
        {
            id: 2, x: 1280, yPct: 62, icon: "⚠️📉", label: "Trancón del Desbalance Severo",
            title: "Trancón 2: El 80.6% de los datos son Quejas",
            problem: "Un modelo perezoso que prediga siempre 'Queja' obtendría 80.6% de Accuracy sin aprender absolutamente nada.",
            solution: "Evaluación con Balanced Accuracy, Macro/Weighted F1-Score y ponderación de costos de clase (class_weight='balanced')."
        },
        {
            id: 3, x: 2000, yPct: 50, icon: "🚧🕵️", label: "Retén de la Fuga de Datos",
            title: "Retén 3: Riesgo de Optimism Bias",
            problem: "Calcular el vocabulario TF-IDF sobre todo el corpus antes del CV filtra información del futuro y falsea las métricas.",
            solution: "Aduana hermética: vectorizador TF-IDF anidado dentro de cada pliegue del 5-Fold Stratified CV y hold-out 20% sellado."
        },
        {
            id: 4, x: 2720, yPct: 66, icon: "🕳️🎲", label: "Hueco de Probabilidades Engañosas",
            title: "Hueco 4: Modelos 'Sobreconfiados' pero Equivocados",
            problem: "Modelos como SVM y boosting emiten puntajes de decisión que no corresponden a probabilidades reales calibradas.",
            solution: "CalibratedClassifierCV mediante regresión logística sigmoidea de Platt y evaluación objetiva con Brier Score (0.0712)."
        },
        {
            id: 5, x: 3440, yPct: 54, icon: "⚠️💰", label: "Trancón del Umbral Simétrico 0.50",
            title: "Trancón 5: La Trampa de Decidir con Moneda al Aire",
            problem: "Fijar umbral p=0.50 ignora que no escuchar una queja real cuesta 4 veces más que revisar un elogio dudoso.",
            solution: "Optimización sistemática de la función de costo asimétrico C(p) = 20 FP + 5 FN, hallando el umbral óptimo p* = 0.64."
        },
        {
            id: 6, x: 4160, yPct: 64, icon: "🎭🌧️", label: "Protesta del Sarcasmo Rolo",
            title: "Protesta 6: Cortesía Irónica Bogotana",
            problem: "'Muchas gracias por dejarme botado en la lluvia': palabras positivas usadas para denunciar fallas operativas severas.",
            solution: "Auditoría forense de los 113 errores de test, n-gramas (1,2) para capturar negaciones y proyección a Transformers (BETO)."
        },
        {
            id: 7, x: 4880, yPct: 52, icon: "🚧🧪", label: "Bache del Cambio de Distribución",
            title: "Bache 7: ¿Qué pasa ante un Paro o Crisis?",
            problem: "El modelo entrenado con datos normales podría colapsar si la jerga cambia drásticamente o las quejas suben al 95%.",
            solution: "Stress Testing con Proceso Generador de Datos (DGP Log-Normal): demostró 100% de Recall de quejas en escenarios OOD."
        }
    ];

    function abrirEstacion(idx) {
        activeModalIndex = idx;
        visitedStations.add(idx);

        const st = STATIONS[idx];
        document.getElementById('modalStationBadge').textContent = st.code;
        document.getElementById('modalCharacterRole').textContent = `${st.avatar} ${st.character}`;
        document.getElementById('modalStationTitle').textContent = st.name;

        const body = document.getElementById('modalStationBody');
        body.innerHTML = construirContenidoConcisoEstacion(idx);

        document.getElementById('stationModal').classList.remove('hidden');

        // Renderizar gráficas interactivas dinámicas
        setTimeout(() => {
            inicializarGraficasDinamicas(idx);
        }, 50);
    }

    function cerrarEstacion() {
        document.getElementById('stationModal').classList.add('hidden');
    }

    function navegarEstacion(delta) {
        const next = (activeModalIndex + delta + STATIONS.length) % STATIONS.length;
        abrirEstacion(next);
    }

    // Storytelling enriquecido: diálogo + texto explicativo didáctico + diapositivas/gráficas dinámicas
    function construirContenidoConcisoEstacion(idx) {
        const st = STATIONS[idx];
        let content = `
            <!-- 1. DIÁLOGO DEL PERSONAJE ANFITRIÓN -->
            <div class="p-3.5 sm:p-4 rounded-2xl bg-slate-900 border-l-4 border-tm-amber flex items-center space-x-3 text-xs sm:text-sm shadow-md">
                <span class="text-3xl shrink-0">${st.avatar}</span>
                <div>
                    <span class="text-[10px] font-mono text-tm-yellow uppercase font-bold block">${st.character}:</span>
                    <p class="text-slate-200 leading-snug font-medium italic">"${st.dialog}"</p>
                </div>
            </div>
        `;

        // -------------------------------------------------------------
        // ESTACIÓN 0: PORTAL DE SALIDA (EL PROBLEMA & TIWARI ET AL.)
        // -------------------------------------------------------------
        if (idx === 0) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-map-location-dot text-tm-amber text-sm"></i>
                        <span>¿Qué problema resolvemos en este viaje de datos?</span>
                    </div>
                    <p>
                        TransMilenio moviliza a más de <strong>4 millones de pasajeros cada día en Bogotá</strong>. Cuando un usuario vive una mala experiencia (un bus que no pasa, una falla de saldo en la tarjeta TuLlave o la aplicación congelada en un mapa), recurre a la tienda oficial de Google Play para expresar su frustración. Para un equipo humano es físicamente imposible leer miles de quejas al día para enterarse de qué está pasando en la calle. Por eso, en este proyecto tomamos como base el artículo internacional de <em>Tiwari et al. (2023)</em> en <em>Artificial Intelligence Review</em> para responder una pregunta vital: <strong>¿puede la Inteligencia Artificial escuchar la voz ciudadana, distinguir automáticamente una queja de un elogio y alertar al centro de despacho en tiempo récord sin cometer trampas estadísticas?</strong>
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-lightbulb text-tm-teal"></i>
                        <span><strong>Meta de la Ruta S23:</strong> Determinar qué algoritmo de Machine Learning predice con mayor certeza las quejas operativas y cómo ponerlo a operar en tiempo real bajo estrictos estándares industriales.</span>
                    </div>
                </div>

                <!-- Tarjetas de Métricas Clave y Comparativa del Paper -->
                <div class="grid grid-cols-2 md:grid-cols-5 gap-2.5 text-center font-mono text-xs">
                    <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                        <span class="text-[10px] text-slate-400 block">Corpus Curado</span>
                        <span class="text-xl font-heading font-black text-white">6.089</span>
                        <span class="text-[9px] text-tm-teal block">Google Play Store</span>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                        <span class="text-[10px] text-slate-400 block">Desbalance Real</span>
                        <span class="text-xl font-heading font-black text-tm-red">80.6%</span>
                        <span class="text-[9px] text-slate-400 block">4.906 Quejas / 1.183 Elogios</span>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                        <span class="text-[10px] text-slate-400 block">Modelo Ganador</span>
                        <span class="text-lg font-heading font-black text-tm-yellow">Bagging SVC</span>
                        <span class="text-[9px] text-slate-400 block">F1: 90.16% | AUC: 0.929</span>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                        <span class="text-[10px] text-slate-400 block">Recall Quejas</span>
                        <span class="text-xl font-heading font-black text-emerald-400">97.45%</span>
                        <span class="text-[9px] text-slate-400 block">Solo 25 quejas omitidas</span>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900 border border-white/10 col-span-2 md:col-span-1">
                        <span class="text-[10px] text-slate-400 block">Latencia Serving</span>
                        <span class="text-xl font-heading font-black text-sky-400">0.06 ms</span>
                        <span class="text-[9px] text-slate-400 block">19x más rápido que Stacking</span>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                    <div class="p-4 rounded-2xl bg-slate-900/90 border border-white/10 space-y-2">
                        <h4 class="font-heading font-bold text-white text-sm flex items-center space-x-2">
                            <i class="fa-solid fa-book-open text-tm-red"></i>
                            <span>1. El Benchmark del Paper (Tiwari et al., 2023)</span>
                        </h4>
                        <ul class="space-y-1.5 text-slate-300">
                            <li>• <strong>Pregunta RQ5:</strong> ¿Cuál arquitectura de ensamble domina en análisis de sentimiento con matrices dispersas TF-IDF?</li>
                            <li>• <strong>Hallazgo Central:</strong> En datasets desbalanceados masivos (<em>Uber Reviews 82.7% quejas</em>), Bagging con Linear SVC superó consistentemente a Boosting.</li>
                        </ul>
                    </div>
                    <div class="p-4 rounded-2xl bg-slate-900/90 border border-white/10 space-y-2">
                        <h4 class="font-heading font-bold text-white text-sm flex items-center space-x-2">
                            <i class="fa-solid fa-award text-tm-teal"></i>
                            <span>2. Nuestro Valor Agregado en Bogotá (ML2)</span>
                        </h4>
                        <ul class="space-y-1.5 text-slate-300">
                            <li>• <strong>Datos Propios:</strong> 6.089 reseñas reales en español bogotano ($N \ge 5.000$).</li>
                            <li>• <strong>Cero Fuga:</strong> 5-Fold CV estricto + Meta-Ensamble Stacking.</li>
                            <li>• <strong>Criterio de Decisión:</strong> Optimización de umbral $p^*=0.64$ y explicabilidad SHAP.</li>
                        </ul>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 1: VOZ BOGOTANA (EDA & ASIMETRÍA DE SENTIMIENTO)
        // -------------------------------------------------------------
        if (idx === 1) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-users text-tm-amber text-sm"></i>
                        <span>¿Cómo hablan los ciudadanos y qué revela la exploración de datos?</span>
                    </div>
                    <p>
                        Antes de entrenar cualquier modelo matemático, necesitamos escuchar a la gente. Al inspeccionar nuestras <strong>6.089 reseñas reales</strong>, descubrimos una realidad sociológica contundente: <strong>el 80.57% del corpus son quejas</strong> (1★ y 2★) frente a solo un 19.43% de elogios (4★ y 5★). Además, la psicología del usuario bogotano genera una gran <em>asimetría de longitud</em>: quien está satisfecho suele dejar frases muy cortas como <em>"buena app, gracias"</em> (mediana de 11 palabras), mientras que quien enfrenta un retraso o un bloqueo de saldo escribe quejas detalladas de hasta 80 palabras (mediana de 18 palabras).
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-chart-column text-tm-teal"></i>
                        <span><strong>Guía de visualización:</strong> Abajo puedes interactuar con dos gráficos: a la izquierda, la distribución de calificaciones por estrellas; a la derecha, la comparación de longitud en palabras que demuestra por qué un modelo ingenuo podría confundirse creyendo que "más texto" equivale automáticamente a una queja.</span>
                    </div>
                </div>

                <!-- Gráficas Dinámicas de Calificaciones y Longitud -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">📊 Distribución Real de Calificaciones (1★ a 5★)</span>
                            <span class="font-mono text-tm-red font-bold">Desbalance 80.6% / 19.4%</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartRatings"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                            <strong>Takeaway:</strong> El 65.6% son 1★ y 14.9% son 2★ (ratio 4.15 a 1). Replica de forma orgánica la estructura de Uber Reviews en el paper base.
                        </p>
                    </div>

                    <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">📏 Longitud Textual: Quejas vs Elogios</span>
                            <span class="font-mono text-tm-yellow font-bold">Mediana: 18 vs 11 palabras</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartLengths"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                            <strong>Insight Estadístico:</strong> Distribución Log-Normal asimétrica. Justifica el uso de sublinear TF-IDF para amortiguar el peso de quejas muy largas.
                        </p>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 2: ADUANA CERO FUGA (PIPELINE HERMÉTICO & 5-FOLD CV)
        // -------------------------------------------------------------
        if (idx === 2) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-shield-halved text-tm-amber text-sm"></i>
                        <span>¿Por qué la integridad científica exige una Aduana "Cero Fuga"?</span>
                    </div>
                    <p>
                        En la vida real, un modelo de Machine Learning debe enfrentarse a ciudadanos y expresiones que nunca ha visto antes. Uno de los errores más graves en ciencia de datos es la <strong>Fuga de Información (Data Leakage)</strong>: si calculamos las frecuencias de las palabras (TF-IDF) usando todo el dataset antes de separar los datos de prueba, el modelo "hace trampa", memoriza las respuestas del examen y muestra un rendimiento falso que luego colapsa en producción. Para garantizar un rigor industrial irreprochable, sellamos un conjunto de prueba independiente del 20% ($N=1.218$) y encapsulamos el preprocesamiento y el entrenamiento dentro de una <strong>Validación Cruzada Estratificada de 5 Pliegues (5-Fold CV)</strong>.
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-magnifying-glass-chart text-tm-teal"></i>
                        <span><strong>Guía de visualización:</strong> La gráfica interactiva inferior compara la estabilidad y el rendimiento de 4 familias distintas de modelos a lo largo de los 5 pliegues. Pasa el cursor sobre las barras para observar cómo el ensamble Bagging Linear SVC logra la menor dispersión y la máxima solidez estadística.</span>
                    </div>
                </div>

                <!-- Gráfica Dinámica de Comparación CV y Salvaguardas -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
                    <div class="lg:col-span-7 p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">📈 Rendimiento y Variabilidad en Stratified 5-Fold CV</span>
                            <span class="font-mono text-tm-teal font-bold">Train Set 80% (N=4.871)</span>
                        </div>
                        <div class="h-64 w-full"><canvas id="chartCVModels"></canvas></div>
                    </div>
                    <div class="lg:col-span-5 space-y-2.5 text-xs">
                        <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                            <strong class="text-tm-teal block mb-1">🛡️ Salvaguardas Contra Data Leakage:</strong>
                            <ul class="space-y-1 text-slate-300">
                                <li>• <strong>Bóveda Sellada:</strong> 20% test (1.218 reseñas) reservado sin abrir.</li>
                                <li>• <strong>TF-IDF en cada Fold:</strong> Vocabulario de 4.000 términos entrenado únicamente en los pliegues de train de cada iteración.</li>
                                <li>• <strong>Zero Optimism Bias:</strong> Estimación imparcial de variabilidad estadística $(\mu \pm \sigma)$.</li>
                            </ul>
                        </div>
                        <div class="p-3 rounded-xl bg-slate-950 border border-slate-800 text-slate-300">
                            <strong class="text-tm-yellow block mb-1">💡 Conclusión CV:</strong>
                            Bagging Linear SVC y Stacking lideran en estabilidad con desviaciones típicas menores al $0.7\%$, mientras que AdaBoost muestra alta inestabilidad.
                        </div>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 3: TORNEO DE ENSAMBLES (ROC & CALIBRACIÓN DE PLATT)
        // -------------------------------------------------------------
        if (idx === 3) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-trophy text-tm-amber text-sm"></i>
                        <span>El Enfrentamiento de Familias y la Honestidad de las Probabilidades</span>
                    </div>
                    <p>
                        Siguiendo los fundamentos del <strong>Teorema de Condorcet</strong> (Sesión 10 del microcurrículo), cuando combinamos múltiples modelos que cometen errores independientes, el conjunto supera a cualquier clasificador individual. Enfrentamos 11 algoritmos repartidos en 4 familias: <em>Líneas Base (Regresión Logística, Naive Bayes), Bagging (Random Forest, Extra Trees, Bagging Linear SVC), Boosting (AdaBoost, Gradient Boosting, XGBoost, CatBoost) y Stacking Meta-Ensamble</em>. Pero predecir no es suficiente; el sistema debe ser **honesto**: si dice que la probabilidad de que una reseña sea un elogio es del 80%, en la práctica 8 de cada 10 deben serlo de verdad. Esto se mide con el <strong>Brier Score</strong> y la <strong>Calibración Sigmoidea de Platt</strong>.
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-code-compare text-tm-teal"></i>
                        <span><strong>Guía de visualización:</strong> A la izquierda tienes la Curva ROC (que evalúa la capacidad de discriminación, donde Bagging SVC y Stacking alcanzan AUC > 0.928); a la derecha tienes la Curva de Calibración, donde la cercanía a la diagonal punteada indica qué tan honesto y fiable es cada modelo al emitir probabilidades.</span>
                    </div>
                </div>

                <!-- Gráficas Dinámicas ROC y Calibración -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">🎯 Curvas ROC en Test Hold-Out (N=1.218)</span>
                            <span class="font-mono text-tm-yellow font-bold">AUC: 0.9288 (SVC) vs 0.9297 (Stack)</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartROC"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2 rounded-xl border border-slate-800">
                            <strong>Condorcet & SVM:</strong> El hiperplano de soporte vectorial maneja con superioridad el espacio de 4.000 palabras dispersas frente a árboles ortogonales.
                        </p>
                    </div>

                    <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">⚖️ Confiabilidad de Probabilidades (Calibración Platt)</span>
                            <span class="font-mono text-tm-teal font-bold">Brier Score Mínimo: 0.0712</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartCalibration"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2 rounded-xl border border-slate-800">
                            <strong>CalibratedClassifierCV:</strong> Las probabilidades emitidas por Bagging SVC son insesgadas, condición vital para la optimización de costos.
                        </p>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 4: DILEMA DEL COSTO (OPTIMIZACIÓN DE UMBRAL p*)
        // -------------------------------------------------------------
        if (idx === 4) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-scale-balanced text-tm-amber text-sm"></i>
                        <span>¿Por qué clasificar al 50% es un grave error para la operación de la ciudad?</span>
                    </div>
                    <p>
                        En la academia tradicional se suele enseñar que si una probabilidad es mayor a $0.50$, se decide por una categoría y si es menor, por la otra. Sin embargo, en el servicio público de TransMilenio <strong>los errores no tienen el mismo impacto</strong>. Un <em>Falso Positivo</em> (ignorar una queja ciudadana real confundiéndola con un elogio) genera bloqueos de estaciones, paros o usuarios indignados sin saldo en la lluvia, por lo que le asignamos una penalización de <strong>$20 unidades de costo</strong>. Por el contrario, un <em>Falso Negativo</em> (revisar por exceso de precaución un elogio creyendo que era una queja) apenas cuesta <strong>$5 unidades</strong>. La solución no es adivinar, sino calcular matemáticamente el <strong>umbral óptimo $p^*$</strong> que minimice la pérdida total de negocio.
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-sliders text-tm-teal"></i>
                        <span><strong>Guía de interactividad:</strong> A la izquierda observa la curva teórica de costo; a la derecha tienes un <strong>slider interactivo en vivo</strong>. Desliza el control: al moverte de 0.50 al punto óptimo $p^* = 0.64$, el costo operativo se reduce de 940 a 800 unidades (ahorro del 14.9% en Bagging SVC y hasta 70.9% en CatBoost), asegurando la captura del 98.3% de las quejas.</span>
                    </div>
                </div>

                <!-- Gráfica Dinámica de Costo y Slider Interactivo -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
                    <div class="lg:col-span-7 p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">📉 Curva de Costo Asimétrico C(p) = 20·FP + 5·FN</span>
                            <span class="font-mono text-tm-amber font-bold">Mínimo en p* = 0.64</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartCostCurve"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2 rounded-xl border border-slate-800">
                            Con p=0.50 el costo era 940. Con <strong>p*=0.64</strong> cae a <strong>800 (15% ahorro)</strong>; en CatBoost el ahorro es del <strong>70.9%</strong>.
                        </p>
                    </div>

                    <div class="lg:col-span-5 p-4 rounded-2xl bg-slate-900 border-2 border-tm-amber/60 space-y-3.5">
                        <div class="flex justify-between items-center">
                            <span class="text-xs font-mono uppercase text-tm-yellow font-bold">🎛️ Desliza el Umbral (p):</span>
                            <span id="st4SliderVal" class="px-2.5 py-0.5 rounded bg-tm-amber text-slate-950 font-mono font-black text-sm">0.64</span>
                        </div>
                        <input id="st4ThresholdSlider" type="range" min="0.05" max="0.95" step="0.01" value="0.64" class="w-full accent-amber-500 cursor-pointer">

                        <div class="grid grid-cols-2 gap-2 text-center text-xs">
                            <div class="p-2 rounded-lg bg-emerald-950/50 border border-emerald-700/50">
                                <span class="text-[10px] text-emerald-400 block">TN (Quejas Detectadas)</span>
                                <span id="st4TN" class="text-base font-bold text-white">964</span>
                            </div>
                            <div class="p-2 rounded-lg bg-red-950/50 border border-red-700/50">
                                <span class="text-[10px] text-red-400 block">FP (Quejas Ignoradas x20)</span>
                                <span id="st4FP" class="text-base font-bold text-white">17</span>
                            </div>
                            <div class="p-2 rounded-lg bg-amber-950/50 border border-amber-700/50">
                                <span class="text-[10px] text-amber-400 block">FN (Elogios Dudosos x5)</span>
                                <span id="st4FN" class="text-base font-bold text-white">92</span>
                            </div>
                            <div class="p-2 rounded-lg bg-emerald-950/50 border border-emerald-700/50">
                                <span class="text-[10px] text-emerald-400 block">TP (Elogios Reales)</span>
                                <span id="st4TP" class="text-base font-bold text-white">145</span>
                            </div>
                        </div>

                        <div class="flex justify-between items-center pt-1 border-t border-white/10 text-xs font-mono">
                            <span>Costo: <strong id="st4Cost" class="text-tm-yellow font-bold">800</strong></span>
                            <span>Recall Quejas: <strong id="st4Rec0" class="text-emerald-400 font-bold">98.27%</strong></span>
                        </div>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 5: CAJA DE CRISTAL (SHAP & AUDITORÍA DE ERRORES)
        // -------------------------------------------------------------
        if (idx === 5) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-brain text-tm-amber text-sm"></i>
                        <span>Explicabilidad con Valores de Shapley y la Trampa del Sarcasmo Rolo</span>
                    </div>
                    <p>
                        Un modelo de Inteligencia Artificial para la ciudadanía no puede ser una "caja negra" opaca. Para auditarlo, recurrimos a los <strong>Valores de Shapley (SHAP, Sesión 32 del microcurrículo)</strong>, una técnica fundamentada en la teoría de juegos cooperativos que descompone de forma matemáticamente exacta y aditiva cuánto empuja cada palabra del usuario hacia una decisión de Queja o hacia una de Elogio. Además, realizamos una auditoría forense de los errores del modelo: descubrimos que la principal causa de confusión de la IA es el <strong>sarcasmo bogotano</strong>, en el que los usuarios utilizan palabras formalmente positivas (como <em>"excelente"</em> o <em>"gracias"</em>) para denunciar que los dejaron botados esperando un bus bajo el aguacero.
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-chart-pie text-tm-teal"></i>
                        <span><strong>Guía de visualización:</strong> A la izquierda verás la gráfica con las 16 palabras con mayor peso de decisión (rojo para quejas, verde para elogios); a la derecha tienes la matriz de aciertos y falsas alarmas, donde puedes comprobar que solo 25 quejas de 981 pasaron inadvertidas.</span>
                    </div>
                </div>

                <!-- Gráficas Dinámicas SHAP y Análisis de Errores -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">🔍 Top 16 Palabras Determinantes SHAP</span>
                            <span class="font-mono text-purple-300 font-bold">Valores de Shapley</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartSHAP"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2 rounded-xl border border-slate-800">
                            <strong>Axiomas de Shapley:</strong> <em>"pésima" (-2.84)</em> y <em>"no sirve" (-2.41)</em> determinan quejas; <em>"excelente" (+3.12)</em> y <em>"muy útil" (+2.20)</em> impulsan elogios.
                        </p>
                    </div>

                    <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">🚨 Auditoría Forense de Errores en Test</span>
                            <span class="font-mono text-red-400 font-bold">9.2% de Error Global</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartErrors"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2 rounded-xl border border-slate-800">
                            <strong>El Reto del Sarcasmo:</strong> 25 Falsos Positivos donde el usuario usa cortesía irónica (<em>"Muchas gracias por dejarme botado 40 minutos"</em>).
                        </p>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 6: LABORATORIO DE DESPACHO EN VIVO (SIMULADOR)
        // -------------------------------------------------------------
        if (idx === 6) {
            content += `
                <!-- Texto explicativo didáctico antes del simulador -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-flask-vial text-tm-amber text-sm"></i>
                        <span>Prueba Interactiva en Tiempo Real (Inferencia Client-Side en 0.06 ms)</span>
                    </div>
                    <p>
                        En esta estación puedes poner a prueba nuestro modelo de forma totalmente libre e interactiva. No hay llamadas a servidores remotos ni trucos de diseño: el motor de inferencia matemática corre <strong>100% en vivo dentro de tu navegador</strong>. El algoritmo normaliza el texto, elimina ruido ortográfico, consulta los pesos del modelo calibrado y predice la categoría en apenas <strong>0.06 milisegundos</strong> (lo que permitiría procesar más de 16.000 reseñas por segundo en un centro de despacho real). Puedes escribir tus propias frases con expresiones bogotanas o presionar los botones preconfigurados para observar cómo reacciona ante distintos niveles de severidad.
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-terminal text-tm-teal"></i>
                        <span><strong>Qué observar en la pantalla:</strong> A medida que escribes o cargas un ejemplo, el <em>Inspector de Palabras</em> pintará en verde los términos positivos y en rojo los negativos según su valor SHAP, actualizando instantáneamente la barra de probabilidades, el estado de alerta operativa y el gráfico waterfall de impacto.</span>
                    </div>
                </div>

                <!-- Interfaz del Simulador en Vivo y Waterfall Chart -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
                    <div class="lg:col-span-7 space-y-3">
                        <div class="p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                            <span class="text-xs font-mono text-tm-yellow uppercase font-bold block">✍️ Escribe o elige una reseña real:</span>
                            <textarea id="simInputText" rows="2" class="w-full bg-slate-950 border border-slate-700 rounded-xl p-2.5 text-xs text-white focus:border-tm-red focus:outline-none">Pesima actualizacion, el sitp nunca pasa por el paradero de la 68 y la app se queda congelada cargando el mapa de rutas</textarea>
                            <div class="flex flex-wrap gap-1.5 text-[11px]">
                                <button onclick="SimulatorModule.cargarEjemplo(1)" class="px-2 py-1 rounded bg-red-950/60 hover:bg-red-900 text-red-200 border border-red-700/50">🔴 Falla SITP</button>
                                <button onclick="SimulatorModule.cargarEjemplo(2)" class="px-2 py-1 rounded bg-red-950/60 hover:bg-red-900 text-red-200 border border-red-700/50">🔴 Saldo TuLlave</button>
                                <button onclick="SimulatorModule.cargarEjemplo(3)" class="px-2 py-1 rounded bg-emerald-950/60 hover:bg-emerald-900 text-emerald-200 border border-emerald-700/50">🟢 Planificación</button>
                                <button onclick="SimulatorModule.cargarEjemplo(4)" class="px-2 py-1 rounded bg-emerald-950/60 hover:bg-emerald-900 text-emerald-200 border border-emerald-700/50">🟢 Ahorro Tiempo</button>
                                <button onclick="SimulatorModule.cargarEjemplo(5)" class="px-2 py-1 rounded bg-amber-950/60 hover:bg-amber-900 text-amber-200 border border-amber-700/50">🎭 Sarcasmo Rolo</button>
                            </div>
                        </div>

                        <div class="p-3.5 rounded-2xl bg-slate-900 border border-white/10 space-y-1.5">
                            <span class="text-xs font-heading font-bold text-white block">Atribución SHAP de Palabras:</span>
                            <div id="simHighlightedWords" class="p-3 rounded-xl bg-slate-950 border border-slate-800 min-h-[50px] flex flex-wrap gap-1 items-center text-xs"></div>
                        </div>
                    </div>

                    <div class="lg:col-span-5 p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-3">
                        <div id="simStatusBanner" class="p-3 rounded-xl border flex items-center space-x-2.5 bg-red-950/60 border-red-700 text-white">
                            <div id="simStatusIcon" class="text-2xl">🚨</div>
                            <div>
                                <span id="simStatusSub" class="text-[9px] font-mono uppercase text-red-300 block">Alerta Operativa</span>
                                <span id="simStatusMain" class="text-sm font-heading font-black">QUEJA OPERATIVA CRÍTICA</span>
                            </div>
                        </div>

                        <div class="space-y-1">
                            <div class="flex justify-between text-xs font-mono">
                                <span class="text-red-400 font-bold">P(Queja): <span id="simProbQueja">95.0%</span></span>
                                <span class="text-emerald-400 font-bold">P(Elogio): <span id="simProbElogio">5.0%</span></span>
                            </div>
                            <div class="w-full h-2 rounded-full bg-slate-800 overflow-hidden flex">
                                <div id="simBarQueja" class="bg-tm-red h-full" style="width:95%"></div>
                                <div id="simBarElogio" class="bg-tm-teal h-full" style="width:5%"></div>
                            </div>
                        </div>

                        <div class="h-36 w-full"><canvas id="simWaterfallChart"></canvas></div>
                    </div>
                </div>
            `;
        }

        // -------------------------------------------------------------
        // ESTACIÓN 7: PORTAL DESTINO (DGP, BETO & CIERRE MLOPS)
        // -------------------------------------------------------------
        if (idx === 7) {
            content += `
                <!-- Texto explicativo didáctico antes de las diapositivas -->
                <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-lg text-xs sm:text-sm text-slate-200 space-y-2 leading-relaxed">
                    <div class="flex items-center space-x-2 text-tm-yellow font-heading font-bold text-xs uppercase tracking-wider">
                        <i class="fa-solid fa-rocket text-tm-amber text-sm"></i>
                        <span>Pruebas de Estrés con Datos Simulados (DGP), Límites y el Futuro con BETO</span>
                    </div>
                    <p>
                        Llegamos a la meta de la Ruta S23. Para que este proyecto alcance una categoría de clase mundial y supere las expectativas de la rúbrica de evaluación, no podíamos limitarnos a probar el modelo en condiciones ideales. Diseñamos un <strong>Proceso Generador de Datos Sintéticos (DGP Log-Normal + Léxico Bogotano)</strong> para someter al modelo a pruebas de esfuerzo extremas fuera de distribución: jerga bogotana no vista, dobles negaciones confusas y escenarios de paro con un 95% de quejas. El modelo demostró su resiliencia logrando un <strong>100% de Recall en quejas críticas</strong>. Asimismo, con honestidad científica, reconocemos sus limitaciones actuales (como la ausencia de análisis de contexto profundo para el sarcasmo) y trazamos la hoja de ruta hacia modelos de lenguaje basados en Transformers (<strong>BETO / RoBERTa</strong>) y despliegue continuo con <strong>MLOps</strong>.
                    </p>
                    <div class="text-[11px] font-mono text-slate-400 pt-1.5 border-t border-slate-800 flex items-center space-x-2">
                        <i class="fa-solid fa-chart-line text-tm-teal"></i>
                        <span><strong>Guía de visualización:</strong> En la gráfica interactiva inferior puedes explorar el desempeño del modelo en cada uno de los 3 escenarios sintéticos de estrés OOD, y revisar la comparativa técnica entre la arquitectura actual y la futura implementación en la nube.</span>
                    </div>
                </div>

                <!-- Gráfica Dinámica de Estrés DGP y Proyección MLOps -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
                    <div class="lg:col-span-7 p-4 rounded-2xl bg-slate-900 border border-white/10 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-heading font-bold text-white">🧪 Resiliencia ante Simulación DGP (Stress-Testing OOD)</span>
                            <span class="font-mono text-emerald-400 font-bold">100% Recall de Quejas</span>
                        </div>
                        <div class="h-56 w-full"><canvas id="chartStress"></canvas></div>
                        <p class="text-[11px] text-slate-400 bg-slate-950 p-2 rounded-xl border border-slate-800">
                            <strong>Prueba de Estrés:</strong> El modelo nunca dejó pasar una queja real bajo jerga extrema bogotana, doble negación o colapso distrital al 95%.
                        </p>
                    </div>

                    <div class="lg:col-span-5 space-y-2.5 text-xs">
                        <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                            <strong class="text-tm-red block mb-1">⚖️ Limitaciones Honestas BoW:</strong>
                            <p class="text-slate-300">Incapaz de entender sarcasmo complejo e insensible al orden gramatical más allá de bi-gramas.</p>
                        </div>
                        <div class="p-3 rounded-xl bg-slate-900 border border-white/10">
                            <strong class="text-tm-teal block mb-1">🚀 Próximo Nivel: BETO & MLOps:</strong>
                            <p class="text-slate-300">Fine-tuning de BETO (Spanish BERT) con atención contextual + canalizaciones en Docker con monitoreo de Drift (Evidently AI).</p>
                        </div>
                    </div>
                </div>
            `;
        }

        return content;
    }

    // Inicializa la gráfica interactiva correspondiente a la estación abierta
    function inicializarGraficasDinamicas(idx) {
        if (idx === 1) {
            ChartsModule.renderRatingsChart('chartRatings');
            ChartsModule.renderLengthDistributionChart('chartLengths');
        } else if (idx === 2) {
            ChartsModule.renderCVComparisonChart('chartCVModels');
        } else if (idx === 3) {
            ChartsModule.renderROCChart('chartROC');
            ChartsModule.renderCalibrationChart('chartCalibration');
        } else if (idx === 4) {
            ChartsModule.renderCostCurveChart('chartCostCurve');
            SimulatorModule.inicializarSliderCosto();
        } else if (idx === 5) {
            ChartsModule.renderSHAPImportanceChart('chartSHAP');
            ChartsModule.renderErrorBreakdownChart('chartErrors');
        } else if (idx === 6) {
            SimulatorModule.inicializarSimulador();
        } else if (idx === 7) {
            ChartsModule.renderStressTestChart('chartStress');
        }
    }

    return {
        STATIONS,
        OBSTACLES,
        abrirEstacion,
        cerrarEstacion,
        navegarEstacion,
        visitedStations
    };
})();
