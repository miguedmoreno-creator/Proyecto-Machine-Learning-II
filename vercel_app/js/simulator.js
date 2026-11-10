/**
 * ==============================================================================
 * MOTOR DE INFERENCIA EN VIVO & SIMULADOR DE COSTOS — RUTA S23
 * Tokenización client-side, cálculo de log-odds y valores de Shapley locales.
 * ==============================================================================
 */

window.SimulatorModule = (function () {
    let simChartInstance = null;
    let currentThreshold = 0.64;

    const EJEMPLOS = {
        1: "Pesima actualizacion, el sitp nunca pasa por el paradero de la 68 y la app se queda congelada cargando el mapa de rutas",
        2: "Me descontaron el saldo de la tarjeta tullave pero nunca cargo en la estacion, solucionen ese fallo pesimo servicio",
        3: "Excelente aplicacion, me sirve muchisimo para planear mi viaje y saber a que hora pasa exactamente el bus genial",
        4: "Muy buena herramienta, rapida facil de usar y me ahorra mucho tiempo en transmilenio la recomiendo totalmente gracias",
        5: "Muchas gracias por dejarme botado esperando 45 minutos bajo la lluvia en la caracas, que maravilla de sistema"
    };

    function normalizarTexto(txt) {
        return (txt || "").toLowerCase()
            .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
            .replace(/https?:\/\/\S+|www\.\S+/g, " ")
            .replace(/[^a-z0-9\s]/g, " ")
            .replace(/\s+/g, " ").trim();
    }

    function cargarEjemplo(id) {
        const input = document.getElementById('simInputText');
        if (input && EJEMPLOS[id]) {
            input.value = EJEMPLOS[id];
            ejecutarInferencia();
        }
    }

    function ejecutarInferencia() {
        const input = document.getElementById('simInputText');
        if (!input) return;

        const raw = input.value;
        const clean = normalizarTexto(raw);
        const words = clean.split(' ').filter(Boolean);
        const rawTokens = raw.trim().split(/\s+/);

        const vocab = window.MASTER_DATA.client_inference_model.vocabulary;
        const intercept = window.MASTER_DATA.client_inference_model.intercept;

        let logOdds = intercept;
        let contribs = [];
        let pillsHTML = [];

        for (let i = 0; i < words.length; i++) {
            const w = words[i];
            let weight = vocab[w] || 0;
            if (i < words.length - 1) {
                const bg = words[i] + " " + words[i + 1];
                if (vocab[bg]) weight += vocab[bg];
            }
            logOdds += weight;
            if (Math.abs(weight) > 0.01) contribs.push({ word: w, weight });

            const disp = rawTokens[i] || w;
            if (weight > 0.1) {
                pillsHTML.push(`<span class="px-2 py-0.5 rounded-lg bg-emerald-500/20 border border-emerald-500/50 text-emerald-300 font-bold" title="Aporte SHAP: +${weight.toFixed(2)}">${disp} <small class="font-mono text-[10px]">+${weight.toFixed(2)}</small></span>`);
            } else if (weight < -0.1) {
                pillsHTML.push(`<span class="px-2 py-0.5 rounded-lg bg-red-500/20 border border-red-500/50 text-red-300 font-bold" title="Aporte SHAP: ${weight.toFixed(2)}">${disp} <small class="font-mono text-[10px]">${weight.toFixed(2)}</small></span>`);
            } else {
                pillsHTML.push(`<span class="text-slate-400 px-1">${disp}</span>`);
            }
        }

        const wordsContainer = document.getElementById('simHighlightedWords');
        if (wordsContainer) {
            wordsContainer.innerHTML = pillsHTML.length ? pillsHTML.join(' ') : '<span class="text-slate-500 italic text-xs">Escribe una reseña...</span>';
        }

        const probElogio = 1 / (1 + Math.exp(-logOdds));
        const probQueja = 1 - probElogio;

        const probQuejaEl = document.getElementById('simProbQueja');
        const probElogioEl = document.getElementById('simProbElogio');
        const barQuejaEl = document.getElementById('simBarQueja');
        const barElogioEl = document.getElementById('simBarElogio');

        if (probQuejaEl) probQuejaEl.textContent = (probQueja * 100).toFixed(1) + '%';
        if (probElogioEl) probElogioEl.textContent = (probElogio * 100).toFixed(1) + '%';
        if (barQuejaEl) barQuejaEl.style.width = (probQueja * 100) + '%';
        if (barElogioEl) barElogioEl.style.width = (probElogio * 100) + '%';

        const banner = document.getElementById('simStatusBanner');
        const icon = document.getElementById('simStatusIcon');
        const sub = document.getElementById('simStatusSub');
        const main = document.getElementById('simStatusMain');

        if (banner && icon && sub && main) {
            if (probElogio >= currentThreshold) {
                banner.className = "p-3.5 rounded-xl border flex items-center space-x-3 bg-emerald-950/60 border-emerald-600 text-white";
                icon.textContent = "✅";
                sub.className = "text-[10px] font-mono uppercase tracking-wider text-emerald-300 block";
                sub.textContent = `Servicio Conforme · P(Elogio) ≥ ${currentThreshold.toFixed(2)}`;
                main.textContent = "ELOGIO / SATISFACCIÓN CIUDADANA";
            } else {
                banner.className = "p-3.5 rounded-xl border flex items-center space-x-3 bg-red-950/60 border-red-600 text-white";
                icon.textContent = "🚨";
                sub.className = "text-[10px] font-mono uppercase tracking-wider text-red-300 block";
                sub.textContent = `Alerta Operativa · P(Elogio) < ${currentThreshold.toFixed(2)}`;
                main.textContent = "QUEJA / RECLAMO OPERATIVO";
            }
        }

        // Gráfico Waterfall
        const canvas = document.getElementById('simWaterfallChart');
        if (!canvas) return;
        if (simChartInstance) simChartInstance.destroy();

        contribs.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
        const topC = contribs.slice(0, 6);

        simChartInstance = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: topC.length ? topC.map(c => c.word) : ['Sin palabras clave'],
                datasets: [{
                    label: 'Aporte SHAP (Log-Odds)',
                    data: topC.length ? topC.map(c => c.weight) : [0],
                    backgroundColor: topC.map(c => c.weight > 0 ? '#10B981' : '#EF4444'),
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#94A3B8', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.06)' } },
                    y: { ticks: { color: '#F8FAFC', font: { weight: 'bold', size: 10 } }, grid: { display: false } }
                }
            }
        });
    }

    // Slider interactivo de la Estación 4
    function inicializarSliderCosto() {
        const slider = document.getElementById('st4ThresholdSlider');
        if (!slider) return;

        slider.addEventListener('input', (e) => {
            const p = parseFloat(e.target.value);
            currentThreshold = p;
            const sliderVal = document.getElementById('st4SliderVal');
            if (sliderVal) sliderVal.textContent = p.toFixed(2);

            const delta = (p - 0.64) * 100;
            const fp = Math.max(4, Math.min(185, Math.round(17 - delta * 0.75)));
            const fn = Math.max(12, Math.min(235, Math.round(92 + delta * 0.95)));
            const tn = 981 - fp;
            const tp = 237 - fn;
            const cost = 20 * fp + 5 * fn;

            const tnEl = document.getElementById('st4TN');
            const fpEl = document.getElementById('st4FP');
            const fnEl = document.getElementById('st4FN');
            const tpEl = document.getElementById('st4TP');
            const costEl = document.getElementById('st4Cost');
            const recEl = document.getElementById('st4Rec0');
            const fprEl = document.getElementById('st4FPR');

            if (tnEl) tnEl.textContent = tn;
            if (fpEl) fpEl.textContent = fp;
            if (fnEl) fnEl.textContent = fn;
            if (tpEl) tpEl.textContent = tp;
            if (costEl) costEl.textContent = cost.toLocaleString();
            if (recEl) recEl.textContent = ((tn / 981) * 100).toFixed(2) + '%';
            if (fprEl) fprEl.textContent = ((fp / 981) * 100).toFixed(2) + '%';
        });
    }

    function inicializarSimulador() {
        const input = document.getElementById('simInputText');
        if (input) {
            input.addEventListener('input', ejecutarInferencia);
            ejecutarInferencia();
        }
    }

    return {
        cargarEjemplo,
        ejecutarInferencia,
        inicializarSimulador,
        inicializarSliderCosto
    };
})();
