/**
 * ==============================================================================
 * MÓDULO DE GRÁFICAS INTERACTIVAS DINÁMICAS (CHART.JS) — RUTA S23
 * 100% Canvas interactivo: tooltips, leyendas interactivas y animación fluida.
 * ==============================================================================
 */

window.ChartsModule = (function () {
    const activeCharts = {};

    function destroyChart(id) {
        if (activeCharts[id]) {
            activeCharts[id].destroy();
            delete activeCharts[id];
        }
    }

    // Estilo común para ejes oscuros
    const commonScales = {
        x: {
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { color: '#94A3B8', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { color: '#94A3B8', font: { family: 'JetBrains Mono', size: 10 } }
        }
    };

    // 1. ESTACIÓN 1: Distribución de Calificaciones (1★ a 5★)
    function renderRatingsChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = window.MASTER_DATA.eda_dynamic.ratings;
        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Número de Reseñas',
                    data: data.counts,
                    backgroundColor: ['#D62828', '#F77F00', '#2A9D8F', '#10B981'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                const count = ctx.raw;
                                const pct = data.percentages[ctx.dataIndex];
                                return ` ${count.toLocaleString()} reseñas (${pct}%)`;
                            }
                        }
                    }
                },
                scales: commonScales
            }
        });
    }

    // 2. ESTACIÓN 1: Asimetría de Longitud de Textos (Quejas vs Elogios)
    function renderLengthDistributionChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = window.MASTER_DATA.eda_dynamic.lengths;
        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: data.bins.map(b => b + ' pal.'),
                datasets: [
                    {
                        label: 'Quejas (Mediana: 18 palabras)',
                        data: data.hist_quejas,
                        backgroundColor: 'rgba(214, 40, 40, 0.8)',
                        borderRadius: 4
                    },
                    {
                        label: 'Elogios (Mediana: 11 palabras)',
                        data: data.hist_elogios,
                        backgroundColor: 'rgba(42, 157, 143, 0.8)',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#E2E8F0', font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            title: (items) => `Longitud: ${items[0].label}`,
                            label: (ctx) => ` ${ctx.dataset.label.split('(')[0]}: ${ctx.raw} opiniones`
                        }
                    }
                },
                scales: commonScales
            }
        });
    }

    // 3. ESTACIÓN 2: Comparativa de Validación Cruzada (Train 80%)
    function renderCVComparisonChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const cvMetrics = window.MASTER_DATA.cv_metrics;
        const models = Object.keys(cvMetrics);
        const balAccs = models.map(m => (cvMetrics[m].CV_BalancedAcc_mean * 100).toFixed(1));
        const f1s = models.map(m => (cvMetrics[m].CV_WeightedF1_mean * 100).toFixed(1));
        const aucs = models.map(m => (cvMetrics[m].CV_ROCAUC_mean * 100).toFixed(1));

        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: models,
                datasets: [
                    { label: 'Balanced Acc (%)', data: balAccs, backgroundColor: '#38BDF8', borderRadius: 4 },
                    { label: 'Weighted F1 (%)', data: f1s, backgroundColor: '#2A9D8F', borderRadius: 4 },
                    { label: 'ROC-AUC (%)', data: aucs, backgroundColor: '#F77F00', borderRadius: 4 }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#E2E8F0', font: { size: 10 } } }
                },
                scales: {
                    x: { min: 45, max: 100, ...commonScales.x },
                    y: { ...commonScales.y, ticks: { color: '#F1F5F9', font: { weight: 'bold', size: 10 } } }
                }
            }
        });
    }

    // 4. ESTACIÓN 3: Curvas ROC en Test (Hold-out 20%)
    function renderROCChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const palette = {
            'Bagging Linear SVC': '#F77F00',
            'Stacking (SVC+CatB+RF)': '#A855F7',
            'Regresión Logística': '#38BDF8',
            'CatBoost': '#EF4444',
            'Multinomial NB': '#10B981',
            'Dummy (Mayoría)': '#64748B'
        };

        const datasets = Object.entries(window.MASTER_DATA.roc_curves).map(([name, pts]) => ({
            label: `${name} (AUC: ${window.MASTER_DATA.test_metrics[name]?.Test_ROCAUC.toFixed(3) || '0.50'})`,
            data: pts.map(p => ({ x: p.fpr, y: p.tpr })),
            borderColor: palette[name] || '#94A3B8',
            borderWidth: name === 'Bagging Linear SVC' ? 3 : 2,
            showLine: true,
            pointRadius: 0
        }));

        datasets.push({
            label: 'Línea Base Aleatoria (AUC = 0.50)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#475569',
            borderDash: [5, 5],
            borderWidth: 1.5,
            showLine: true,
            pointRadius: 0
        });

        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#E2E8F0', font: { size: 10 } } }
                },
                scales: {
                    x: { title: { display: true, text: 'FPR (1 - Especificidad)', color: '#94A3B8' }, min: 0, max: 1, ...commonScales.x },
                    y: { title: { display: true, text: 'TPR (Sensibilidad)', color: '#94A3B8' }, min: 0, max: 1.02, ...commonScales.y }
                }
            }
        });
    }

    // 5. ESTACIÓN 3: Calibración de Probabilidades de Platt (Brier Score)
    function renderCalibrationChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const palette = {
            'Bagging Linear SVC': '#F77F00',
            'Stacking (SVC+CatB+RF)': '#A855F7',
            'Regresión Logística': '#38BDF8',
            'CatBoost': '#EF4444'
        };

        const datasets = Object.entries(window.MASTER_DATA.calibration_curves).map(([name, pts]) => ({
            label: `${name} (Brier: ${window.MASTER_DATA.test_metrics[name]?.Test_BrierScore.toFixed(4)})`,
            data: pts.map(p => ({ x: p.prob_pred, y: p.prob_true })),
            borderColor: palette[name] || '#94A3B8',
            borderWidth: 2.5,
            showLine: true,
            pointRadius: 3.5
        }));

        datasets.push({
            label: 'Calibración Ideal (y = x)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#64748B',
            borderDash: [5, 5],
            borderWidth: 1.5,
            showLine: true,
            pointRadius: 0
        });

        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#E2E8F0', font: { size: 10 } } }
                },
                scales: {
                    x: { title: { display: true, text: 'Probabilidad Predicha Media', color: '#94A3B8' }, min: 0, max: 1, ...commonScales.x },
                    y: { title: { display: true, text: 'Fracción Real de Elogios', color: '#94A3B8' }, min: 0, max: 1.02, ...commonScales.y }
                }
            }
        });
    }

    // 6. ESTACIÓN 4: Curva de Costo Asimétrico vs Umbral p
    function renderCostCurveChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const dataSVC = window.MASTER_DATA.threshold_analysis['Bagging Linear SVC'];
        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: dataSVC.thresholds,
                datasets: [
                    {
                        label: 'Costo C(p) = 20·FP + 5·FN (Mínimo en p*=0.64)',
                        data: dataSVC.costs,
                        borderColor: '#F77F00',
                        backgroundColor: 'rgba(247, 127, 0, 0.1)',
                        fill: true,
                        tension: 0.2,
                        borderWidth: 2.5
                    },
                    {
                        label: 'Weighted F1-Score (%)',
                        data: dataSVC.f1_scores.map(f => f * 100),
                        borderColor: '#2A9D8F',
                        borderDash: [4, 4],
                        borderWidth: 2,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#E2E8F0', font: { size: 10 } } }
                },
                scales: {
                    x: { title: { display: true, text: 'Umbral de Decisión (p)', color: '#94A3B8' }, ...commonScales.x },
                    y: { title: { display: true, text: 'Costo Ponderado', color: '#F77F00' }, ...commonScales.y },
                    y1: { position: 'right', min: 65, max: 100, ticks: { color: '#2A9D8F' }, grid: { display: false } }
                }
            }
        });
    }

    // 7. ESTACIÓN 5: Top Términos Determinantes SHAP
    function renderSHAPImportanceChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const quejas = window.MASTER_DATA.lexicon_quejas.slice(0, 8);
        const elogios = window.MASTER_DATA.lexicon_elogios.slice(0, 8);

        const terms = [...quejas.map(q => q.termino), ...elogios.map(e => e.termino)];
        const weights = [...quejas.map(q => -q.peso), ...elogios.map(e => e.peso)];
        const colors = [...quejas.map(() => '#EF4444'), ...elogios.map(() => '#10B981')];

        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: terms,
                datasets: [{
                    label: 'Peso / Impacto en Log-Odds',
                    data: weights,
                    backgroundColor: colors,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` Impacto: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(2)} (${ctx.raw < 0 ? 'Queja' : 'Elogio'})`
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: '← Más Queja | Más Elogio →', color: '#94A3B8' }, ...commonScales.x },
                    y: { ...commonScales.y, ticks: { color: '#F8FAFC', font: { weight: 'bold', size: 10 } } }
                }
            }
        });
    }

    // 8. ESTACIÓN 5: Diagnóstico Forense de Predicciones en Test
    function renderErrorBreakdownChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Quejas Reales Detectadas (TN)', 'Elogios Confirmados (TP)', 'Falsos Negativos (FN - Brevedad)', 'Falsos Positivos (FP - Sarcasmo)'],
                datasets: [{
                    data: [956, 177, 60, 25],
                    backgroundColor: ['#2A9D8F', '#10B981', '#F59E0B', '#EF4444'],
                    borderWidth: 2,
                    borderColor: '#0F172A'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#E2E8F0', font: { size: 9 }, boxWidth: 12 } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.label.split('(')[0]}: ${ctx.raw} casos (${((ctx.raw / 1218) * 100).toFixed(1)}%)`
                        }
                    }
                }
            }
        });
    }

    // 9. ESTACIÓN 7: Resiliencia ante Pruebas de Estrés Sintéticas (DGP OOD)
    function renderStressTestChart(canvasId) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = window.MASTER_DATA.stress_test_results;
        const labels = Object.keys(data);
        const recalls = labels.map(k => (data[k].Recall_Quejas * 100).toFixed(1));
        const balAccs = labels.map(k => (data[k].Balanced_Accuracy * 100).toFixed(1));
        const aucs = labels.map(k => (data[k].ROC_AUC * 100).toFixed(1));

        activeCharts[canvasId] = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels.map(l => l.replace(' / Complejidad', '')),
                datasets: [
                    { label: 'Recall Quejas (%)', data: recalls, backgroundColor: '#EF4444', borderRadius: 5 },
                    { label: 'Balanced Acc (%)', data: balAccs, backgroundColor: '#38BDF8', borderRadius: 5 },
                    { label: 'ROC-AUC (%)', data: aucs, backgroundColor: '#10B981', borderRadius: 5 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#E2E8F0', font: { size: 10 } } }
                },
                scales: {
                    y: { min: 60, max: 105, ...commonScales.y },
                    x: { ...commonScales.x, ticks: { color: '#F1F5F9', font: { weight: 'bold', size: 10 } } }
                }
            }
        });
    }

    return {
        renderRatingsChart,
        renderLengthDistributionChart,
        renderCVComparisonChart,
        renderROCChart,
        renderCalibrationChart,
        renderCostCurveChart,
        renderSHAPImportanceChart,
        renderErrorBreakdownChart,
        renderStressTestChart,
        destroyChart
    };
})();
