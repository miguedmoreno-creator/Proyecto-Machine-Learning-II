/**
 * ==============================================================================
 * MOTOR 2D DEL JUEGO Y CONTROLADOR DE LA TRAVESÍA — RUTA S23 TRANSMILENIO
 * Maneja físicas del bus biarticulado, cámara suave, HUD interactivo,
 * detección de estaciones, obstáculos y controles de teclado/táctil.
 * ==============================================================================
 */

window.GameModule = (function () {
    // Estado de movimiento y físicas del bus
    let busX = 160;
    let busY = 52; // porcentaje vertical dentro de la troncal (47% - 73%)
    let vx = 0;
    let vy = 0;
    let targetAutoX = null;
    let currentNearestStation = null;

    const keys = {
        left: false,
        right: false,
        up: false,
        down: false
    };

    /**
     * Construye los elementos del mundo: fondo urbano, estaciones y obstáculos.
     */
    function construirMundo() {
        const stations = window.StationsModule ? window.StationsModule.STATIONS : [];
        const obstacles = window.StationsModule ? window.StationsModule.OBSTACLES : [];

        // 1. Capa de Escenografía Urbana (Edificios bogotanos y árboles)
        const scenery = document.getElementById('sceneryLayer');
        if (scenery) {
            let sceneryHTML = '';
            for (let x = 100; x < 5700; x += 240) {
                const h = 70 + (x * 19) % 90;
                sceneryHTML += `
                    <div style="position:absolute; left:${x}px; bottom:62%; width:72px; height:${h}px; background:#0F172A; border:1px solid #1E293B; border-bottom:none;">
                        <div class="grid grid-cols-3 gap-1.5 p-2">
                            <div class="h-2 bg-amber-300/30 rounded-sm"></div>
                            <div class="h-2 bg-sky-300/20 rounded-sm"></div>
                            <div class="h-2 bg-amber-300/40 rounded-sm"></div>
                            <div class="h-2 bg-slate-700 rounded-sm"></div>
                            <div class="h-2 bg-amber-300/30 rounded-sm"></div>
                            <div class="h-2 bg-sky-300/30 rounded-sm"></div>
                        </div>
                    </div>
                    <div style="position:absolute; left:${x + 105}px; top:39%; font-size:20px;">🌳</div>
                    <div style="position:absolute; left:${x + 130}px; top:84%; font-size:18px;">🌲</div>
                `;
            }
            scenery.innerHTML = sceneryHTML;
        }

        // 2. Renderizar Estaciones de TransMilenio
        const stLayer = document.getElementById('stationsLayer');
        if (stLayer && stations.length > 0) {
            let stHTML = '';
            stations.forEach((st, idx) => {
                const isPortal = idx === 0 || idx === stations.length - 1;
                const headerBg = isPortal
                    ? 'bg-gradient-to-r from-tm-amber to-tm-red text-slate-950 font-black'
                    : 'bg-tm-red text-white font-bold';

                stHTML += `
                    <div id="stationNode_${idx}" onclick="GameModule.viajarYAbrirEstacion(${idx})" class="station-node" style="left: ${st.x - 95}px;">
                        <div id="stationPrompt_${idx}" class="station-prompt hidden mb-2 px-3 py-1 rounded-full bg-tm-yellow text-slate-950 font-mono font-extrabold text-[11px] text-center shadow-lg cursor-pointer">
                            ⚡ [ESPACIO] O CLIC AQUÍ
                        </div>
                        <div class="rounded-2xl bg-slate-900/95 border-2 ${isPortal ? 'border-tm-yellow' : 'border-slate-500'} shadow-2xl overflow-hidden hover:border-tm-amber transition">
                            <div class="${headerBg} px-3 py-1 flex items-center justify-between text-[10px] font-mono">
                                <span>${st.code}</span>
                                <span>RUTA S23</span>
                            </div>
                            <div class="p-3 flex items-center space-x-3">
                                <div class="w-10 h-10 rounded-xl bg-slate-800 border border-white/15 flex items-center justify-center text-2xl shrink-0">
                                    ${st.avatar}
                                </div>
                                <div class="leading-tight">
                                    <div class="text-xs font-heading font-extrabold text-white">${st.name}</div>
                                    <div class="text-[10px] text-tm-yellow font-mono mt-0.5">${st.character}</div>
                                </div>
                            </div>
                            <div class="grid grid-cols-4 gap-1 px-3 pb-2">
                                <div class="h-2.5 rounded-sm bg-sky-400/20 border border-sky-300/30"></div>
                                <div class="h-2.5 rounded-sm bg-sky-400/20 border border-sky-300/30"></div>
                                <div class="h-2.5 rounded-sm bg-sky-400/20 border border-sky-300/30"></div>
                                <div class="h-2.5 rounded-sm bg-sky-400/20 border border-sky-300/30"></div>
                            </div>
                        </div>
                    </div>
                `;
            });
            stLayer.innerHTML = stHTML;
        }

        // 3. Renderizar Obstáculos en la Vía (Lecciones Aprendidas)
        const obsLayer = document.getElementById('obstaclesLayer');
        if (obsLayer && obstacles.length > 0) {
            let obsHTML = '';
            obstacles.forEach((obs, idx) => {
                obsHTML += `
                    <div class="obstacle-node" style="left: ${obs.x}px; top: ${obs.yPct}%;"
                         onmouseenter="GameModule.mostrarTooltipObstaculo(event, ${idx})"
                         onmousemove="GameModule.moverTooltipObstaculo(event)"
                         onmouseleave="GameModule.ocultarTooltipObstaculo()"
                         onclick="GameModule.mostrarTooltipObstaculo(event, ${idx})">
                        <div class="flex flex-col items-center">
                            <div class="px-2.5 py-1 rounded-xl bg-slate-950/90 border-2 border-tm-amber shadow-lg flex items-center space-x-1.5">
                                <span class="text-base">${obs.icon}</span>
                                <span class="text-[10px] font-mono font-bold text-tm-yellow whitespace-nowrap">${obs.label}</span>
                            </div>
                            <div class="text-[8px] font-mono bg-black/75 text-slate-300 px-1 py-0.5 rounded mt-0.5">
                                Hover / Clic
                            </div>
                        </div>
                    </div>
                `;
            });
            obsLayer.innerHTML = obsHTML;
        }

        actualizarMapaHUD();
    }

    /**
     * Actualiza el esquema de paradas en el HUD superior.
     */
    function actualizarMapaHUD() {
        const mapContainer = document.getElementById('hudRouteMap');
        if (!mapContainer || !window.StationsModule) return;

        const stations = window.StationsModule.STATIONS;
        const visited = window.StationsModule.visitedStations;

        let html = '';
        stations.forEach((st, idx) => {
            const isNear = currentNearestStation === idx;
            const isVisited = visited.has(idx);

            const btnStyle = isNear
                ? 'bg-tm-yellow text-slate-950 border-white font-black scale-105 shadow-md'
                : (isVisited ? 'bg-tm-teal/25 text-emerald-300 border-tm-teal/50' : 'bg-slate-800/90 text-slate-300 border-slate-700 hover:bg-slate-700');

            html += `
                <button onclick="GameModule.viajarAEstacion(${idx})" class="px-2 py-0.5 rounded-lg border text-[11px] font-mono transition whitespace-nowrap flex items-center space-x-1 ${btnStyle}" title="${st.name}">
                    <span>${idx === 0 ? '🏁' : (idx === stations.length - 1 ? '🏆' : 'E' + idx)}</span>
                    <span class="hidden xl:inline">${st.name.split('(')[0].replace('Estación ', '').replace('Portal ', '')}</span>
                </button>
            `;

            if (idx < stations.length - 1) {
                html += `<div class="w-1.5 sm:w-3 h-0.5 ${isVisited ? 'bg-tm-teal' : 'bg-slate-700'} shrink-0"></div>`;
            }
        });
        mapContainer.innerHTML = html;
    }

    /**
     * Tooltips de Obstáculos
     */
    function mostrarTooltipObstaculo(e, idx) {
        if (!window.StationsModule) return;
        const obs = window.StationsModule.OBSTACLES[idx];
        const tip = document.getElementById('obstacleTooltip');
        if (!tip || !obs) return;

        document.getElementById('obsTooltipIcon').textContent = obs.icon;
        document.getElementById('obsTooltipTag').textContent = `OBSTÁCULO #${obs.id} EN LA TRONCAL`;
        document.getElementById('obsTooltipTitle').textContent = obs.title;
        document.getElementById('obsTooltipProblem').textContent = obs.problem;
        document.getElementById('obsTooltipSolution').textContent = obs.solution;

        tip.classList.remove('hidden');
        moverTooltipObstaculo(e);
    }

    function moverTooltipObstaculo(e) {
        const tip = document.getElementById('obstacleTooltip');
        if (!tip) return;
        const x = Math.min(window.innerWidth - 370, Math.max(16, e.clientX + 16));
        const y = Math.min(window.innerHeight - 230, Math.max(70, e.clientY - 110));
        tip.style.left = x + 'px';
        tip.style.top = y + 'px';
    }

    function ocultarTooltipObstaculo() {
        const tip = document.getElementById('obstacleTooltip');
        if (tip) tip.classList.add('hidden');
    }

    /**
     * Navegación a Estaciones
     */
    function viajarAEstacion(idx) {
        if (!window.StationsModule) return;
        targetAutoX = window.StationsModule.STATIONS[idx].x - 70;
    }

    function viajarYAbrirEstacion(idx) {
        if (!window.StationsModule) return;
        targetAutoX = window.StationsModule.STATIONS[idx].x - 70;
        abrirEstacionActualOIndice(idx);
    }

    function abrirEstacionActualOIndice(idx) {
        if (window.StationsModule) {
            window.StationsModule.abrirEstacion(idx);
        }
    }

    function cerrarModalEstacion() {
        if (window.StationsModule) {
            window.StationsModule.cerrarEstacion();
        }
    }

    function navegarEstacionModal(delta) {
        if (window.StationsModule) {
            window.StationsModule.navegarEstacion(delta);
        }
    }

    function iniciarExperiencia() {
        const splash = document.getElementById('splashScreen');
        if (splash) {
            splash.classList.add('hidden-splash');
        }
    }

    function mostrarSplash() {
        const splash = document.getElementById('splashScreen');
        if (splash) {
            splash.classList.remove('hidden-splash');
        }
    }

    function setTouchKey(key, state) {
        if (keys.hasOwnProperty(key)) {
            keys[key] = state;
            if (state) targetAutoX = null;
        }
    }

    /**
     * Bucle de Físicas y Animación (60 FPS)
     */
    function actualizarFisicaJuego() {
        // Modo piloto automático
        if (targetAutoX !== null) {
            const diff = targetAutoX - busX;
            if (Math.abs(diff) < 8) {
                busX = targetAutoX;
                vx = 0;
                targetAutoX = null;
            } else {
                vx = Math.sign(diff) * Math.min(12, Math.max(4, Math.abs(diff) * 0.08));
            }
        } else {
            // Conducción manual
            if (keys.right) vx = Math.min(vx + 0.65, 9.5);
            else if (keys.left) vx = Math.max(vx - 0.65, -9.5);
            else vx *= 0.86;

            if (keys.up) vy = Math.max(vy - 0.12, -0.65);
            else if (keys.down) vy = Math.min(vy + 0.12, 0.65);
            else vy *= 0.82;
        }

        if (Math.abs(vx) < 0.05) vx = 0;
        if (Math.abs(vy) < 0.02) vy = 0;

        busX = Math.max(60, Math.min(5520, busX + vx));
        busY = Math.max(47, Math.min(73, busY + vy));

        // Actualizar posición del bus
        const busEl = document.getElementById('playerBus');
        if (busEl) {
            busEl.style.left = busX + 'px';
            busEl.style.top = busY + '%';
            const tilt = vy * 4;
            busEl.style.transform = `${vx < -0.2 ? 'scaleX(-1)' : 'scaleX(1)'} rotate(${tilt}deg)`;
        }

        // Velocímetro HUD
        const speedEl = document.getElementById('speedIndicator');
        if (speedEl) {
            const kmh = Math.round(Math.abs(vx) * 6.5);
            speedEl.textContent = `${kmh} km/h`;
        }

        // Cámara horizontal con seguimiento suave
        const viewportW = window.innerWidth;
        const cameraX = Math.max(0, Math.min(5800 - viewportW, busX - viewportW * 0.35));
        const gameWorld = document.getElementById('gameWorld');
        if (gameWorld) {
            gameWorld.style.transform = `translate3d(${-cameraX}px, 0, 0)`;
        }

        // Detección de proximidad a estaciones
        if (window.StationsModule) {
            let foundNearest = null;
            let minDist = 9999;
            window.StationsModule.STATIONS.forEach((st, idx) => {
                const dist = Math.abs((busX + 70) - st.x);
                const promptEl = document.getElementById(`stationPrompt_${idx}`);
                if (dist < 175) {
                    if (promptEl) promptEl.classList.remove('hidden');
                    if (dist < minDist) {
                        minDist = dist;
                        foundNearest = idx;
                    }
                } else {
                    if (promptEl) promptEl.classList.add('hidden');
                }
            });

            if (foundNearest !== currentNearestStation) {
                currentNearestStation = foundNearest;
                const proxText = document.getElementById('proximityText');
                const busStatus = document.getElementById('busFloatingStatus');

                if (foundNearest !== null) {
                    window.StationsModule.visitedStations.add(foundNearest);
                    const st = window.StationsModule.STATIONS[foundNearest];
                    if (proxText) proxText.textContent = `${st.code}: ${st.name}`;
                    if (busStatus) busStatus.textContent = `🚏 En ${st.code} · Oprime [ESPACIO]`;
                } else {
                    if (proxText) proxText.textContent = `En tránsito por la Troncal Ruta S23...`;
                    if (busStatus) busStatus.textContent = `🚌 Ruta S23 · Conduciendo`;
                }
                actualizarMapaHUD();
            }
        }

        requestAnimationFrame(actualizarFisicaJuego);
    }

    // Manejadores de Teclado
    window.addEventListener('keydown', (e) => {
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (activeTag === 'textarea' || activeTag === 'input') return;

        if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') { keys.right = true; targetAutoX = null; }
        if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') { keys.left = true; targetAutoX = null; }
        if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') { keys.up = true; targetAutoX = null; }
        if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') { keys.down = true; targetAutoX = null; }

        if (e.code === 'Space') {
            e.preventDefault();
            const modal = document.getElementById('stationModal');
            const modalOpen = modal && !modal.classList.contains('hidden');
            if (modalOpen) {
                cerrarModalEstacion();
            } else if (currentNearestStation !== null) {
                abrirEstacionActualOIndice(currentNearestStation);
            }
        }
        if (e.key === 'Escape') {
            cerrarModalEstacion();
        }
    });

    window.addEventListener('keyup', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.right = false;
        if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.left = false;
        if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') keys.up = false;
        if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') keys.down = false;
    });

    // Inicialización al cargar la ventana
    window.addEventListener('DOMContentLoaded', () => {
        construirMundo();
        requestAnimationFrame(actualizarFisicaJuego);
    });

    return {
        construirMundo,
        viajarAEstacion,
        viajarYAbrirEstacion,
        abrirEstacionActualOIndice,
        cerrarModalEstacion,
        navegarEstacionModal,
        iniciarExperiencia,
        mostrarSplash,
        mostrarTooltipObstaculo,
        moverTooltipObstaculo,
        ocultarTooltipObstaculo,
        setTouchKey,
        getCurrentNearestStation: () => currentNearestStation
    };
})();
