"""
=============================================================================
SUITE AUTOMATIZADA DE PRUEBAS DE REPRODUCIBILIDAD CIENTÍFICA Y REGRESIÓN
Proyecto: SentiTransMilenio | Machine Learning II (Universidad Externado)
Autores: Santiago Sandoval Cortés & Miguel David Moreno Lozano
Docente: Prof. Wilmer Pineda
=============================================================================
"""

import os
import sys
import json
import re
import unicodedata
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# UTILIDADES DE RESOLUCIÓN DE RUTAS
# ---------------------------------------------------------------------------
def get_repo_root():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(test_dir, '..'))

def find_dataset():
    repo_root = get_repo_root()
    candidates = [
        os.path.join(repo_root, 'dataset', 'Datos obtenidos.csv'),
        os.path.join(repo_root, 'dataset', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(repo_root, '..', 'datasets', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(repo_root, '..', 'repo_segunda_entrega', 'dataset', 'Datos obtenidos.csv'),
        os.path.join(os.getcwd(), 'dataset', 'Datos obtenidos.csv'),
        os.path.join(os.getcwd(), 'Proyecto', 'datasets', 'reseñas_transmiapp_bogota.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Dataset no encontrado en las rutas: {candidates}")

def find_master_payload():
    repo_root = get_repo_root()
    candidates = [
        os.path.join(repo_root, 'dataset', 'data_payload_master.json'),
        os.path.join(repo_root, '..', 'entrega_final', 'data_payload_master.json'),
        os.path.join(os.getcwd(), 'dataset', 'data_payload_master.json'),
        os.path.join(os.getcwd(), 'Proyecto', 'entrega_final', 'data_payload_master.json'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"data_payload_master.json no encontrado en: {candidates}")

def preprocesar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# ---------------------------------------------------------------------------
# TEST 1: INTEGRIDAD DE DEPENDENCIAS Y ENTORNO PYTHON
# ---------------------------------------------------------------------------
def test_environment_and_packages():
    """Verifica que todas las librerías científicas requeridas estén instaladas."""
    import sklearn
    import xgboost
    import catboost
    import shap
    import matplotlib
    import seaborn
    import markdown

    assert sys.version_info >= (3, 10), "Se requiere Python >= 3.10 para reproducibilidad"
    assert hasattr(sklearn, '__version__')
    assert hasattr(xgboost, '__version__')
    assert hasattr(catboost, '__version__')
    assert hasattr(shap, '__version__')
    assert hasattr(matplotlib, '__version__')
    assert hasattr(seaborn, '__version__')
    assert hasattr(markdown, '__version__')

# ---------------------------------------------------------------------------
# TEST 2: INTEGRIDAD Y CALIDAD DEL DATASET OFICIAL
# ---------------------------------------------------------------------------
def test_dataset_integrity():
    """Valida la integridad estructural, columnas y dimensiones del corpus."""
    csv_path = find_dataset()
    df = pd.read_csv(csv_path)

    # Validar dimensiones esperadas
    assert len(df) >= 6080, f"El corpus debe contener al menos 6.080 reseñas, encontrado: {len(df)}"
    
    # Validar columnas requeridas
    expected_cols = {'contenido', 'calificacion', 'sentimiento_binario'}
    assert expected_cols.issubset(set(df.columns)), f"Faltan columnas requeridas: {expected_cols - set(df.columns)}"

    # Validar ausencia de nulos en la etiqueta objetivo
    assert df['sentimiento_binario'].isnull().sum() == 0, "Existen valores nulos en sentimiento_binario"

    # Validar rango de calificaciones (1 a 5)
    califs = set(df['calificacion'].dropna().unique())
    assert califs.issubset({1, 2, 3, 4, 5}), f"Calificaciones fuera de rango: {califs}"

    # Validar mapeo determinista de etiquetas (1-2 -> 0 Queja, 4-5 -> 1 Elogio)
    sub = df[df['calificacion'].isin([1, 2])]
    assert (sub['sentimiento_binario'] == 0).all(), "Las calificaciones 1 y 2 deben mapearse estrictamente a 0 (Queja)"
    
    sub_pos = df[df['calificacion'].isin([4, 5])]
    assert (sub_pos['sentimiento_binario'] == 1).all(), "Las calificaciones 4 y 5 deben mapearse estrictamente a 1 (Elogio)"

# ---------------------------------------------------------------------------
# TEST 3: DETERMINISMO EN EL PREPROCESAMIENTO LÉXICO
# ---------------------------------------------------------------------------
def test_preprocessing_determinism():
    """Verifica la normalización léxica (tildes, mayúsculas, signos y espacios)."""
    raw_sample = "¡PÉSIMA actualización! El SITP nunca pasa por la Cr. 68... www.transmi.gov.co :("
    clean = preprocesar_texto(raw_sample)
    
    assert "pesima" in clean, "Debe remover tildes y convertir a minúsculas"
    assert "actualizacion" in clean
    assert "sitp nunca pasa por la cr 68" in clean
    assert "http" not in clean and "www" not in clean, "Debe remover URLs"
    assert "!" not in clean and ":" not in clean, "Debe remover signos de puntuación"
    
    # Idempotencia
    assert preprocesar_texto(clean) == clean, "La función de preprocesamiento debe ser idempotente"

# ---------------------------------------------------------------------------
# TEST 4: PARTICIÓN ESTRATIFICADA DETERMINISTA (Zero Leakage, Seed 42)
# ---------------------------------------------------------------------------
def test_train_test_split_determinism():
    """Garantiza que la partición 80/20 estratificada sea reproducible con semilla 42."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(find_dataset())
    df['contenido_limpio'] = df['contenido'].apply(preprocesar_texto)
    df = df[df['contenido_limpio'].str.len() >= 3].reset_index(drop=True)

    X = np.array(df['contenido_limpio'].tolist(), dtype=str)
    y = np.array(df['sentimiento_binario'].tolist(), dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Verificación de tamaños exactos
    assert len(X_train) == 4871, f"Tamaño de entrenamiento incorrecto: {len(X_train)} (esperado 4871)"
    assert len(X_test) == 1218, f"Tamaño de prueba incorrecto: {len(X_test)} (esperado 1218)"

    # Verificación de distribución de clases en prueba
    test_quejas = int((y_test == 0).sum())
    test_elogios = int((y_test == 1).sum())
    assert test_quejas == 981, f"Quejas en test deben ser 981, encontrado: {test_quejas}"
    assert test_elogios == 237, f"Elogios en test deben ser 237, encontrado: {test_elogios}"

# ---------------------------------------------------------------------------
# TEST 5: SCHEMA Y BENCHMARKS DEL PAYLOAD MAESTRO
# ---------------------------------------------------------------------------
def test_master_payload_schema_and_metrics():
    """Valida el esquema del archivo data_payload_master.json y sus resultados de benchmark."""
    payload_path = find_master_payload()
    with open(payload_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    # Claves de nivel superior obligatorias
    required_keys = {
        'proyecto', 'cv_metrics', 'test_metrics', 'threshold_analysis',
        'roc_curves', 'pr_curves', 'calibration_curves', 'stress_test_results',
        'lexicon_quejas', 'lexicon_elogios', 'client_inference_model', 'eda_dynamic'
    }
    assert required_keys.issubset(set(payload.keys())), f"Faltan claves en payload: {required_keys - set(payload.keys())}"

    # Validar eda_dynamic para gráficas de Estación 1
    eda = payload['eda_dynamic']
    assert 'ratings' in eda and 'lengths' in eda, "eda_dynamic debe contener 'ratings' y 'lengths'"
    assert eda['ratings']['counts'] == [3996, 910, 335, 852], f"Conteo de calificaciones inesperado: {eda['ratings']['counts']}"

    # Validar que los 11 modelos evaluados estén registrados
    model_count = len(payload['cv_metrics'])
    assert model_count >= 11, f"Deben registrarse 11 modelos, encontrados: {model_count}"

    # Validar métricas del modelo ganador (Bagging Linear SVC)
    winner_test = payload['test_metrics']['Bagging Linear SVC']
    assert winner_test['Test_WeightedF1'] >= 0.89, f"Weighted F1 de Bagging SVC debe ser >= 0.89, fue: {winner_test['Test_WeightedF1']}"
    assert winner_test['Test_ROCAUC'] >= 0.92, f"ROC-AUC de Bagging SVC debe ser >= 0.92, fue: {winner_test['Test_ROCAUC']}"
    assert winner_test['Recall_Quejas_0'] >= 0.97, f"Recall de quejas debe ser >= 0.97, fue: {winner_test['Recall_Quejas_0']}"
    assert winner_test['Test_BrierScore'] <= 0.08, f"Brier score debe ser <= 0.08, fue: {winner_test['Test_BrierScore']}"

    # Validar optimización de umbral de costo
    opt_threshold = payload['threshold_analysis']['Bagging Linear SVC']['optimal_threshold']
    assert 0.60 <= opt_threshold <= 0.70, f"Umbral óptimo debe estar entre 0.60 y 0.70, fue: {opt_threshold}"
    assert winner_test['Costo_Test_pStar'] < winner_test['Costo_Test_p050'], "El costo en p* debe ser estrictamente menor que en p=0.50"

# ---------------------------------------------------------------------------
# TEST 6: MOTOR DE INFERENCIA CLIENT-SIDE Y SHAP LOCAL
# ---------------------------------------------------------------------------
def test_client_inference_math_and_shap():
    """Verifica que el modelo ligero client-side reproduzca las probabilidades y SHAP."""
    payload_path = find_master_payload()
    with open(payload_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    model = payload['client_inference_model']
    intercept = model['intercept']
    vocab = model['vocabulary']
    opt_threshold = model['optimal_threshold']

    def inferir(texto):
        clean = preprocesar_texto(texto)
        words = clean.split()
        log_odds = intercept
        for i, w in enumerate(words):
            log_odds += vocab.get(w, 0.0)
            if i < len(words) - 1:
                bg = f"{words[i]} {words[i+1]}"
                log_odds += vocab.get(bg, 0.0)
        prob_elogio = 1.0 / (1.0 + np.exp(-log_odds))
        return prob_elogio

    # Caso 1: Queja severa
    prob_queja_severa = inferir("Pésima actualización, nunca pasa la ruta y la app no sirve pésimo servicio")
    assert prob_queja_severa < 0.15, f"Queja severa debe tener P(Elogio) < 0.15, dio: {prob_queja_severa:.4f}"
    assert prob_queja_severa < opt_threshold, "Debe clasificarse como QUEJA bajo el umbral óptimo"

    # Caso 2: Elogio claro
    prob_elogio_claro = inferir("Excelente aplicación, muy buena herramienta, me ayuda a ahorrar tiempo gracias genial")
    assert prob_elogio_claro > 0.85, f"Elogio claro debe tener P(Elogio) > 0.85, dio: {prob_elogio_claro:.4f}"
    assert prob_elogio_claro >= opt_threshold, "Debe clasificarse como ELOGIO bajo el umbral óptimo"

# ---------------------------------------------------------------------------
# TEST 7: ESTRUCTURA MODULAR DE LA APLICACIÓN WEB VERCEL
# ---------------------------------------------------------------------------
def test_modular_web_app_architecture():
    """Valida que la app de Vercel mantenga una arquitectura limpia y modular sin archivos monolíticos."""
    repo_root = get_repo_root()
    vercel_dir = os.path.join(repo_root, 'vercel_app')

    # Archivos requeridos
    required_files = [
        'index.html',
        'vercel.json',
        os.path.join('css', 'style.css'),
        os.path.join('js', 'data.js'),
        os.path.join('js', 'charts.js'),
        os.path.join('js', 'simulator.js'),
        os.path.join('js', 'stations.js'),
        os.path.join('js', 'game.js'),
    ]

    for rf in required_files:
        full_path = os.path.join(vercel_dir, rf)
        assert os.path.exists(full_path), f"Falta archivo modular obligatorio: {rf}"

    # Validar que index.html sea conciso (< 400 líneas)
    index_path = os.path.join(vercel_dir, 'index.html')
    with open(index_path, 'r', encoding='utf-8') as f:
        index_lines = f.readlines()
    assert len(index_lines) < 400, f"index.html debe ser esqueleto modular (< 400 líneas), tiene: {len(index_lines)}"

    # Validar que stations.js NO contenga imágenes estáticas <img src="graphs/
    stations_path = os.path.join(vercel_dir, 'js', 'stations.js')
    with open(stations_path, 'r', encoding='utf-8') as f:
        stations_code = f.read()
    assert '<img' not in stations_code, "stations.js no debe contener imágenes estáticas; todas las gráficas deben ser dinámicas con Chart.js"

    # Validar presencia de canvas para gráficas interactivas
    expected_canvas_ids = [
        'chartRatings', 'chartLengths', 'chartCVModels', 'chartROC',
        'chartCalibration', 'chartCostCurve', 'chartSHAP', 'chartErrors',
        'chartStress', 'simWaterfallChart'
    ]
    for cid in expected_canvas_ids:
        assert f'id="{cid}"' in stations_code, f"Falta canvas interactivo con ID: {cid}"

    # Validar que el botón de inicio del Splash esté presente y use clase compacta
    index_content = "".join(index_lines)
    assert 'splash-modal-card' in index_content, "index.html debe usar la clase compacta .splash-modal-card"
    assert 'VALIDAR PASAJE E INICIAR RUTA S23' in index_content, "El botón de inicio debe estar presente en el Splash screen"

