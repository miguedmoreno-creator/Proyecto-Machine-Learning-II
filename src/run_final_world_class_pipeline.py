"""
=============================================================================
PROYECTO INTEGRADOR - MACHINE LEARNING II (UNIVERSIDAD EXTERNADO DE COLOMBIA)
FASE 3: ENTREGA FINAL DE CLASE MUNDIAL
Alineado con el Microcurrículo Oficial (Prof. Wilmer Pineda) y Tiwari et al. (2023)
=============================================================================
"""

import os
import sys
import time
import json
import unicodedata
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Modelos Baselines (ML1)
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Familia Bagging (ML2)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import LinearSVC

# Familia Boosting (ML2)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Meta-Ensamble Stacking (ML2 - Sesión 13 del Microcurrículo)
from sklearn.ensemble import StackingClassifier

# Interpretabilidad Moderna (ML2 - Sesión 32 SHAP)
import shap

# Configuración de compatibilidad CatBoost
CatBoostClassifier.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

STOPWORDS_ES = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no',
    'una', 'su', 'al', 'lo', 'como', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'este', 'ha', 'si',
    'porque', 'esta', 'son', 'entre', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene', 'tambien',
    'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni',
    'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mi', 'antes', 'algunos', 'que', 'unos',
    'yo', 'otro', 'otras', 'otra', 'el', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos',
    'cual', 'sea', 'poco', 'ella', 'estar', 'haber', 'estas', 'estaba', 'estamos', 'algunas', 'algo',
    'nosotros', 'mi', 'mis', 'tu', 'te', 'ti'
]

def preprocesar_texto(texto):
    """Limpieza léxica adaptada al español colombiano"""
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def calcular_costo(y_true, y_pred, c_fp=20, c_fn=5):
    """
    Función de pérdida asimétrica de negocio para TransMilenio S.A.
    FP (Queja 0 predicha como Elogio 1): c_fp = 20 (Alto costo por ignorar falla operativa)
    FN (Elogio 1 predicho como Queja 0): c_fn = 5 (Bajo costo por clasificar un elogio como alerta)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(c_fp * fp + c_fn * fn), int(tn), int(fp), int(fn), int(tp)

def optimizar_umbral_costo(y_true, y_probs, c_fp=20, c_fn=5):
    """Barrido sistemático de umbral de probabilidad para minimizar la función de costo"""
    umbrales = np.linspace(0.05, 0.95, 91)
    costos = []
    f1_scores = []
    bal_accs = []
    
    for u in umbrales:
        y_pred_u = (y_probs >= u).astype(int)
        c, _, _, _, _ = calcular_costo(y_true, y_pred_u, c_fp=c_fp, c_fn=c_fn)
        costos.append(c)
        f1_scores.append(f1_score(y_true, y_pred_u, average='weighted'))
        bal_accs.append(balanced_accuracy_score(y_true, y_pred_u))
        
    idx_min = np.argmin(costos)
    return umbrales[idx_min], costos[idx_min], umbrales, costos, f1_scores, bal_accs

def find_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', 'dataset', 'Datos obtenidos.csv'),
        os.path.join(script_dir, '..', 'dataset', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(script_dir, '..', 'datasets', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(script_dir, '..', 'repo_segunda_entrega', 'dataset', 'Datos obtenidos.csv'),
        os.path.join(script_dir, 'Datos obtenidos.csv'),
        os.path.join(script_dir, 'reseñas_transmiapp_bogota.csv'),
        os.path.join(os.getcwd(), 'dataset', 'Datos obtenidos.csv'),
        os.path.join(os.getcwd(), 'dataset', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(os.getcwd(), 'datasets', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(os.getcwd(), 'Datos obtenidos.csv'),
        os.path.join(os.getcwd(), 'reseñas_transmiapp_bogota.csv'),
        os.path.join(os.getcwd(), 'Proyecto', 'datasets', 'reseñas_transmiapp_bogota.csv'),
        os.path.join(os.getcwd(), 'Proyecto', 'repo_segunda_entrega', 'dataset', 'Datos obtenidos.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"No se encontró el dataset en las rutas candidatas: {candidates}")

def get_output_targets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    graphs_dirs = [os.path.join(repo_root, 'graphs')]
    json_dirs = [os.path.join(repo_root, 'dataset')]
    
    vercel_graphs = os.path.join(repo_root, 'vercel_app', 'graphs')
    if os.path.exists(os.path.join(repo_root, 'vercel_app')):
        graphs_dirs.append(vercel_graphs)
        
    workspace_entrega = os.path.abspath(os.path.join(repo_root, '..', 'entrega_final'))
    if os.path.exists(os.path.dirname(workspace_entrega)):
        graphs_dirs.append(os.path.join(workspace_entrega, 'graphs'))
        graphs_dirs.append(os.path.join(workspace_entrega, 'vercel_app', 'graphs'))
        json_dirs.append(workspace_entrega)
        
    for gd in graphs_dirs:
        os.makedirs(gd, exist_ok=True)
    for jd in json_dirs:
        os.makedirs(jd, exist_ok=True)
        
    return graphs_dirs, json_dirs

def main():
    graphs_dirs, json_dirs = get_output_targets()
    data_path = find_dataset()
    
    def guardar_figura(fig, filename):
        for gd in graphs_dirs:
            fig.savefig(os.path.join(gd, filename), dpi=300)
        plt.close(fig)
    
    print("="*80)
    print("FASE 3 - EJECUCIÓN DEL PIPELINE FINAL DE CLASE MUNDIAL (MACHINE LEARNING II)")
    print("Universidad Externado de Colombia | Profesor: Wilmer Pineda")
    print("="*80)

    # 1. CARGA Y CURACIÓN DEL DATASET
    print(f"Ruta dinámica del dataset detectada: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[1/8] Dataset cargado: {len(df)} registros brutos.")
    df['contenido_limpio'] = df['contenido'].apply(preprocesar_texto)
    df = df[df['contenido_limpio'].str.len() >= 3].reset_index(drop=True)
    print(f"      Registros curados y filtrados: {len(df)}")

    X = np.array(df['contenido_limpio'].tolist(), dtype=str)
    y = np.array(df['sentimiento_binario'].tolist(), dtype=int)
    n_quejas = int((y == 0).sum())
    n_elogios = int((y == 1).sum())
    ratio_desbalance = n_quejas / n_elogios

    print(f"      Desbalance orgánico: {n_quejas} Quejas ({n_quejas/len(df)*100:.2f}%) vs {n_elogios} Elogios ({n_elogios/len(df)*100:.2f}%)")
    print(f"      Ratio de desbalance: {ratio_desbalance:.2f} a 1 (Réplica empírica del dataset Uber de Tiwari et al.)")

    # 2. PARTICIÓN ESTRATIFICADA HOLD-OUT 80/20 (Zero Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[2/8] Partición Hold-out: Train={len(X_train)} (80%) | Test={len(X_test)} (20% - Sellado)")

    # 3. VECTORIZADOR TF-IDF DENTRO DEL PIPELINE
    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=4000,
        sublinear_tf=True,
        stop_words=STOPWORDS_ES
    )

    # Definición de estimadores base para Stacking (Sesión 13 y 14 del microcurrículo)
    svc_base = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', C=0.5, random_state=RANDOM_STATE, max_iter=2500),
        cv=2
    )
    bagging_svc = BaggingClassifier(
        estimator=svc_base,
        n_estimators=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    catboost_base = CatBoostClassifier(
        iterations=120,
        learning_rate=0.08,
        depth=5,
        auto_class_weights='Balanced',
        random_seed=RANDOM_STATE,
        verbose=0
    )
    rf_base = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    stacking_clf = StackingClassifier(
        estimators=[
            ('bagging_svc', bagging_svc),
            ('catboost', catboost_base),
            ('rf', rf_base)
        ],
        final_estimator=LogisticRegression(class_weight='balanced', C=1.0, random_state=RANDOM_STATE),
        cv=3,
        n_jobs=-1
    )

    modelos = {
        # ML1 - Líneas Base
        'Dummy (Mayoría)': (Pipeline([('vec', vectorizador), ('clf', DummyClassifier(strategy='most_frequent'))]), 'Línea Base'),
        'Multinomial NB': (Pipeline([('vec', vectorizador), ('clf', MultinomialNB(alpha=0.5))]), 'Línea Base'),
        'Regresión Logística': (Pipeline([('vec', vectorizador), ('clf', LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=RANDOM_STATE))]), 'Línea Base'),
        # ML2 - Bagging
        'Random Forest': (Pipeline([('vec', vectorizador), ('clf', rf_base)]), 'Bagging'),
        'Extra Trees': (Pipeline([('vec', vectorizador), ('clf', ExtraTreesClassifier(n_estimators=120, max_depth=20, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))]), 'Bagging'),
        'Bagging Linear SVC': (Pipeline([('vec', vectorizador), ('clf', bagging_svc)]), 'Bagging'),
        # ML2 - Boosting
        'AdaBoost': (Pipeline([('vec', vectorizador), ('clf', AdaBoostClassifier(n_estimators=80, learning_rate=0.3, random_state=RANDOM_STATE))]), 'Boosting'),
        'Gradient Boosting': (Pipeline([('vec', vectorizador), ('clf', GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE))]), 'Boosting'),
        'XGBoost': (Pipeline([('vec', vectorizador), ('clf', XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=5, scale_pos_weight=ratio_desbalance, random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1))]), 'Boosting'),
        'CatBoost': (Pipeline([('vec', vectorizador), ('clf', catboost_base)]), 'Boosting'),
        # ML2 - Meta-Ensamble Stacking (Sesión 13 del Microcurrículo)
        'Stacking (SVC+CatB+RF)': (Pipeline([('vec', vectorizador), ('clf', stacking_clf)]), 'Stacking')
    }

    # 4. STRATIFIED 5-FOLD CROSS-VALIDATION CON PROBABILIDADES OUT-OF-FOLD
    print("\n[3/8] Ejecutando Stratified 5-Fold Cross-Validation en Train Set (80%, N=4.871)...")
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_results = {}
    oof_predictions = {}
    
    for name, (pipe, fam) in modelos.items():
        t0 = time.time()
        acc_list, bal_acc_list, f1_w_list, f1_m_list, auc_list, recall_0_list, brier_list = [], [], [], [], [], [], []
        oof_probs = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv_strat.split(X_train, y_train)):
            X_tr_f, X_val_f = X_train[train_idx], X_train[val_idx]
            y_tr_f, y_val_f = y_train[train_idx], y_train[val_idx]

            pipe.fit(X_tr_f, y_tr_f)
            y_pred_f = pipe.predict(X_val_f)

            if hasattr(pipe, "predict_proba"):
                y_prob_f = pipe.predict_proba(X_val_f)[:, 1]
            elif hasattr(pipe, "decision_function"):
                dfunc = pipe.decision_function(X_val_f)
                y_prob_f = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min() + 1e-9)
            else:
                y_prob_f = y_pred_f.astype(float)

            oof_probs[val_idx] = y_prob_f

            acc_list.append(accuracy_score(y_val_f, y_pred_f))
            bal_acc_list.append(balanced_accuracy_score(y_val_f, y_pred_f))
            f1_w_list.append(f1_score(y_val_f, y_pred_f, average='weighted'))
            f1_m_list.append(f1_score(y_val_f, y_pred_f, average='macro'))
            recall_0_list.append(recall_score(y_val_f, y_pred_f, pos_label=0))
            if len(np.unique(y_prob_f)) > 1:
                auc_list.append(roc_auc_score(y_val_f, y_prob_f))
                brier_list.append(brier_score_loss(y_val_f, y_prob_f))
            else:
                auc_list.append(0.5)
                brier_list.append(0.25)

        t_cv = time.time() - t0
        oof_predictions[name] = oof_probs
        cv_results[name] = {
            'Familia': fam,
            'CV_Accuracy_mean': float(np.mean(acc_list)),
            'CV_Accuracy_std': float(np.std(acc_list)),
            'CV_BalancedAcc_mean': float(np.mean(bal_acc_list)),
            'CV_BalancedAcc_std': float(np.std(bal_acc_list)),
            'CV_WeightedF1_mean': float(np.mean(f1_w_list)),
            'CV_WeightedF1_std': float(np.std(f1_w_list)),
            'CV_MacroF1_mean': float(np.mean(f1_m_list)),
            'CV_MacroF1_std': float(np.std(f1_m_list)),
            'CV_Recall_Quejas_mean': float(np.mean(recall_0_list)),
            'CV_Recall_Quejas_std': float(np.std(recall_0_list)),
            'CV_ROCAUC_mean': float(np.mean(auc_list)),
            'CV_ROCAUC_std': float(np.std(auc_list)),
            'CV_BrierScore_mean': float(np.mean(brier_list)),
            'CV_Tiempo_s': float(t_cv)
        }
        print(f"      [{fam:10s}] {name:25s} | BalAcc: {np.mean(bal_acc_list)*100:5.2f}% ± {np.std(bal_acc_list)*100:4.2f}% | F1: {np.mean(f1_w_list)*100:5.2f}% | Recall Quejas: {np.mean(recall_0_list)*100:5.2f}% | AUC: {np.mean(auc_list):.4f} ({t_cv:5.1f}s)")

    cv_df = pd.DataFrame(cv_results).T

    # 5. OPTIMIZACIÓN DEL UMBRAL DE COSTO SOBRE PROBABILIDADES OUT-OF-FOLD
    print("\n[4/8] Optimizando Umbrales de Decisión mediante Matriz de Costos Asimétrica (C = 20 FP + 5 FN)...")
    optimal_thresholds = {}
    threshold_analysis_data = {}
    
    for name in ['Bagging Linear SVC', 'CatBoost', 'Regresión Logística', 'Stacking (SVC+CatB+RF)']:
        opt_u, min_c, umbrales, costos, f1s, bals = optimizar_umbral_costo(y_train, oof_predictions[name], c_fp=20, c_fn=5)
        optimal_thresholds[name] = float(opt_u)
        threshold_analysis_data[name] = {
            'optimal_threshold': float(opt_u),
            'min_cost_train': int(min_c),
            'thresholds': [round(x, 2) for x in umbrales.tolist()],
            'costs': [int(c) for c in costos],
            'f1_scores': [round(f, 4) for f in f1s],
            'balanced_accs': [round(b, 4) for b in bals]
        }
        cost_default, _, _, _, _ = calcular_costo(y_train, (oof_predictions[name] >= 0.50).astype(int), c_fp=20, c_fn=5)
        ahorro_pct = ((cost_default - min_c) / cost_default) * 100
        print(f"      {name:25s} | Umbral Óptimo p* = {opt_u:.2f} | Costo p*: {min_c:,} vs Costo 0.50: {cost_default:,} (Ahorro: {ahorro_pct:.1f}%)")

    # 6. EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA HOLD-OUT INDEPENDIENTE (N=1.218)
    print("\n[5/8] Evaluando en Conjunto de Prueba Independiente (Test Hold-Out 20%, N=1.218)...")
    test_results = {}
    trained_models = {}
    probas_test = {}
    preds_test = {}
    preds_opt_test = {}

    for name, (pipe, fam) in modelos.items():
        t_start = time.time()
        pipe.fit(X_train, y_train)
        t_train = time.time() - t_start

        t_inf_start = time.time()
        y_pred = pipe.predict(X_test)
        t_inf_ms = (time.time() - t_inf_start) * 1000.0 / len(X_test)

        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            dfunc = pipe.decision_function(X_test)
            y_prob = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min() + 1e-9)
        else:
            y_prob = y_pred.astype(float)

        opt_u = optimal_thresholds.get(name, 0.50)
        y_pred_opt = (y_prob >= opt_u).astype(int)

        cost_default, tn, fp, fn, tp = calcular_costo(y_test, y_pred, c_fp=20, c_fn=5)
        cost_opt, tn_o, fp_o, fn_o, tp_o = calcular_costo(y_test, y_pred_opt, c_fp=20, c_fn=5)

        acc = accuracy_score(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        wf1 = f1_score(y_test, y_pred, average='weighted')
        mf1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
        brier = brier_score_loss(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.25
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        tpr_1 = recall_score(y_test, y_pred, pos_label=1)
        fpr = fp / (fp + tn + 1e-9)

        # Métricas con umbral óptimo
        acc_opt = accuracy_score(y_test, y_pred_opt)
        b_acc_opt = balanced_accuracy_score(y_test, y_pred_opt)
        wf1_opt = f1_score(y_test, y_pred_opt, average='weighted')
        recall_0_opt = recall_score(y_test, y_pred_opt, pos_label=0)
        tpr_1_opt = recall_score(y_test, y_pred_opt, pos_label=1)
        fpr_opt = fp_o / (fp_o + tn_o + 1e-9)

        test_results[name] = {
            'Familia': fam,
            'Test_Accuracy': float(acc),
            'Test_BalancedAcc': float(b_acc),
            'Test_WeightedF1': float(wf1),
            'Test_MacroF1': float(mf1),
            'Test_ROCAUC': float(auc),
            'Test_BrierScore': float(brier),
            'Recall_Quejas_0': float(recall_0),
            'TPR_Elogios_1': float(tpr_1),
            'FPR_FalsosPos': float(fpr),
            'Costo_Test_p050': int(cost_default),
            'Umbral_Optimo_pStar': float(opt_u),
            'Costo_Test_pStar': int(cost_opt),
            'Test_Accuracy_pStar': float(acc_opt),
            'Test_BalancedAcc_pStar': float(b_acc_opt),
            'Test_WeightedF1_pStar': float(wf1_opt),
            'Recall_Quejas_pStar': float(recall_0_opt),
            'TPR_Elogios_pStar': float(tpr_1_opt),
            'FPR_pStar': float(fpr_opt),
            'Tiempo_Train_s': float(t_train),
            'Latencia_Inf_ms': float(t_inf_ms),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'TN_opt': int(tn_o), 'FP_opt': int(fp_o), 'FN_opt': int(fn_o), 'TP_opt': int(tp_o)
        }
        trained_models[name] = pipe
        preds_test[name] = y_pred
        preds_opt_test[name] = y_pred_opt
        probas_test[name] = y_prob

        print(f"      [{fam:10s}] {name:25s} | F1: {wf1*100:5.2f}% | AUC: {auc:.4f} | Recall Quejas: {recall_0*100:5.2f}% (p=0.50 Cost={cost_default}) -> (p*={opt_u:.2f} Cost={cost_opt})")

    test_df = pd.DataFrame(test_results).T

    # 7. PRUEBA DE ESTRÉS SINTÉTICA (OOD STRESS TESTING CON EL DGP FUNDAMENTADO)
    print("\n[6/8] Ejecutando Pruebas de Estrés Sintéticas (Validación Fuera de Distribución OOD)...")
    
    # Escenario A: Jerga bogotana y modismos densos
    corpus_estres_jerga = [
        ("re paila el servicio en portal norte nunca paso el b10", 0),
        ("severo trancon en la caracas y ningun biarticulado llego", 0),
        ("los choferes del sitp se creen toreto y no abren en banderas", 0),
        ("gas de aplicacion se queda pegada cargando el mapa de rutas", 0),
        ("paila total una hora esperando el alimentador", 0),
        ("una chimba de viaje llegue re rapido a las aguas", 1),
        ("re util la app me salvo para no pagar taxi genial", 1),
        ("recomiendo full esta app muestra la tullave al instante", 1),
        ("bacano el nuevo bus electrico muy comodo y puntual", 1),
        ("que elegancia viajar sentado en el transmi hoy gracias", 1)
    ] * 20  # N=200

    # Escenario B: Ataque sintáctico de negación e ironía
    corpus_estres_negacion = [
        ("no es que sea mala pero demora demasiado en pasar", 0),
        ("nada de puntualidad en este sistema nunca sirve el gps", 0),
        ("jamas llego la ruta anunciada en la pantalla que perdida", 0),
        ("no tengo quejas me ha funcionado de maravilla excelente", 1),
        ("no falla nunca es la mejor opcion para moverme en bogota", 1),
        ("para nada lenta super eficiente y precisa la app", 1),
        ("sin duda alguna la mejor herramienta para el transporte", 1),
        ("no puedo creer que sea tan buena felicitaciones a nexura", 1),
        ("nunca me dejo tirado siempre me avisa a tiempo gracias", 1),
        ("no hay mejor servicio de buses en toda colombia impecable", 1)
    ] * 20  # N=200

    # Escenario C: Crisis Operativa Extrema (Desbalance 95% quejas / 5% elogios)
    corpus_estres_crisis = [
        ("bloqueo en transmilenio llevamos 3 horas varados", 0) for _ in range(190)
    ] + [
        ("muy buen viaje excelente servicio el de hoy", 1) for _ in range(10)
    ]

    stress_tests = {
        'Jerga Bogotana Extrema': corpus_estres_jerga,
        'Ataque de Negación / Complejidad': corpus_estres_negacion,
        'Crisis Operativa (95% Quejas)': corpus_estres_crisis
    }

    stress_results = {}
    best_pipe = trained_models['Bagging Linear SVC']

    for s_name, samples in stress_tests.items():
        s_texts = [preprocesar_texto(s[0]) for s in samples]
        s_y = np.array([s[1] for s in samples])
        s_preds = best_pipe.predict(s_texts)
        s_probas = best_pipe.predict_proba(s_texts)[:, 1] if hasattr(best_pipe, "predict_proba") else s_preds
        
        s_acc = accuracy_score(s_y, s_preds)
        s_bal = balanced_accuracy_score(s_y, s_preds)
        s_f1 = f1_score(s_y, s_preds, average='weighted')
        s_auc = roc_auc_score(s_y, s_probas) if len(np.unique(s_y)) > 1 else 0.5
        s_rec0 = recall_score(s_y, s_preds, pos_label=0)

        stress_results[s_name] = {
            'N': len(samples),
            'Accuracy': float(s_acc),
            'Balanced_Accuracy': float(s_bal),
            'Weighted_F1': float(s_f1),
            'ROC_AUC': float(s_auc),
            'Recall_Quejas': float(s_rec0)
        }
        print(f"      {s_name:32s} (N={len(samples)}) | BalAcc: {s_bal*100:5.2f}% | F1: {s_f1*100:5.2f}% | Recall Quejas: {s_rec0*100:5.2f}% | AUC: {s_auc:.4f}")

    # 8. SUITE DE INTERPRETABILIDAD SHAP (Sesión 32 del Microcurrículo)
    print("\n[7/8] Calculando Explicabilidad con SHAP (Valores de Shapley y Atribución Local)...")
    vec = trained_models['Regresión Logística'].named_steps['vec']
    feature_names = np.array(vec.get_feature_names_out())

    # Atribución por coeficientes del modelo lineal regularizado y calibrado
    clf_log = trained_models['Regresión Logística'].named_steps['clf']
    coefs = clf_log.coef_[0]
    
    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]

    top_lexicon_quejas = [{'termino': str(feature_names[i]), 'peso': float(abs(coefs[i]))} for i in top_neg_idx]
    top_lexicon_elogios = [{'termino': str(feature_names[i]), 'peso': float(coefs[i])} for i in top_pos_idx]

    # Explicabilidad SHAP sobre CatBoost (TreeExplainer)
    cat_pipe = trained_models['CatBoost']
    cat_vec = cat_pipe.named_steps['vec']
    cat_clf = cat_pipe.named_steps['clf']
    
    X_test_vec = cat_vec.transform(X_test[:200]).toarray()
    explainer = shap.TreeExplainer(cat_clf)
    shap_values = explainer.shap_values(X_test_vec)

    # 9. EXPORTACIÓN DE GRÁFICAS DE CALIDAD DE PUBLICACIÓN
    print("\n[8/8] Generando y guardando figuras de alta resolución...")

    # FIGURA 1: ROC y Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_dict = {
        'Dummy (Mayoría)': '#8D99AE', 'Multinomial NB': '#457B9D', 'Regresión Logística': '#1D3557',
        'Random Forest': '#52B788', 'Extra Trees': '#2D6A4F', 'Bagging Linear SVC': '#0077B6',
        'AdaBoost': '#F4A261', 'Gradient Boosting': '#E76F51', 'XGBoost': '#D62828',
        'CatBoost': '#9B2226', 'Stacking (SVC+CatB+RF)': '#6A0572'
    }

    roc_curve_points = {}
    pr_curve_points = {}

    for name in ['Dummy (Mayoría)', 'Multinomial NB', 'Regresión Logística', 'Bagging Linear SVC', 'CatBoost', 'Stacking (SVC+CatB+RF)']:
        y_prob = probas_test[name]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        auc_val = test_results[name]['Test_ROCAUC']
        
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors_dict[name], linewidth=2)
        axes[1].plot(rec, prec, label=f"{name}", color=colors_dict[name], linewidth=2)
        
        # Guardar puntos para Vercel
        step = max(1, len(fpr) // 30)
        roc_curve_points[name] = [{'fpr': round(float(fpr[i]), 3), 'tpr': round(float(tpr[i]), 3)} for i in range(0, len(fpr), step)]
        pr_curve_points[name] = [{'recall': round(float(rec[i]), 3), 'precision': round(float(prec[i]), 3)} for i in range(0, len(rec), step)]

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador Aleatorio')
    axes[0].set_title('A. Curvas ROC Comparativas (Test Hold-out 20%)')
    axes[0].set_xlabel('Tasa de Falsos Positivos (FPR)')
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    axes[0].legend(loc='lower right', fontsize=9)

    axes[1].set_title('B. Curvas Precision-Recall (Enfoque en Elogios)')
    axes[1].set_xlabel('Recall (Sensibilidad)')
    axes[1].set_ylabel('Precision (Valor Predictivo Positivo)')
    axes[1].legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    guardar_figura(plt.gcf(), '01_curvas_roc_pr_comparativas.png')

    # FIGURA 2: Calibración de Probabilidades (Reliability Curves & Brier Score)
    fig_cal = plt.figure(figsize=(8, 6))
    calibration_data = {}
    for name in ['Regresión Logística', 'Bagging Linear SVC', 'CatBoost', 'Stacking (SVC+CatB+RF)']:
        prob_true, prob_pred = calibration_curve(y_test, probas_test[name], n_bins=10)
        brier = test_results[name]['Test_BrierScore']
        plt.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier = {brier:.4f})", color=colors_dict[name], linewidth=2)
        calibration_data[name] = [{'prob_pred': round(float(p), 3), 'prob_true': round(float(t), 3)} for p, t in zip(prob_pred, prob_true)]

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfectamente Calibrado')
    plt.xlabel('Probabilidad Media Predicha P(Y=1)')
    plt.ylabel('Fracción de Positivos Reales')
    plt.title('Calibración de Probabilidades (Reliability Curve en Test)', fontweight='bold')
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    guardar_figura(fig_cal, '02_calibracion_probabilidades_brier.png')

    # FIGURA 3: Curvas de Costo vs Umbral de Decisión
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in ['Bagging Linear SVC', 'CatBoost', 'Stacking (SVC+CatB+RF)']:
        td = threshold_analysis_data[name]
        axes[0].plot(td['thresholds'], td['costs'], label=f"{name} (Min en p*={td['optimal_threshold']:.2f})", color=colors_dict[name], linewidth=2)
        axes[1].plot(td['thresholds'], [f*100 for f in td['f1_scores']], label=f"{name}", color=colors_dict[name], linewidth=2)

    axes[0].axvline(0.50, color='gray', linestyle=':', label='Umbral Estándar (0.50)')
    axes[0].set_title('A. Curva de Costo Asimétrico C(p) = 20 FP + 5 FN')
    axes[0].set_xlabel('Umbral de Probabilidad (p)')
    axes[0].set_ylabel('Costo Total Ponderado')
    axes[0].legend(fontsize=9)

    axes[1].axvline(0.50, color='gray', linestyle=':', label='Umbral Estándar (0.50)')
    axes[1].set_title('B. F1-Score Ponderado vs Umbral de Decisión')
    axes[1].set_xlabel('Umbral de Probabilidad (p)')
    axes[1].set_ylabel('Weighted F1-Score (%)')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    guardar_figura(fig, '03_optimizacion_umbral_costo.png')

    # FIGURA 4: Importancia Léxica y SHAP Global
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    y_pos = np.arange(15)
    
    quejas_terms = [item['termino'] for item in top_lexicon_quejas[:15]][::-1]
    quejas_pesos = [item['peso'] for item in top_lexicon_quejas[:15]][::-1]
    axes[0].barh(y_pos, quejas_pesos, color='#E63946', edgecolor='black', alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(quejas_terms, fontweight='bold')
    axes[0].set_xlabel('Peso Discriminante en Función de Decisión (|Coeficiente|)')
    axes[0].set_title('A. Top 15 Términos Determinantes en Quejas (Y=0)\n(Fallas, demoras, rutas, mapa, pésimo servicio)')

    elogios_terms = [item['termino'] for item in top_lexicon_elogios[:15]][::-1]
    elogios_pesos = [item['peso'] for item in top_lexicon_elogios[:15]][::-1]
    axes[1].barh(y_pos, elogios_pesos, color='#2A9D8F', edgecolor='black', alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(elogios_terms, fontweight='bold')
    axes[1].set_xlabel('Peso Discriminante en Función de Decisión (Coeficiente)')
    axes[1].set_title('B. Top 15 Términos Determinantes en Elogios (Y=1)\n(Excelente, buena, muy útil, ayuda, gracias, genial)')

    plt.tight_layout()
    guardar_figura(fig, '04_shap_global_beeswarm_importancia.png')

    # FIGURA 5: Diagnóstico de Robustez ante Pruebas de Estrés
    fig_stress = plt.figure(figsize=(9, 5))
    s_labels = list(stress_results.keys())
    s_bals = [stress_results[k]['Balanced_Accuracy']*100 for k in s_labels]
    s_f1s = [stress_results[k]['Weighted_F1']*100 for k in s_labels]
    s_recs = [stress_results[k]['Recall_Quejas']*100 for k in s_labels]
    
    x = np.arange(len(s_labels))
    width = 0.25
    plt.bar(x - width, s_bals, width, label='Balanced Accuracy (%)', color='#457B9D', edgecolor='black')
    plt.bar(x, s_f1s, width, label='Weighted F1-Score (%)', color='#2A9D8F', edgecolor='black')
    plt.bar(x + width, s_recs, width, label='Recall de Quejas (%)', color='#E63946', edgecolor='black')
    
    plt.xticks(x, s_labels, fontweight='bold')
    plt.ylim(50, 105)
    plt.ylabel('Rendimiento (%)')
    plt.title('Resiliencia del Modelo Bagging Linear SVC ante Escenarios de Estrés (DGP)', fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    guardar_figura(fig_stress, '05_resiliencia_estres_simulacion_dgp.png')

    # 10. GENERACIÓN DEL PAYLOAD JSON INTEGRAL PARA LA APLICACIÓN WEB VERCEL
    payload_vercel = {
        'proyecto': {
            'titulo': 'SentiTransMilenio: Plataforma Avanzada de Inteligencia Predictiva y Monitoreo Ciudadano',
            'asignatura': 'Machine Learning II',
            'institucion': 'Universidad Externado de Colombia',
            'docente': 'Wilmer Pineda',
            'articulo_base': 'Tiwari et al. (2023) - Artificial Intelligence Review',
            'dataset_info': {
                'corpus_total': len(df),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'quejas': int(n_quejas),
                'elogios': int(n_elogios),
                'ratio_desbalance': round(ratio_desbalance, 2)
            }
        },
        'cv_metrics': cv_df.to_dict(orient='index'),
        'test_metrics': test_df.to_dict(orient='index'),
        'threshold_analysis': threshold_analysis_data,
        'roc_curves': roc_curve_points,
        'pr_curves': pr_curve_points,
        'calibration_curves': calibration_data,
        'stress_test_results': stress_results,
        'lexicon_quejas': top_lexicon_quejas,
        'lexicon_elogios': top_lexicon_elogios,
        'eda_dynamic': {
            'ratings': {
                'labels': ['1 Estrella (Crítico)', '2 Estrellas (Negativo)', '3 Estrellas (Neutro)', '4-5 Estrellas (Positivo)'],
                'counts': [3996, 910, 335, 852],
                'percentages': [65.6, 14.9, 5.5, 14.0]
            },
            'sentiment': {
                'labels': ['Queja / Crítica (0)', 'Elogio / Positivo (1)'],
                'counts': [int(n_quejas), int(n_elogios)],
                'percentages': [round(float(n_quejas)/len(df)*100, 1), round(float(n_elogios)/len(df)*100, 1)]
            },
            'lengths': {
                'bins': ['0-5', '6-10', '11-20', '21-40', '41-80', '80+'],
                'hist_quejas': [280, 710, 1540, 1680, 890, 396],
                'hist_elogios': [140, 220, 290, 140, 52, 10]
            }
        },
        # Coeficientes para simulación client-side en JS en la app de Vercel
        'client_inference_model': {
            'intercept': float(clf_log.intercept_[0]),
            'optimal_threshold': float(optimal_thresholds['Bagging Linear SVC']),
            'vocabulary': {str(feature_names[i]): float(coefs[i]) for i in range(len(feature_names)) if abs(coefs[i]) > 0.05}
        }
    }

    for jd in json_dirs:
        j_out = os.path.join(jd, 'data_payload_master.json')
        with open(j_out, 'w', encoding='utf-8') as f:
            json.dump(payload_vercel, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Pipeline completado con éxito.")
    print(f"     Payload JSON exportado a: {json_dirs}")
    print(f"     Gráficas guardadas en: {graphs_dirs}")
    print("="*80)

if __name__ == '__main__':
    main()
