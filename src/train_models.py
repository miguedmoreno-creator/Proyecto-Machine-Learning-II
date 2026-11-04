"""
=============================================================================
PROYECTO INTEGRADOR - MACHINE LEARNING II (UNIVERSIDAD EXTERNADO DE COLOMBIA)
FASE 2: SEGUNDA ENTREGA - MODELAMIENTO AVANZADO, VALIDACIÓN E INTERPRETACIÓN
Reproducción y Adaptación de Tiwari et al. (2023) al caso TransMi[App]
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
    recall_score, confusion_matrix, roc_curve, precision_score
)
from sklearn.calibration import CalibratedClassifierCV

# Modelos Línea Base (ML1)
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

# Compatibilidad de CatBoost con scikit-learn 1.6+
CatBoostClassifier.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

# Estilo de gráficos
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

def main():
    os.makedirs('graphs', exist_ok=True)
    
    print("="*75)
    print("1. CARGA DE DATOS Y CONSTRUCCIÓN DE PARTICIONES SIN FUGA DE DATOS")
    print("="*75)

    df = pd.read_csv('reseñas_transmiapp_bogota.csv')
    print(f"Total registros cargados: {len(df)}")
    df['contenido_limpio'] = df['contenido'].apply(preprocesar_texto)

    # Filtrar registros mínimos con texto significativo
    df = df[df['contenido_limpio'].str.len() >= 3].reset_index(drop=True)
    print(f"Registros tras depuración mínima: {len(df)}")

    X = df['contenido_limpio'].values
    y = df['sentimiento_binario'].values

    ratio_desbalance = (y == 0).sum() / (y == 1).sum()
    print(f"Distribución de Clases: Quejas (0) = {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%) | Elogios (1) = {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
    print(f"Ratio de Desbalance Neg/Pos: {ratio_desbalance:.2f}:1")

    # Partición estratificada Hold-out 80/20 (Semilla fija 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Conjunto de Entrenamiento (80%): {len(X_train)} registros (Quejas: {(y_train == 0).sum()}, Elogios: {(y_train == 1).sum()})")
    print(f"Conjunto de Prueba Independiente (20%): {len(X_test)} registros (Quejas: {(y_test == 0).sum()}, Elogios: {(y_test == 1).sum()})")

    # Vectorizador TF-IDF base para los Pipelines
    vectorizador_base = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=4000,
        sublinear_tf=True,
        stop_words=STOPWORDS_ES
    )

    modelos = {
        # Líneas Base (ML1)
        'Dummy (Mayoría)': (
            Pipeline([('vec', vectorizador_base), ('clf', DummyClassifier(strategy='most_frequent'))]),
            'Línea Base'
        ),
        'Multinomial NB': (
            Pipeline([('vec', vectorizador_base), ('clf', MultinomialNB(alpha=0.5))]),
            'Línea Base'
        ),
        'Regresión Logística': (
            Pipeline([('vec', vectorizador_base), ('clf', LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=RANDOM_STATE))]),
            'Línea Base'
        ),
        # Familia Bagging (ML2)
        'Random Forest': (
            Pipeline([('vec', vectorizador_base), ('clf', RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=RANDOM_STATE))]),
            'Bagging'
        ),
        'Extra Trees': (
            Pipeline([('vec', vectorizador_base), ('clf', ExtraTreesClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=RANDOM_STATE))]),
            'Bagging'
        ),
        'Bagging Linear SVC': (
            Pipeline([('vec', vectorizador_base), ('clf', BaggingClassifier(
                estimator=CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.5, random_state=RANDOM_STATE, max_iter=2000), cv=2),
                n_estimators=10, random_state=RANDOM_STATE
            ))]),
            'Bagging'
        ),
        # Familia Boosting (ML2)
        'AdaBoost': (
            Pipeline([('vec', vectorizador_base), ('clf', AdaBoostClassifier(n_estimators=80, learning_rate=0.3, random_state=RANDOM_STATE))]),
            'Boosting'
        ),
        'Gradient Boosting': (
            Pipeline([('vec', vectorizador_base), ('clf', GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE))]),
            'Boosting'
        ),
        'XGBoost': (
            Pipeline([('vec', vectorizador_base), ('clf', XGBClassifier(
                n_estimators=100, learning_rate=0.08, max_depth=5, scale_pos_weight=ratio_desbalance,
                random_state=RANDOM_STATE, eval_metric='logloss'
            ))]),
            'Boosting'
        ),
        'CatBoost': (
            Pipeline([('vec', vectorizador_base), ('clf', CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=5, auto_class_weights='Balanced',
                random_seed=RANDOM_STATE, verbose=0
            ))]),
            'Boosting'
        )
    }

    # -------------------------------------------------------------------------
    # 2. VALIDACIÓN CRUZADA ESTRATIFICADA (5-FOLD CV SIN FUGA DE DATOS)
    # -------------------------------------------------------------------------
    print("\n" + "="*75)
    print("2. EVALUACIÓN MEDIANTE STRATIFIED 5-FOLD CROSS-VALIDATION EN TRAIN (80%)")
    print("="*75)

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, (pipe, fam) in modelos.items():
        t0 = time.time()
        acc_list = []
        bal_acc_list = []
        f1_w_list = []
        f1_m_list = []
        auc_list = []
        recall_0_list = []

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
                y_prob_f = y_pred_f

            acc_list.append(accuracy_score(y_val_f, y_pred_f))
            bal_acc_list.append(balanced_accuracy_score(y_val_f, y_pred_f))
            f1_w_list.append(f1_score(y_val_f, y_pred_f, average='weighted'))
            f1_m_list.append(f1_score(y_val_f, y_pred_f, average='macro'))
            recall_0_list.append(recall_score(y_val_f, y_pred_f, pos_label=0))
            if len(np.unique(y_prob_f)) > 1:
                auc_list.append(roc_auc_score(y_val_f, y_prob_f))
            else:
                auc_list.append(0.5)

        t_cv = time.time() - t0
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
            'CV_Tiempo_s': float(t_cv)
        }
        print(f"[{fam:10s}] {name:20s} | BalAcc: {np.mean(bal_acc_list)*100:5.2f}% +/- {np.std(bal_acc_list)*100:4.2f}% | F1-Pond: {np.mean(f1_w_list)*100:5.2f}% | Recall Quejas: {np.mean(recall_0_list)*100:5.2f}% | AUC: {np.mean(auc_list):.4f} ({t_cv:5.1f}s)")

    cv_df = pd.DataFrame(cv_results).T

    # -------------------------------------------------------------------------
    # 3. EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA HOLD-OUT (20%, N=1.219)
    # -------------------------------------------------------------------------
    print("\n" + "="*75)
    print("3. EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA HOLD-OUT (20%, N=1.219)")
    print("="*75)

    test_results = {}
    trained_models = {}
    probas_test = {}
    preds_test = {}

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
            y_prob = y_pred

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        acc = accuracy_score(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        wf1 = f1_score(y_test, y_pred, average='weighted')
        mf1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
        recall_0 = recall_score(y_test, y_pred, pos_label=0)  # Recall Quejas
        tpr_1 = recall_score(y_test, y_pred, pos_label=1)     # Sensibilidad Elogios
        fpr_fallas = fp / (fp + tn + 1e-9)                    # Tasa de falsas alarmas

        test_results[name] = {
            'Familia': fam,
            'Test_Accuracy': float(acc),
            'Test_BalancedAcc': float(b_acc),
            'Test_WeightedF1': float(wf1),
            'Test_MacroF1': float(mf1),
            'Test_ROCAUC': float(auc),
            'Recall_Quejas_0': float(recall_0),
            'TPR_Elogios_1': float(tpr_1),
            'FPR_FalsosPos': float(fpr_fallas),
            'Tiempo_Train_s': float(t_train),
            'Latencia_Inf_ms': float(t_inf_ms),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
        }
        trained_models[name] = pipe
        preds_test[name] = y_pred
        probas_test[name] = y_prob
        
        print(f"[{fam:10s}] {name:25s} | BalAcc: {b_acc*100:5.2f}% | F1-Pond: {wf1*100:5.2f}% | Recall Quejas: {recall_0*100:5.2f}% | TPR Pos: {tpr_1*100:5.2f}% | FPR: {fpr_fallas*100:4.2f}% | AUC: {auc:.4f} | Inf: {t_inf_ms:4.2f} ms")

    test_df = pd.DataFrame(test_results).T

    # -------------------------------------------------------------------------
    # 4. GENERACIÓN DE GRÁFICAS DE MODELAMIENTO
    # -------------------------------------------------------------------------

    # FIGURA 7: Comparación en Validación Cruzada
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    cv_plot_df = cv_df.copy()
    cv_plot_df['BalancedAcc_%'] = cv_plot_df['CV_BalancedAcc_mean'] * 100
    cv_plot_df['BalancedAcc_std_%'] = cv_plot_df['CV_BalancedAcc_std'] * 100
    cv_plot_df['WeightedF1_%'] = cv_plot_df['CV_WeightedF1_mean'] * 100
    cv_plot_df['WeightedF1_std_%'] = cv_plot_df['CV_WeightedF1_std'] * 100
    cv_plot_df['ROCAUC'] = cv_plot_df['CV_ROCAUC_mean']
    cv_plot_df['ROCAUC_std'] = cv_plot_df['CV_ROCAUC_std']

    colors_fam = {'Línea Base': '#457B9D', 'Bagging': '#2A9D8F', 'Boosting': '#E76F51'}
    colores = [colors_fam[f] for f in cv_plot_df['Familia']]

    y_pos = np.arange(len(cv_plot_df))
    axes[0, 0].barh(y_pos, cv_plot_df['BalancedAcc_%'], xerr=cv_plot_df['BalancedAcc_std_%'],
                    color=colores, edgecolor='black', alpha=0.85, capsize=4)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(cv_plot_df.index, fontweight='bold')
    axes[0, 0].set_xlim(40, 100)
    axes[0, 0].set_xlabel('Balanced Accuracy (%) ± 1σ')
    axes[0, 0].set_title('A. Balanced Accuracy en Validación Cruzada (5-Fold CV)')

    axes[0, 1].barh(y_pos, cv_plot_df['WeightedF1_%'], xerr=cv_plot_df['WeightedF1_std_%'],
                    color=colores, edgecolor='black', alpha=0.85, capsize=4)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(cv_plot_df.index, fontweight='bold')
    axes[0, 1].set_xlim(60, 100)
    axes[0, 1].set_xlabel('Weighted F1-Score (%) ± 1σ')
    axes[0, 1].set_title('B. F1-Score Ponderado en Validación Cruzada')

    axes[1, 0].barh(y_pos, cv_plot_df['ROCAUC'], xerr=cv_plot_df['ROCAUC_std'],
                    color=colores, edgecolor='black', alpha=0.85, capsize=4)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(cv_plot_df.index, fontweight='bold')
    axes[1, 0].set_xlim(0.4, 1.0)
    axes[1, 0].set_xlabel('Área Bajo la Curva ROC (ROC-AUC) ± 1σ')
    axes[1, 0].set_title('C. Capacidad Discriminante (ROC-AUC) en Validación Cruzada')

    axes[1, 1].barh(y_pos, cv_plot_df['CV_Tiempo_s'], color=colores, edgecolor='black', alpha=0.85)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(cv_plot_df.index, fontweight='bold')
    axes[1, 1].set_xlabel('Tiempo Total de Validación Cruzada (Segundos)')
    axes[1, 1].set_title('D. Eficiencia Computacional (Tiempo de Entrenamiento CV)')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='black', label=f) for f, c in colors_fam.items()]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('graphs/model_01_cv_comparison.png', dpi=300)
    plt.close()
    print("-> Gráfica 7 guardada: graphs/model_01_cv_comparison.png")

    # FIGURA 8: Matrices de Confusión en Test Hold-out
    modelos_destacados = [
        'Dummy (Mayoría)',
        'Regresión Logística',
        'Bagging Linear SVC',
        'XGBoost'
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for i, m_name in enumerate(modelos_destacados):
        cm = confusion_matrix(y_test, preds_test[m_name])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array([[f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)"],
                          [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)"]])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Queja (0)', 'Elogio (1)'],
                    yticklabels=['Queja (0)', 'Elogio (1)'])
        axes[i].set_title(f"{m_name}\nBalAcc: {test_results[m_name]['Test_BalancedAcc']*100:.1f}% | AUC: {test_results[m_name]['Test_ROCAUC']:.3f}", fontsize=11)
        axes[i].set_xlabel('Predicción del Modelo')
        if i == 0:
            axes[i].set_ylabel('Etiqueta Real de Usuario')
        else:
            axes[i].set_ylabel('')

    plt.tight_layout()
    plt.savefig('graphs/model_02_confusion_matrices.png', dpi=300)
    plt.close()
    print("-> Gráfica 8 guardada: graphs/model_02_confusion_matrices.png")

    # FIGURA 9: Curvas ROC Comparativas
    plt.figure(figsize=(10, 7))
    modelos_roc = [
        ('Dummy (Mayoría)', probas_test['Dummy (Mayoría)'], '#8D99AE', ':'),
        ('Multinomial NB', probas_test['Multinomial NB'], '#457B9D', '--'),
        ('Regresión Logística', probas_test['Regresión Logística'], '#1D3557', '-'),
        ('Random Forest', probas_test['Random Forest'], '#52B788', '-'),
        ('Extra Trees', probas_test['Extra Trees'], '#2D6A4F', '-'),
        ('Bagging Linear SVC', probas_test['Bagging Linear SVC'], '#0077B6', '-'),
        ('AdaBoost', probas_test['AdaBoost'], '#F4A261', '--'),
        ('Gradient Boosting', probas_test['Gradient Boosting'], '#E76F51', '-'),
        ('XGBoost', probas_test['XGBoost'], '#D62828', '-'),
        ('CatBoost', probas_test['CatBoost'], '#9B2226', '-')
    ]

    for name, y_prob, color, ls in modelos_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
        plt.plot(fpr, tpr, label=f"{name:25s} (AUC = {auc:.4f})", color=color, linestyle=ls, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador Aleatorio (AUC = 0.5000)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR / 1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR / Sensibilidad)')
    plt.title('Curvas ROC Comparativas en Prueba Hold-out (Test 20%, N=1.219)', fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig('graphs/model_03_roc_curves.png', dpi=300)
    plt.close()
    print("-> Gráfica 9 guardada: graphs/model_03_roc_curves.png")

    # FIGURA 10: Importancia Léxica de Características
    log_pipe = trained_models['Regresión Logística']
    vectorizer = log_pipe.named_steps['vec']
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_clf = log_pipe.named_steps['clf']

    coefs = log_clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-15:]
    top_neg_idx = np.argsort(coefs)[:15]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(feature_names[top_neg_idx], np.abs(coefs[top_neg_idx]), color='#E63946', edgecolor='black', alpha=0.85)
    axes[0].set_xlabel('Magnitud del Peso en Función de Decisión (|Coeficiente|)')
    axes[0].set_title('A. Términos Determinantes en Detección de Quejas (Y=0)\n(Fallas de servicio, paraderos, actualización, demora, pésima)')

    axes[1].barh(feature_names[top_pos_idx], coefs[top_pos_idx], color='#2A9D8F', edgecolor='black', alpha=0.85)
    axes[1].set_xlabel('Magnitud del Peso en Función de Decisión (Coeficiente)')
    axes[1].set_title('B. Términos Determinantes en Detección de Elogios (Y=1)\n(Excelente, buena, muy útil, ayuda, gracias, genial)')

    plt.tight_layout()
    plt.savefig('graphs/model_04_feature_importance.png', dpi=300)
    plt.close()
    print("-> Gráfica 10 guardada: graphs/model_04_feature_importance.png")

    # FIGURA 11: Análisis Cualitativo y Cuantitativo de Errores
    best_preds = preds_test['Bagging Linear SVC']
    df_test = pd.DataFrame({
        'contenido': X_test,
        'y_real': y_test,
        'y_pred': best_preds,
        'prob': probas_test['Bagging Linear SVC']
    })

    falsos_positivos = df_test[(df_test['y_real'] == 0) & (df_test['y_pred'] == 1)]
    falsos_negativos = df_test[(df_test['y_real'] == 1) & (df_test['y_pred'] == 0)]

    print(f"\n--- Diagnóstico de Errores de Bagging Linear SVC en Test ---")
    print(f"Total Falsos Positivos (Quejas clasificadas como Elogio): {len(falsos_positivos)}")
    print(f"Total Falsos Negativos (Elogios clasificados como Queja): {len(falsos_negativos)}")

    error_breakdown = pd.DataFrame({
        'Categoría': ['Acierto: Queja Real (TN)', 'Acierto: Elogio Real (TP)', 'Error: Falso Negativo (FN)', 'Error: Falso Positivo (FP)'],
        'Cantidad': [
            int(((df_test['y_real'] == 0) & (df_test['y_pred'] == 0)).sum()),
            int(((df_test['y_real'] == 1) & (df_test['y_pred'] == 1)).sum()),
            len(falsos_negativos),
            len(falsos_positivos)
        ],
        'Tipo': ['Acierto', 'Acierto', 'Error', 'Error']
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=error_breakdown, x='Categoría', y='Cantidad', hue='Tipo',
                palette={'Acierto': '#2A9D8F', 'Error': '#E63946'}, ax=axes[0], edgecolor='black')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha='right', fontweight='bold')
    axes[0].set_title('A. Desglose Numérico de Predicciones en Test (N=1.219)')
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(f"{int(h):,}\n({h/len(df_test)*100:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                             ha='center', va='center', color='white', fontweight='bold')

    df_test['tipo_resultado'] = 'Acierto'
    df_test.loc[(df_test['y_real'] == 0) & (df_test['y_pred'] == 1), 'tipo_resultado'] = 'Falso Positivo (FP)'
    df_test.loc[(df_test['y_real'] == 1) & (df_test['y_pred'] == 0), 'tipo_resultado'] = 'Falso Negativo (FN)'
    df_test['longitud_palabras'] = df_test['contenido'].apply(lambda x: len(str(x).split()))

    sns.boxplot(data=df_test[df_test['tipo_resultado'] != 'Acierto'], x='tipo_resultado', y='longitud_palabras',
                palette={'Falso Positivo (FP)': '#F4A261', 'Falso Negativo (FN)': '#E76F51'}, ax=axes[1])
    axes[1].set_title('B. Longitud de Reseñas en Casos de Error (Boxplot)')
    axes[1].set_xlabel('Modalidad de Error Predictivo')
    axes[1].set_ylabel('Longitud en Palabras')

    plt.tight_layout()
    plt.savefig('graphs/model_05_error_analysis.png', dpi=300)
    plt.close()
    print("-> Gráfica 11 guardada: graphs/model_05_error_analysis.png")

    # Exportación de métricas a disco
    test_df.to_csv('results_test_metrics.csv', index=True)
    cv_df.to_csv('results_cv_metrics.csv', index=True)

    resumen_json = {
        'total_corpus': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'clase_quejas_0': int((y == 0).sum()),
        'clase_elogios_1': int((y == 1).sum()),
        'ratio_desbalance': round(ratio_desbalance, 2),
        'cv_metrics': cv_df.to_dict(orient='index'),
        'test_metrics': test_df.to_dict(orient='index'),
        'ejemplos_falsos_positivos': falsos_positivos[['contenido', 'y_real', 'y_pred', 'prob']].head(5).to_dict(orient='records'),
        'ejemplos_falsos_negativos': falsos_negativos[['contenido', 'y_real', 'y_pred', 'prob']].head(5).to_dict(orient='records')
    }

    with open('results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(resumen_json, f, ensure_ascii=False, indent=2)

    print("\n" + "="*75)
    print("EXPERIMENTACIÓN COMPLETADA CON ÉXITO.")
    print("Resultados exportados: results_test_metrics.csv, results_cv_metrics.csv, results_summary.json")
    print("Gráficas exportadas a /graphs")
    print("="*75)

if __name__ == '__main__':
    main()
