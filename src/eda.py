"""
=============================================================================
PROYECTO INTEGRADOR - MACHINE LEARNING II (UNIVERSIDAD EXTERNADO DE COLOMBIA)
FASE 2: SEGUNDA ENTREGA - ANÁLISIS EXPLORATORIO DE DATOS (EDA)
Dataset: Reseñas Oficiales de TransMi[App] (com.nexura.transmilenio)
=============================================================================
"""

import os
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Configuración estética profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'

PALETA_SENTIMIENTO = {0: '#E63946', 1: '#2A9D8F'}  # Rojo coral (Quejas), Verde azulado (Elogios)
PALETA_ESTRELLAS = {1: '#D62828', 2: '#F77F00', 4: '#588157', 5: '#2A9D8F'}

os.makedirs('graphs', exist_ok=True)

# 1. Carga y verificación de calidad del dataset
print("="*70)
print("1. CARGA Y REVISIÓN DE INTEGRIDAD DE DATOS")
print("="*70)

df = pd.read_csv('reseñas_transmiapp_bogota.csv')
print(f"Total registros cargados: {df.shape[0]}")
print(f"Total columnas: {df.shape[1]}")
print(f"Columnas disponibles: {list(df.columns)}")

# Métricas de calidad y faltantes
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df)) * 100
quality_df = pd.DataFrame({'Valores_Faltantes': null_counts, 'Porcentaje_%': null_pct.round(2), 'Tipo_Dato': df.dtypes})
print("\n--- Estado de Calidad y Faltantes ---")
print(quality_df)

# Feature engineering preliminar para EDA
df['contenido_str'] = df['contenido'].fillna('').astype(str)
df['longitud_caracteres'] = df['contenido_str'].apply(len)
df['longitud_palabras'] = df['contenido_str'].apply(lambda x: len(x.split()))
df['tiene_exclamacion'] = df['contenido_str'].apply(lambda x: 1 if '!' in x or '¡' in x else 0)
df['tiene_interrogacion'] = df['contenido_str'].apply(lambda x: 1 if '?' in x or '¿' in x else 0)
df['mayusculas_ratio'] = df['contenido_str'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5))

# Guardar resumen estadístico
stats_words = df.groupby('sentimiento_binario')['longitud_palabras'].agg(['count', 'mean', 'std', 'median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).rename(columns={'<lambda_0>': 'IQR'})
print("\n--- Estadísticas de Longitud en Palabras por Sentimiento ---")
print(stats_words)

# -----------------------------------------------------------------------------
# FIGURA 1: Calidad de Datos, Faltantes y Composición del Corpus
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Subplot A: Completitud de atributos
completitud = 100 - null_pct
colors_comp = ['#2A9D8F' if p == 100 else '#E76F51' for p in completitud]
bars = axes[0].barh(completitud.index, completitud.values, color=colors_comp, edgecolor='black', alpha=0.85)
axes[0].set_xlim(70, 105)
axes[0].set_xlabel('Porcentaje de Registros Completos (%)')
axes[0].set_title('A. Tasa de Completitud por Atributo (Calidad de Datos)')
for bar in bars:
    w = bar.get_width()
    axes[0].text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')

# Subplot B: Distribución de calificaciones (1 a 5 estrellas)
star_counts = df['calificacion'].value_counts().sort_index()
colors_stars = [PALETA_ESTRELLAS.get(s, '#457B9D') for s in star_counts.index]
bars_s = axes[1].bar([str(s) + '★' for s in star_counts.index], star_counts.values, color=colors_stars, edgecolor='black', alpha=0.85)
axes[1].set_ylabel('Número de Reseñas')
axes[1].set_xlabel('Calificación de Usuario (Google Play Store)')
axes[1].set_title('B. Distribución de Calificaciones de Usuario')
for bar in bars_s:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, h + 50, f"{h:,}\n({pct:.1f}%)", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/eda_01_calidad_y_calificaciones.png', dpi=300)
plt.close()
print("-> Gráfica 1 guardada: graphs/eda_01_calidad_y_calificaciones.png")

# -----------------------------------------------------------------------------
# FIGURA 2: Distribución de la Variable Objetivo — Solo datos reales TransMi[App]
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

target_counts = df['sentimiento_binario'].value_counts().sort_index()

# Subplot A: Donut chart de proporción de clases reales
labels = [
    f"Queja / Reclamo (Y=0)\n{target_counts[0]:,} reseñas ({target_counts[0]/len(df)*100:.1f}%)",
    f"Elogio / Satisfacción (Y=1)\n{target_counts[1]:,} reseñas ({target_counts[1]/len(df)*100:.1f}%)"
]
wedges, texts = axes[0].pie(
    target_counts,
    labels=labels,
    colors=[PALETA_SENTIMIENTO[0], PALETA_SENTIMIENTO[1]],
    startangle=140,
    explode=(0.05, 0.05),
    wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2)
)
axes[0].set_title('A. Distribución de la Variable Objetivo (Y) — TransMi[App]')
# Anotación central
axes[0].text(0, 0, f"N={len(df):,}\nreseñas", ha='center', va='center', fontsize=11, fontweight='bold', color='#1a365d')

# Subplot B: Composición por calificación de estrella
star_sentiment = df.groupby(['calificacion', 'sentimiento_binario']).size().unstack(fill_value=0)
star_sentiment.index = [f"{s}★" for s in star_sentiment.index]
colors_bar = [PALETA_SENTIMIENTO[0], PALETA_SENTIMIENTO[1]]
star_sentiment.plot(
    kind='bar', ax=axes[1],
    color=colors_bar,
    edgecolor='black', alpha=0.88,
    width=0.7
)
axes[1].set_xlabel('Calificación del Usuario (Google Play Store)')
axes[1].set_ylabel('Número de Reseñas')
axes[1].set_title('B. Composición de Sentimiento por Calificación (1★–5★)')
axes[1].set_xticklabels(star_sentiment.index, rotation=0, fontweight='bold')
axes[1].legend(['Queja/Reclamo (Y=0)', 'Elogio/Satisfacción (Y=1)'], loc='upper center')
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%d', fontsize=8.5, fontweight='bold', padding=3)

plt.tight_layout()
plt.savefig('graphs/eda_02_desbalance_y_benchmark.png', dpi=300)
plt.close()
print("-> Gráfica 2 guardada: graphs/eda_02_desbalance_y_benchmark.png")

# -----------------------------------------------------------------------------
# FIGURA 3: Distribución de Longitud de Reseñas (Palabras y Caracteres)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot A: KDE densidad de palabras (truncada en percentil 98 para visualización limpia)
p98 = df['longitud_palabras'].quantile(0.98)
sns.kdeplot(data=df[df['sentimiento_binario'] == 0], x='longitud_palabras', ax=axes[0],
            color=PALETA_SENTIMIENTO[0], fill=True, alpha=0.35, linewidth=2, label=f'Quejas (Y=0) [μ={stats_words.loc[0, "mean"]:.1f}, med={stats_words.loc[0, "median"]:.0f}]')
sns.kdeplot(data=df[df['sentimiento_binario'] == 1], x='longitud_palabras', ax=axes[0],
            color=PALETA_SENTIMIENTO[1], fill=True, alpha=0.35, linewidth=2, label=f'Elogios (Y=1) [μ={stats_words.loc[1, "mean"]:.1f}, med={stats_words.loc[1, "median"]:.0f}]')
axes[0].set_xlim(0, p98)
axes[0].set_xlabel('Longitud de la Reseña (Número de Palabras)')
axes[0].set_ylabel('Densidad de Probabilidad')
axes[0].set_title('A. Densidad (KDE) de la Longitud Textual por Sentimiento')
axes[0].legend()

# Subplot B: Boxplot comparativo
sns.boxplot(data=df, x='sentimiento_binario', y='longitud_palabras', ax=axes[1],
            palette=[PALETA_SENTIMIENTO[0], PALETA_SENTIMIENTO[1]], showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":"8"})
axes[1].set_xticklabels(['Quejas / Reclamos (Y=0)', 'Elogios / Satisfacción (Y=1)'], fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].set_ylabel('Número de Palabras')
axes[1].set_xlabel('')
axes[1].set_title('B. Dispersión y Asimetría de Longitud (Boxplot)')

plt.tight_layout()
plt.savefig('graphs/eda_03_distribucion_longitud.png', dpi=300)
plt.close()
print("-> Gráfica 3 guardada: graphs/eda_03_distribucion_longitud.png")

# -----------------------------------------------------------------------------
# FIGURA 4: Análisis Multivariado y Matriz de Correlación
# -----------------------------------------------------------------------------
corr_cols = ['calificacion', 'sentimiento_binario', 'longitud_palabras', 'longitud_caracteres', 'thumbs_up', 'mayusculas_ratio']
corr_matrix_pearson = df[corr_cols].corr(method='pearson')
corr_matrix_spearman = df[corr_cols].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
col_labels = ['Calificación', 'Sentimiento', 'Palabras', 'Caracteres', 'Votos Útiles', 'Ratio Mayús.']

sns.heatmap(corr_matrix_pearson, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1,
            xticklabels=col_labels, yticklabels=col_labels, ax=axes[0], cbar_kws={'label': 'Coeficiente r'})
axes[0].set_title('A. Correlación Lineal de Pearson')

sns.heatmap(corr_matrix_spearman, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1,
            xticklabels=col_labels, yticklabels=col_labels, ax=axes[1], cbar_kws={'label': 'Coeficiente ρ'})
axes[1].set_title('B. Correlación de Rangos de Spearman')

plt.tight_layout()
plt.savefig('graphs/eda_04_correlaciones_multivariadas.png', dpi=300)
plt.close()
print("-> Gráfica 4 guardada: graphs/eda_04_correlaciones_multivariadas.png")

# -----------------------------------------------------------------------------
# FIGURA 5: Nubes de Palabras Léxicas por Sentimiento
# -----------------------------------------------------------------------------
STOPWORDS_ES = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no',
    'una', 'su', 'al', 'lo', 'como', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'este', 'ha', 'si',
    'porque', 'esta', 'son', 'entre', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene', 'tambien',
    'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni',
    'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mi', 'antes', 'algunos', 'que', 'unos',
    'yo', 'otro', 'otras', 'otra', 'el', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos',
    'cual', 'sea', 'poco', 'ella', 'estar', 'haber', 'estas', 'estaba', 'estamos', 'algunas', 'algo',
    'nosotros', 'mi', 'mis', 'tu', 'te', 'ti', 'app', 'aplicacion', 'transmilenio', 'transmiapp', 'bogota'
}

def limpiar_texto_simple(texto):
    texto = str(texto).lower()
    # Quitar acentos
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    tokens = [w for w in texto.split() if len(w) > 2 and w not in STOPWORDS_ES]
    return ' '.join(tokens)

df['texto_limpio'] = df['contenido'].apply(limpiar_texto_simple)
texto_quejas = ' '.join(df[df['sentimiento_binario'] == 0]['texto_limpio'])
texto_elogios = ' '.join(df[df['sentimiento_binario'] == 1]['texto_limpio'])

wc_neg = WordCloud(width=800, height=450, background_color='white', colormap='Reds_r',
                   max_words=80, random_state=42).generate(texto_quejas)
wc_pos = WordCloud(width=800, height=450, background_color='white', colormap='Greens_r',
                   max_words=80, random_state=42).generate(texto_elogios)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(wc_neg, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('A. Léxico Dominante en Quejas Ciudadanas (Y=0)\n(Problemas con rutas, actualización, buses, tiempos, mapa)', fontsize=12)

axes[1].imshow(wc_pos, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('B. Léxico Dominante en Elogios y Satisfacción (Y=1)\n(Excelente, buena, útil, ayuda, fácil, servicio)', fontsize=12)

plt.tight_layout()
plt.savefig('graphs/eda_05_wordclouds_sentimiento.png', dpi=300)
plt.close()
print("-> Gráfica 5 guardada: graphs/eda_05_wordclouds_sentimiento.png")

# -----------------------------------------------------------------------------
# FIGURA 6: Top N-gramas Más Frecuentes por Polaridad
# -----------------------------------------------------------------------------
def get_top_ngrams(corpus, n=1, top_k=15):
    tokens_list = []
    for doc in corpus:
        words = doc.split()
        if n == 1:
            tokens_list.extend(words)
        else:
            tokens_list.extend([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
    counter = Counter(tokens_list)
    return counter.most_common(top_k)

top_unigrams_neg = get_top_ngrams(df[df['sentimiento_binario'] == 0]['texto_limpio'], n=1, top_k=10)
top_unigrams_pos = get_top_ngrams(df[df['sentimiento_binario'] == 1]['texto_limpio'], n=1, top_k=10)

top_bigrams_neg = get_top_ngrams(df[df['sentimiento_binario'] == 0]['texto_limpio'], n=2, top_k=10)
top_bigrams_pos = get_top_ngrams(df[df['sentimiento_binario'] == 1]['texto_limpio'], n=2, top_k=10)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Unigramas Negativos
words_neg, counts_neg = zip(*reversed(top_unigrams_neg))
axes[0, 0].barh(words_neg, counts_neg, color='#E63946', edgecolor='black', alpha=0.85)
axes[0, 0].set_title('A. Top 10 Términos en Quejas (Y=0)')
axes[0, 0].set_xlabel('Frecuencia Absoluta')

# Subplot 2: Unigramas Positivos
words_pos, counts_pos = zip(*reversed(top_unigrams_pos))
axes[0, 1].barh(words_pos, counts_pos, color='#2A9D8F', edgecolor='black', alpha=0.85)
axes[0, 1].set_title('B. Top 10 Términos en Elogios (Y=1)')
axes[0, 1].set_xlabel('Frecuencia Absoluta')

# Subplot 3: Bigramas Negativos
bi_neg, bi_counts_neg = zip(*reversed(top_bigrams_neg))
axes[1, 0].barh(bi_neg, bi_counts_neg, color='#D62828', edgecolor='black', alpha=0.85)
axes[1, 0].set_title('C. Top 10 Bigramas en Quejas (Y=0)')
axes[1, 0].set_xlabel('Frecuencia Absoluta')

# Subplot 4: Bigramas Positivos
bi_pos, bi_counts_pos = zip(*reversed(top_bigrams_pos))
axes[1, 1].barh(bi_pos, bi_counts_pos, color='#588157', edgecolor='black', alpha=0.85)
axes[1, 1].set_title('D. Top 10 Bigramas en Elogios (Y=1)')
axes[1, 1].set_xlabel('Frecuencia Absoluta')

plt.tight_layout()
plt.savefig('graphs/eda_06_top_ngrams.png', dpi=300)
plt.close()
print("-> Gráfica 6 guardada: graphs/eda_06_top_ngrams.png")

print("\n" + "="*70)
print("EDA COMPLETADO EXITOSAMENTE. Todas las figuras se guardaron en /graphs")
print("="*70)
