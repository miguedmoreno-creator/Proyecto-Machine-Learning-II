import pandas as pd
import numpy as np

df = pd.read_csv('reseñas_transmiapp_bogota.csv')
print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS & DTYPES ===")
print(df.dtypes)

print("\n=== NULL VALUES ===")
print(df.isnull().sum())

print("\n=== CALIFICACION COUNTS ===")
print(df['calificacion'].value_counts().sort_index())

print("\n=== SENTIMIENTO_BINARIO COUNTS ===")
print(df['sentimiento_binario'].value_counts(dropna=False))

print("\n=== SENTIMIENTO_3CLASES COUNTS ===")
print(df['sentimiento_3clases'].value_counts(dropna=False))

print("\n=== LENGTH STATS BY SENTIMENT (WORDS) ===")
df['word_count'] = df['contenido'].fillna('').apply(lambda x: len(str(x).split()))
df['char_count'] = df['contenido'].fillna('').apply(lambda x: len(str(x)))
print(df.groupby('sentimiento_binario')[['word_count', 'char_count']].describe())

print("\n=== THUMBS UP STATS ===")
print(df.groupby('sentimiento_binario')['thumbs_up'].describe())

print("\n=== DATES RANGE ===")
print("Min date:", df['fecha'].min(), "| Max date:", df['fecha'].max())

print("\n=== SAMPLE ROWS ===")
for idx, row in df.head(8).iterrows():
    print(f"[{row['calificacion']}* | Bin: {row['sentimiento_binario']}] {str(row['contenido'])[:100]}...")
