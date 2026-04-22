# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- NUEVOS IMPORTS AÑADIDOS ---
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# 1. DIVISIÓN DE DATOS POR SENTIMIENTO
# ==========================================
def dividir_por_sentimiento(df, target_col="score"):
    """
    Divide el DataFrame en tres subconjuntos basados en la puntuación (1-5).
    - Negativos: 1 y 2
    - Neutros: 3
    - Positivos: 4 y 5
    """
    print(" -> Dividiendo los datos en Positivos, Negativos y Neutros...")

    df_negativos = df[df[target_col].isin([1, 2])].copy()
    df_neutros = df[df[target_col] == 3].copy()
    df_positivos = df[df[target_col].isin([4, 5])].copy()

    print(f"    * Positivos: {len(df_positivos)} filas")
    print(f"    * Neutros: {len(df_neutros)} filas")
    print(f"    * Negativos: {len(df_negativos)} filas")

    return df_positivos, df_negativos, df_neutros


# ==========================================
# 2. BARRIDO K-MEANS Y GRÁFICO DEL CODO
# ==========================================
def plot_grafico_codo(k_values, inertias, sentimiento, output_dir="graficos"):
    """
    Genera y guarda el gráfico del codo para evaluar K-Means.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o', linestyle='--', color='b')
    plt.title(f'Gráfico del Codo - Sentimiento: {sentimiento.capitalize()}')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia (Distancia al centroide)')
    plt.grid(True)

    ruta_guardado = os.path.join(output_dir, f"codo_kmeans_{sentimiento}.png")
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"    [+] Gráfico del codo guardado en: {ruta_guardado}")


def barrido_kmeans(datos_procesados, k_min=2, k_max=10, sentimiento="general"):
    """
    Realiza un barrido de hiperparámetros para K-Means.
    """
    print(f"\n--- Iniciando barrido K-Means para: {sentimiento.upper()} ---")

    resultados_barrido = []
    k_values = []
    inertias = []

    for k in range(k_min, k_max + 1):
        print(f"  -> Probando K-Means con k={k}")

        # IMPLEMENTACIÓN REAL DE K-MEANS
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(datos_procesados)
        inercia_actual = kmeans.inertia_

        k_values.append(k)
        inertias.append(inercia_actual)

        resultados_barrido.append({
            "k": k,
            "inercia": inercia_actual,
            "modelo": kmeans
        })

    plot_grafico_codo(k_values, inertias, sentimiento)
    return resultados_barrido


# ==========================================
# 3. BARRIDO LDA (Latent Dirichlet Allocation)
# ==========================================
def barrido_lda(datos_procesados, n_topics_list=[2, 3, 4, 5], sentimiento="general"):
    """
    Realiza un barrido de hiperparámetros para LDA usando bucles.
    """
    print(f"\n--- Iniciando barrido LDA para: {sentimiento.upper()} ---")

    resultados_barrido = []

    for n_topics in n_topics_list:
        print(f"  -> Probando LDA con n_topics={n_topics}")

        # IMPLEMENTACIÓN REAL DE LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(datos_procesados)
        perplejidad = lda.perplexity(datos_procesados)

        resultados_barrido.append({
            "n_topics": n_topics,
            "perplexity": perplejidad,
            "modelo": lda
        })

    return resultados_barrido


# ==========================================
# 4. FUNCIÓN PRINCIPAL / ORQUESTADOR
# ==========================================
def pipeline_clustering(df, col_texto_procesado, target_col="score"):
    print("\n" + "=" * 50)
    print(" 🚀 INICIANDO PIPELINE DE CLUSTERING Y TOPIC MODELING")
    print("=" * 50)

    # 1. Separar datos
    df_pos, df_neg, df_neu = dividir_por_sentimiento(df, target_col)

    datasets = {
        "positivos": df_pos,
        "negativos": df_neg,
        "neutros": df_neu
    }

    resultados_totales = {}

    # Inicializamos el vectorizador (Convierte texto en matriz numérica)
    vectorizer = TfidfVectorizer(max_features=1000)

    # 2. Bucle principal por cada sentimiento
    for sentimiento, sub_df in datasets.items():
        n_muestras = len(sub_df)

        if n_muestras < 2:
            print(f"\n⚠️ Saltando '{sentimiento}': Solo hay {n_muestras} filas (insuficiente para clusterizar).")
            continue

        print(f"\n[+] Vectorizando textos para: {sentimiento.upper()}...")
        matriz_vectorizada = vectorizer.fit_transform(sub_df[col_texto_procesado])

        # Ajuste dinámico de hiperparámetros según la cantidad de datos reales
        # k (clusters) no puede ser mayor a la cantidad de filas
        max_k = min(8, n_muestras - 1)
        temas_validos = [t for t in [2, 3, 4, 5] if t <= n_muestras - 1]

        # --- Ejecución de barridos ---
        res_kmeans = barrido_kmeans(matriz_vectorizada, k_min=2, k_max=max_k, sentimiento=sentimiento)
        res_lda = barrido_lda(matriz_vectorizada, n_topics_list=temas_validos, sentimiento=sentimiento)

        resultados_totales[sentimiento] = {
            "kmeans": res_kmeans,
            "lda": res_lda,
            "vectorizador": vectorizer  # Guardamos el vectorizador por si necesitas predecir nuevos textos luego
        }

    print("\n✅ Tareas de barrido de Clustering finalizadas.")
    return resultados_totales