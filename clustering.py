# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# --- NUEVOS IMPORTS AÑADIDOS ---
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel


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
#  PREPARACIÓN DEL CORPUS PARA GENSIM
# ==========================================
def preparar_corpus_gensim(textos):
    """
    Convierte una lista de textos al formato que necesita Gensim:
    - tokeniza cada texto en lista de palabras
    - crea el diccionario (vocabulary)
    - crea el corpus en formato Bag-of-Words (BoW)
    """
    # Tokenizamos: cada documento pasa a ser una lista de palabras en minúsculas
    textos_tokenizados = [str(texto).lower().split() for texto in textos]

    # Creamos el diccionario: mapea cada palabra a un ID único
    diccionario = corpora.Dictionary(textos_tokenizados)

    # Eliminamos palabras muy raras (en menos de 2 docs) o muy frecuentes (>50% docs)
    diccionario.filter_extremes(no_below=2, no_above=0.5)

    # Creamos el corpus en formato BoW: lista de (word_id, frecuencia)
    corpus_bow = [diccionario.doc2bow(doc) for doc in textos_tokenizados]

    return corpus_bow, diccionario, textos_tokenizados


# ==========================================
#  BÚSQUEDA MATEMÁTICA DEL CODO
# ==========================================
def encontrar_codo(x, y):
    """
    Encuentra el número de temas óptimo analizando la forma de la curva de coherencia.
    Para la coherencia, buscamos MAXIMIZAR (mayor coherencia = tópicos más interpretables).
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) == 1:
        return x[0], 0

    # 1. Buscamos el punto con la MAYOR coherencia
    idx_max_absoluto = np.argmax(y)

    # 2. Si el maximo esta en el medio o al inicio, es nuestro optimo
    if idx_max_absoluto != len(y) - 1:
        return x[idx_max_absoluto], idx_max_absoluto

    # 3. Si sube continuamente, usamos el metodo del codo geometrico
    punto_inicio = np.array([x[0], y[0]])
    punto_fin = np.array([x[-1], y[-1]])

    linea_vec = punto_fin - punto_inicio
    linea_vec_norm = linea_vec / np.sqrt(np.sum(linea_vec ** 2))

    distancias = []
    for i in range(len(x)):
        punto = np.array([x[i], y[i]])
        vec_a_punto = punto - punto_inicio
        proyeccion = np.sum(vec_a_punto * linea_vec_norm)
        vec_proyectado = proyeccion * linea_vec_norm
        distancia = np.sqrt(np.sum((vec_a_punto - vec_proyectado) ** 2))
        distancias.append(distancia)

    idx_codo = np.argmax(distancias)

    return x[idx_codo], idx_codo


# ==========================================
#  BARRIDO LDA Y GRÁFICO DE COHERENCIA
# ==========================================
def plot_codo_lda(n_topics_list, coherencias, sentimiento, mejor_n, output_dir="graficos"):
    """
    Genera y guarda el gráfico para evaluar la Coherencia de LDA.
    He añadido etiquetas en cada punto para que puedas elegir el codo manualmente.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(n_topics_list, coherencias, marker='s', linestyle='-', color='b', label='Coherencia (c_v)')

    # Marcamos todos los puntos con su valor para facilitar la elección manual
    for i, txt in enumerate(coherencias):
        plt.annotate(f"{txt:.3f}", (n_topics_list[i], coherencias[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    # Marcamos el codo sugerido por el algoritmo
    plt.axvline(x=mejor_n, color='r', linestyle='--', alpha=0.5, label=f'Sugerencia: {mejor_n} temas')
    plt.scatter(mejor_n, coherencias[n_topics_list.index(mejor_n)], color='red', s=100, zorder=5)

    plt.title(
        f'Evaluación de Coherencia LDA - {sentimiento.capitalize()}\n(Elige el punto donde la curva se estabiliza)')
    plt.xlabel('Número de Temas (k)')
    plt.ylabel('Coherencia (Mayor es mejor)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    ruta_guardado = os.path.join(output_dir, f"codo_lda_coherencia_{sentimiento}.png")
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"    [+] Gráfico con etiquetas guardado en: {ruta_guardado}")


def barrido_lda(corpus_bow, diccionario, textos_tokenizados, config_lda, sentimiento="general", output_dir="graficos"):
    """
    Realiza un barrido de LDA con Gensim y devuelve todos los modelos entrenados.
    """
    print(f"\n--- Iniciando barrido LDA (Gensim) para: {sentimiento.upper()} ---")

    n_topics_list = config_lda.get("n_components_range", [2, 3, 4, 5])
    passes = config_lda.get("passes", [10])[0]
    iterations = config_lda.get("max_iter", [50])[0]
    random_state = config_lda.get("random_state", [42])[0]
    metrica_coherencia = config_lda.get("coherence_metric", "c_v")

    resultados_barrido = []
    temas_evaluados = []
    coherencias = []

    for n_topics in n_topics_list:
        print(f"  -> Entrenando LDA con k={n_topics}...")

        lda = LdaModel(
            corpus=corpus_bow,
            id2word=diccionario,
            num_topics=n_topics,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            alpha='auto',
            eta='auto'
        )

        coherence_model = CoherenceModel(
            model=lda,
            texts=textos_tokenizados,
            dictionary=diccionario,
            coherence=metrica_coherencia
        )
        coherencia = coherence_model.get_coherence()

        temas_evaluados.append(n_topics)
        coherencias.append(coherencia)

        resultados_barrido.append({
            "n_topics": n_topics,
            "coherencia": coherencia,
            "modelo": lda
        })

    mejor_n_topics, _ = encontrar_codo(temas_evaluados, coherencias)
    plot_codo_lda(temas_evaluados, coherencias, sentimiento, mejor_n_topics, output_dir=output_dir)

    return resultados_barrido


# ==========================================
#            FUNCIÓN PRINCIPAL
# ==========================================
def pipeline_clustering(df, col_texto_procesado, json_file, target_col="score"):
    print("\n" + "=" * 50)
    print(" 🚀 INICIANDO PIPELINE DE CLUSTERING MULTI-RESULTADO")
    print("=" * 50)

    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config_lda = config.get("clustering_lda", {})
    rango_topics_original = config_lda.get("n_components_range", [2, 3, 4, 5])

    # 1. Separar datos
    df_pos, df_neg, df_neu = dividir_por_sentimiento(df, target_col)
    datasets = {"positivos": df_pos, "negativos": df_neg, "neutros": df_neu}

    # 2. Procesar cada grupo
    for sentimiento, sub_df in datasets.items():
        if len(sub_df) < 2:
            continue

        # Crear carpeta específica para este sentimiento
        dir_sentimiento = os.path.join("resultados_clustering", sentimiento)
        os.makedirs(dir_sentimiento, exist_ok=True)

        print(f"\n[+] Procesando grupo: {sentimiento.upper()}")
        textos_limpios = sub_df[col_texto_procesado].fillna("").tolist()
        corpus_bow, diccionario, textos_tokenizados = preparar_corpus_gensim(textos_limpios)

        # Filtrar temas validos segun tamaño de datos
        temas_validos = [t for t in rango_topics_original if t <= len(sub_df) - 1]
        config_lda_actual = config_lda.copy()
        config_lda_actual["n_components_range"] = temas_validos

        # Ejecutar barrido
        todos_los_modelos = barrido_lda(corpus_bow, diccionario, textos_tokenizados, config_lda_actual, sentimiento)

        # 3. GUARDAR TODOS LOS RESULTADOS
        print(f"  -> Guardando archivos CSV para cada k en: {dir_sentimiento}")
        for res in todos_los_modelos:
            k = res["n_topics"]
            modelo = res["modelo"]

            # Asignación de clusters
            temas_asignados = []
            for doc_bow in corpus_bow:
                distribucion = modelo.get_document_topics(doc_bow, minimum_probability=0)
                topico_principal = max(distribucion, key=lambda x: x[1])[0]
                temas_asignados.append(topico_principal)

            # Crear copia y guardar
            df_resultado = sub_df.copy()
            df_resultado['Cluster_LDA'] = temas_asignados

            nombre_archivo = f"clusters_k{k}.csv"
            ruta_csv = os.path.join(dir_sentimiento, nombre_archivo)
            df_resultado.to_csv(ruta_csv, index=False)

    print("\n✅ Proceso finalizado. Revisa las carpetas en 'resultados_clustering/'")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python clustering.py <data.csv> <config_file.json>")
        sys.exit(1)

    csv_data = sys.argv[1]
    json_file = sys.argv[2]

    try:
        df = pd.read_csv(csv_data)
        with open(json_file, 'r', encoding='utf-8') as f:
            config_json = json.load(f)

        target_col = config_json.get("target", "score")
        text_features = config_json.get("preprocessing", {}).get("text_features", ["content"])
        col_texto_procesado = text_features[0] if text_features else "content"

        pipeline_clustering(df, col_texto_procesado, json_file, target_col)
    except Exception as e:
        print(f"❌ Error: {e}")