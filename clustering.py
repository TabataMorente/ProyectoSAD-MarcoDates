# -*- coding: utf-8 -*-
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find('corpora/stopwords')
except Exception:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords


# ==========================================
# STOPWORDS DE DOMINIO
# ==========================================
DOMAIN_STOPWORDS = {
    "app", "use", "like", "get", "make", "time", "really", "just",
    "would", "one", "also", "even", "still", "well", "much", "many",
    "it", "its", "the", "and", "for", "that", "this", "with",
    "have", "has", "was", "are", "not", "but", "all", "very",
    "update", "version", "phone", "review", "rating",
}


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
    df_neutros   = df[df[target_col] == 3].copy()
    df_positivos = df[df[target_col].isin([4, 5])].copy()

    print(f"    * Positivos: {len(df_positivos)} filas")
    print(f"    * Neutros:   {len(df_neutros)} filas")
    print(f"    * Negativos: {len(df_negativos)} filas")

    return df_positivos, df_negativos, df_neutros


# ==========================================
# LIMPIEZA DE TEXTO PARA CLUSTERING
# ==========================================
def limpiar_texto_para_lda(texto, stop_words_set):
    """
    Limpia y tokeniza un texto para LDA:
    - Minúsculas
    - Elimina puntuación y números
    - Elimina stopwords generales + stopwords de dominio
    - Filtra tokens muy cortos (< 3 caracteres)
    """
    if not isinstance(texto, str):
        return []
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)  # elimina puntuación
    texto = re.sub(r'\d+', '', texto)        # elimina números
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stop_words_set and len(t) >= 3]
    return tokens


def construir_stopwords(lang='english'):
    """Combina stopwords de NLTK con stopwords de dominio."""
    try:
        sw = set(stopwords.words(lang))
    except Exception:
        sw = set()
    return sw | DOMAIN_STOPWORDS


# ==========================================
# PREPARACIÓN DEL CORPUS PARA GENSIM
# ==========================================
def preparar_corpus_gensim(textos_tokenizados, n_docs):
    """
    Recibe lista de listas de tokens ya limpios y construye
    el diccionario y corpus BoW para Gensim.
    filter_extremes se adapta al tamaño del corpus.

    CAMBIO 2: no_above bajado de 0.85→0.70 (corpus pequeños) y 0.6→0.55 (corpus grandes).
    Evita que palabras muy frecuentes pero poco informativas ensucien los temas,
    sin llegar al 0.5 del código viejo que era demasiado restrictivo.
    """
    diccionario = corpora.Dictionary(textos_tokenizados)

    no_below = max(2, int(n_docs * 0.01))          # al menos 1% de docs
    no_above = 0.70 if n_docs < 200 else 0.55      # equilibrio entre permisividad y limpieza

    diccionario.filter_extremes(no_below=no_below, no_above=no_above)

    corpus_bow_full = [diccionario.doc2bow(doc) for doc in textos_tokenizados]

    indices_validos  = [i for i, doc in enumerate(corpus_bow_full) if len(doc) > 0]
    corpus_bow       = [corpus_bow_full[i] for i in indices_validos]
    textos_filtrados = [textos_tokenizados[i] for i in indices_validos]

    print(f"    Vocabulario: {len(diccionario)} términos | Docs válidos: {len(corpus_bow)}")
    return corpus_bow, diccionario, textos_filtrados, indices_validos


# ==========================================
# BÚSQUEDA MATEMÁTICA DEL CODO
# ==========================================
def encontrar_codo(x, y):
    """
    Encuentra el número de temas óptimo maximizando coherencia.
    Si la curva sube continuamente, usa el mé_todo geométrico del codo.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) == 1:
        return x[0], 0

    idx_max = np.argmax(y)
    if idx_max != len(y) - 1:
        return x[idx_max], idx_max

    # Mé_todo del codo geométrico
    p_inicio = np.array([x[0], y[0]])
    p_fin    = np.array([x[-1], y[-1]])
    linea    = p_fin - p_inicio
    linea_n  = linea / np.linalg.norm(linea)

    distancias = []
    for i in range(len(x)):
        p          = np.array([x[i], y[i]])
        v          = p - p_inicio
        proy       = np.dot(v, linea_n) * linea_n
        distancias.append(np.linalg.norm(v - proy))

    idx_codo = int(np.argmax(distancias))
    return x[idx_codo], idx_codo


# ==========================================
# GRÁFICO DE COHERENCIA
# ==========================================
def plot_codo_lda(n_topics_list, coherencias, sentimiento, mejor_n, output_dir="graficos"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(n_topics_list, coherencias, marker='s', linestyle='-', color='b', label='Coherencia (c_v)')

    for i, val in enumerate(coherencias):
        plt.annotate(f"{val:.3f}", (n_topics_list[i], coherencias[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.axvline(x=mejor_n, color='r', linestyle='--', alpha=0.5, label=f'Sugerencia: {mejor_n} temas')
    plt.scatter(mejor_n, coherencias[n_topics_list.index(mejor_n)], color='red', s=100, zorder=5)

    plt.title(f'Evaluación de Coherencia LDA - {sentimiento.capitalize()}\n'
              f'(Elige el punto donde la curva se estabiliza)')
    plt.xlabel('Número de Temas (k)')
    plt.ylabel('Coherencia (Mayor es mejor)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    ruta = os.path.join(output_dir, f"codo_lda_coherencia_{sentimiento}.png")
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [+] Gráfico guardado en: {ruta}")


# ==========================================
# BARRIDO LDA
# ==========================================
def barrido_lda(corpus_bow, diccionario, textos_tokenizados,
                config_lda, sentimiento="general", output_dir="graficos"):
    """
    Barrido de LDA con Gensim. Devuelve todos los modelos entrenados.
    Lee passes y random_state del JSON (con defaults seguros).

    CAMBIO 3: default de passes subido de 20 → 30.
    Más iteraciones de entrenamiento = modelos más convergidos = coherencia más fiable.
    Si el JSON especifica un valor propio, ese tiene prioridad.
    """
    print(f"\n--- Barrido LDA (Gensim) para: {sentimiento.upper()} ---")

    n_topics_list      = config_lda.get("n_components_range", [2, 3, 4, 5])
    passes             = config_lda.get("passes", [30])[0]
    iterations         = config_lda.get("max_iter", [50])[0]
    random_state       = config_lda.get("random_state", [42])[0]
    metrica_coherencia = config_lda.get("coherence_metric", "c_v")

    resultados = []
    temas_eval  = []
    coherencias = []

    for n_topics in n_topics_list:
        print(f"  -> Entrenando LDA con k={n_topics} | passes={passes} | iter={iterations}...")

        lda = LdaModel(
            corpus=corpus_bow,
            id2word=diccionario,
            num_topics=n_topics,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
        )

        coherence_model = CoherenceModel(
            model=lda,
            texts=textos_tokenizados,
            dictionary=diccionario,
            coherence=metrica_coherencia,
        )
        coherencia = coherence_model.get_coherence()
        print(f"     Coherencia ({metrica_coherencia}): {coherencia:.4f}")

        temas_eval.append(n_topics)
        coherencias.append(coherencia)
        resultados.append({"n_topics": n_topics, "coherencia": coherencia, "modelo": lda})

    mejor_n, _ = encontrar_codo(temas_eval, coherencias)
    plot_codo_lda(temas_eval, coherencias, sentimiento, mejor_n, output_dir=output_dir)

    return resultados


# ==========================================
# PIPELINE PRINCIPAL DE CLUSTERING
# ==========================================
def pipeline_clustering(df, col_texto, json_file, target_col="score", lang='english'):
    print("\n" + "=" * 55)
    print(" 🚀 INICIANDO PIPELINE DE CLUSTERING MULTI-RESULTADO")
    print("=" * 55)

    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config_lda            = config.get("clustering_lda", {})
    rango_topics_original = config_lda.get("n_components_range", [2, 3, 4, 5])

    # Construcción de stopwords combinadas
    stop_words_set = construir_stopwords(lang)

    # 1. Dividir por sentimiento
    df_pos, df_neg, df_neu = dividir_por_sentimiento(df, target_col)
    datasets = {"positivos": df_pos, "negativos": df_neg, "neutros": df_neu}

    # 2. Procesar cada grupo
    for sentimiento, sub_df in datasets.items():
        if len(sub_df) < 10:
            print(f"\n⚠️  Grupo '{sentimiento}' tiene menos de 10 filas. Se omite.")
            continue

        dir_sentimiento = os.path.join("resultados_clustering", sentimiento)
        os.makedirs(dir_sentimiento, exist_ok=True)

        print(f"\n[+] Procesando grupo: {sentimiento.upper()} ({len(sub_df)} docs)")

        textos_crudos      = sub_df[col_texto].fillna("").tolist()
        textos_tokenizados = [limpiar_texto_para_lda(t, stop_words_set) for t in textos_crudos]

        # Filtrar documentos que quedaron vacíos tras la limpieza inicial
        indices_no_vacios  = [i for i, t in enumerate(textos_tokenizados) if len(t) >= 3]
        textos_tokenizados = [textos_tokenizados[i] for i in indices_no_vacios]

        if len(textos_tokenizados) < 5:
            print(f"  ⚠️  Muy pocos documentos válidos en '{sentimiento}' tras limpieza. Se omite.")
            continue

        # corpus_bow y sub_df_valido siempre tendrán el mismo número de filas
        corpus_bow, diccionario, textos_limpios, indices_corpus = preparar_corpus_gensim(
            textos_tokenizados, n_docs=len(textos_tokenizados)
        )

        if len(corpus_bow) < 5:
            print(f"  ⚠️  Corpus demasiado pequeño en '{sentimiento}' tras filter_extremes. Se omite.")
            continue

        indices_finales = [indices_no_vacios[i] for i in indices_corpus]
        sub_df_valido   = sub_df.iloc[indices_finales].copy().reset_index(drop=True)

        # Filtrar temas válidos según tamaño del corpus
        temas_validos = [t for t in rango_topics_original if t <= len(corpus_bow) - 1]
        if not temas_validos:
            print(f"  ⚠️  No hay temas válidos para '{sentimiento}' (corpus={len(corpus_bow)} docs).")
            continue

        config_lda_actual = config_lda.copy()
        config_lda_actual["n_components_range"] = temas_validos

        # Barrido LDA
        todos_los_modelos = barrido_lda(
            corpus_bow, diccionario, textos_limpios,
            config_lda_actual, sentimiento,
            output_dir=os.path.join(dir_sentimiento, "graficos"),
        )

        # 3. Guardar CSVs de resultados
        print(f"  -> Guardando CSVs en: {dir_sentimiento}")

        resumen_coherencias = []

        for res in todos_los_modelos:
            k      = res["n_topics"]
            modelo = res["modelo"]
            coh    = res["coherencia"]

            temas_asignados = []
            for doc_bow in corpus_bow:
                dist     = modelo.get_document_topics(doc_bow, minimum_probability=0)
                topic_id = max(dist, key=lambda x: x[1])[0]
                temas_asignados.append(topic_id)

            df_resultado = sub_df_valido.copy()
            df_resultado['Cluster_LDA'] = temas_asignados

            ruta_csv = os.path.join(dir_sentimiento, f"clusters_k{k}.csv")
            df_resultado.to_csv(ruta_csv, index=False)

            resumen_coherencias.append({"k": k, "coherencia": round(coh, 4)})

        # Guardar resumen de coherencias por sentimiento
        df_resumen = pd.DataFrame(resumen_coherencias)
        df_resumen.to_csv(os.path.join(dir_sentimiento, "resumen_coherencias.csv"), index=False)
        print(f"  -> Resumen de coherencias guardado.")

    print("\n✅ Proceso finalizado. Revisa 'resultados_clustering/'")


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

        target_col    = config_json.get("target", "score")
        text_features = config_json.get("preprocessing", {}).get("text_features", ["content"])
        col_texto     = text_features[0] if text_features else "content"
        lang          = config_json.get("preprocessing", {}).get("language", "english")

        pipeline_clustering(df, col_texto, json_file, target_col, lang)

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)