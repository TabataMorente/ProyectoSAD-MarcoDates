# -*- coding: utf-8 -*-
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel, Phrases
from gensim.models.phrases import Phraser

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find('corpora/stopwords')
except Exception:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords


# ==========================================
# STOPWORDS POR MODO DE N-GRAMA
# ==========================================

# --- Monogramas ---
# Palabras sueltas sin contenido semántico para el dominio Tinder.
DOMAIN_STOPWORDS_UNIGRAM = {
    "app", "use", "like", "get", "make", "time", "really", "just",
    "would", "one", "also", "even", "still", "well", "much", "many",
    "it", "its", "the", "and", "for", "that", "this", "with",
    "have", "has", "was", "are", "not", "but", "all", "very",
    "update", "version", "phone", "review", "rating",
    "apps", "tinder",
    "dating",
    "application",
    "good", "great", "best", "bad", "nice", "better", "cool", "worse",
    "feel", "feels", "seem", "seems", "got", "let", "want", "find",
    "makes", "gets", "keeps", "keep",
    "experience",
    "things", "way", "part", "overall",
    "lot", "lots", "less", "pretty", "quickly", "always", "never",
    "features", "feature",
}

# --- Bigramas ---
DOMAIN_STOPWORDS_BIGRAM = DOMAIN_STOPWORDS_UNIGRAM | {
    # Valoraciones genéricas compuestas
    "pretty_good", "really_good", "really_great", "really_bad",
    "really_like", "really_love", "really_enjoy",
    "good_app", "great_app", "best_app", "bad_app",
    "good_dating", "great_dating",
    "highly_recommend", "would_recommend", "definitely_recommend",
    "five_stars", "star_rating", "one_star",
    # Frases de puntuación/queja sin detalle
    "would_give", "give_stars", "give_five",
    "used_app", "use_app", "using_app",
    "like_app", "love_app",
    # Temporales/proceso
    "long_time", "first_time", "last_time",
    "every_time", "one_time",
    "still_good", "still_bad", "still_working",
    # Genérico de usuario
    "people_use", "user_experience", "customer_service",
    "new_update", "recent_update", "latest_update",
    "overall_good", "overall_great", "overall_bad",
    # Contexto Tinder sin información
    "tinder_app", "tinder_dating", "dating_app", "dating_apps",
    "tinder_gold", "tinder_plus",
    "swipe_right", "swipe_left",
    "profile_picture", "good_experience", "bad_experience",
    "waste_time", "waste_money",
}

# --- Trigramas ---
DOMAIN_STOPWORDS_TRIGRAM = DOMAIN_STOPWORDS_BIGRAM | {
    # Cuantificadores y muletillas de tres palabras
    "lot_of_people", "lot_of_bots", "lot_of_fake",
    "one_of_the", "some_of_the", "most_of_the", "all_of_the",
    "a_lot_of", "bit_of_a",
    # Frases de valoración
    "give_it_five", "give_it_stars", "give_it_one",
    "would_give_five", "would_give_stars",
    "best_dating_app", "good_dating_app", "great_dating_app",
    "would_recommend_this", "highly_recommend_this",
    "waste_of_time", "waste_of_money",
    # Genéricas de experiencia
    "good_user_experience", "overall_good_experience",
    "new_to_app", "first_time_using", "used_app_for",
    "using_app_for", "love_this_app", "like_this_app",
    "hate_this_app", "delete_this_app", "deleted_this_app",
    # Contexto Tinder sin semántica útil
    "tinder_dating_app", "swipe_right_on", "swipe_left_on",
    "tinder_plus_subscription", "tinder_gold_subscription",
    "pay_for_tinder", "paid_for_tinder",
    "match_with_people", "match_with_someone",
    # Quejas genéricas de soporte
    "contact_customer_service", "reach_customer_service",
    "customer_service_bad", "customer_service_good",
    "app_does_not", "app_does_not_work", "app_not_working",
    "does_not_work", "not_work_properly", "not_working_properly",
}


def get_domain_stopwords(ngram_mode: str) -> set:
    """
    Devuelve el conjunto de stopwords de dominio apropiado según el modo.
    Los trigramas y bigramas incluyen los monogramas por herencia.
    """
    mode = ngram_mode.lower()
    if mode == "trigram":
        return DOMAIN_STOPWORDS_TRIGRAM
    elif mode == "bigram":
        return DOMAIN_STOPWORDS_BIGRAM
    else:  # unigram (default)
        return DOMAIN_STOPWORDS_UNIGRAM


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


def construir_stopwords(lang='english', ngram_mode='unigram'):
    """Combina stopwords de NLTK con stopwords de dominio según el modo n-grama."""
    try:
        sw = set(stopwords.words(lang))
    except Exception:
        sw = set()
    return sw | get_domain_stopwords(ngram_mode)


# ==========================================
# CONSTRUCCIÓN DE N-GRAMAS CON GENSIM Phrases
# ==========================================
def construir_ngramas(textos_tokenizados, ngram_mode: str, stopwords_compuestas: set):
    """
    Dado una lista de listas de tokens (monogramas limpios), aplica
    Gensim Phrases para construir bi- o trigramas según ngram_mode.

    Los n-gramas resultantes que aparezcan en stopwords_compuestas se eliminan.
    Devuelve la lista de documentos con los n-gramas insertados.
    """
    mode = ngram_mode.lower()
    if mode == "unigram":
        return textos_tokenizados  # sin cambios

    print(f"  -> Construyendo {mode}s con Gensim Phrases...")

    # Bigramas
    bigram_phrases = Phrases(textos_tokenizados, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram_phrases)
    textos_bigrama = [bigram_phraser[doc] for doc in textos_tokenizados]

    if mode == "bigram":
        textos_filtrados = [
            [t for t in doc if t not in stopwords_compuestas]
            for doc in textos_bigrama
        ]
        return textos_filtrados

    # Trigramas (encadenado sobre los bigramas)
    trigram_phrases = Phrases(textos_bigrama, min_count=5, threshold=10)
    trigram_phraser = Phraser(trigram_phrases)
    textos_trigrama = [trigram_phraser[doc] for doc in textos_bigrama]

    textos_filtrados = [
        [t for t in doc if t not in stopwords_compuestas]
        for doc in textos_trigrama
    ]
    return textos_filtrados


# ==========================================
# PREPARACIÓN DEL CORPUS PARA GENSIM
# ==========================================
def preparar_corpus_gensim(textos_tokenizados, n_docs):
    """
    Recibe lista de listas de tokens ya limpios y construye
    el diccionario y corpus BoW para Gensim.
    filter_extremes se adapta al tamaño del corpus.
    """
    diccionario = corpora.Dictionary(textos_tokenizados)

    no_below = max(2, int(n_docs * 0.01))
    no_above = 0.70 if n_docs < 200 else 0.55

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
    x = np.array(x)
    y = np.array(y)

    if len(x) == 1:
        return x[0], 0

    idx_max = np.argmax(y)
    if idx_max != len(y) - 1:
        return x[idx_max], idx_max

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
    print(f"\n--- Barrido LDA (Gensim) para: {sentimiento.upper()} ---")

    n_topics_list      = config_lda.get("n_components_range", [2, 3, 4, 5])
    passes             = config_lda.get("passes", [30])[0]
    iterations         = config_lda.get("max_iter", [50])[0]
    random_state       = config_lda.get("random_state", [42])[0]
    metrica_coherencia = config_lda.get("coherence_metric", "c_v")

    resultados  = []
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
# GUARDAR CSV RESUMEN DE TEMAS
# ==========================================
def guardar_resumen_temas(modelo, k, corpus_bow, sub_df_valido, col_texto,
                          dir_resumen, n_palabras=10, n_top_resenas=10):
    os.makedirs(dir_resumen, exist_ok=True)

    dist_completa = [
        dict(modelo.get_document_topics(doc, minimum_probability=0))
        for doc in corpus_bow
    ]

    filas = []
    for topic_id in range(k):
        palabras_raw   = modelo.show_topic(topic_id, topn=n_palabras)
        palabras_clave = ", ".join([w for w, _ in palabras_raw])

        probs_tema = np.array([d.get(topic_id, 0.0) for d in dist_completa])
        idx_top    = np.argsort(probs_tema)[::-1][:n_top_resenas]

        fila = {
            "cluster_id":     topic_id,
            "palabras_clave": palabras_clave,
        }
        for rank, idx in enumerate(idx_top, start=1):
            texto_resena = sub_df_valido.iloc[idx][col_texto] if idx < len(sub_df_valido) else ""
            prob_resena  = round(float(probs_tema[idx]), 4)
            fila[f"top_resena_{rank}"]      = texto_resena
            fila[f"top_resena_{rank}_prob"] = prob_resena

        filas.append(fila)

    df_temas = pd.DataFrame(filas)
    ruta = os.path.join(dir_resumen, f"resumen_temas_k{k}.csv")
    df_temas.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"    [+] Resumen de temas guardado en: {ruta}")


# ==========================================
# GUARDAR CSV DISTRIBUCIÓN DE DOCUMENTOS
# ==========================================
def guardar_distribucion_docs(modelo, k, corpus_bow, sub_df_valido,
                              dir_distribucion):
    os.makedirs(dir_distribucion, exist_ok=True)

    registros = []
    for i, doc_bow in enumerate(corpus_bow):
        dist     = dict(modelo.get_document_topics(doc_bow, minimum_probability=0))
        probs    = {f"cluster_{t}": round(dist.get(t, 0.0), 4) for t in range(k)}
        asignado = max(dist, key=dist.get) if dist else 0
        registros.append({**probs, "Cluster_asignado": asignado})

    df_dist  = pd.DataFrame(registros)
    df_final = pd.concat([sub_df_valido.reset_index(drop=True), df_dist], axis=1)

    ruta = os.path.join(dir_distribucion, f"distribucion_docs_k{k}.csv")
    df_final.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"    [+] Distribución de documentos guardada en: {ruta}")


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

    # ── Leer modo n-grama del JSON ─────────────────────────────────────────
    # Valores válidos: "unigram" | "bigram" | "trigram"
    ngram_mode = config_lda.get("ngram_mode", "unigram").lower()
    if ngram_mode not in ("unigram", "bigram", "trigram"):
        print(f"⚠️  ngram_mode '{ngram_mode}' no reconocido. Se usará 'unigram'.")
        ngram_mode = "unigram"

    print(f"\n📐 Modo n-grama seleccionado: {ngram_mode.upper()}")

    # La carpeta de resultados refleja el modo para no mezclar ejecuciones
    carpeta_base = f"resultados_clustering_{ngram_mode}s"

    # Stopwords de tokens individuales (incluye las del modo)
    stop_words_set = construir_stopwords(lang, ngram_mode)
    # Stopwords de expresiones compuestas (bi/trigramas a eliminar tras Phrases)
    stopwords_compuestas = get_domain_stopwords(ngram_mode)

    # 1. Dividir por sentimiento
    df_pos, df_neg, df_neu = dividir_por_sentimiento(df, target_col)
    datasets = {"positivos": df_pos, "negativos": df_neg, "neutros": df_neu}

    # 2. Procesar cada grupo
    for sentimiento, sub_df in datasets.items():
        if len(sub_df) < 10:
            print(f"\n⚠️  Grupo '{sentimiento}' tiene menos de 10 filas. Se omite.")
            continue

        dir_sentimiento  = os.path.join(carpeta_base, sentimiento)
        dir_graficos     = os.path.join(dir_sentimiento, "graficos")
        dir_resumen      = os.path.join(dir_sentimiento, "resumen_temas")
        dir_distribucion = os.path.join(dir_sentimiento, "distribucion_docs")

        for d in [dir_graficos, dir_resumen, dir_distribucion]:
            os.makedirs(d, exist_ok=True)

        print(f"\n[+] Procesando grupo: {sentimiento.upper()} ({len(sub_df)} docs)")

        textos_crudos = sub_df[col_texto].fillna("").tolist()

        # Tokenización base (monogramas limpios, con stopwords del modo)
        textos_tokenizados = [limpiar_texto_para_lda(t, stop_words_set) for t in textos_crudos]

        # Filtrar documentos vacíos tras limpieza inicial
        indices_no_vacios  = [i for i, t in enumerate(textos_tokenizados) if len(t) >= 3]
        textos_tokenizados = [textos_tokenizados[i] for i in indices_no_vacios]

        if len(textos_tokenizados) < 5:
            print(f"  ⚠️  Muy pocos documentos válidos en '{sentimiento}' tras limpieza. Se omite.")
            continue

        # Construir bi/trigramas y filtrar expresiones compuestas irrelevantes
        textos_tokenizados = construir_ngramas(
            textos_tokenizados, ngram_mode, stopwords_compuestas
        )

        corpus_bow, diccionario, textos_limpios, indices_corpus = preparar_corpus_gensim(
            textos_tokenizados, n_docs=len(textos_tokenizados)
        )

        if len(corpus_bow) < 5:
            print(f"  ⚠️  Corpus demasiado pequeño en '{sentimiento}' tras filter_extremes. Se omite.")
            continue

        indices_finales = [indices_no_vacios[i] for i in indices_corpus]
        sub_df_valido   = sub_df.iloc[indices_finales].copy().reset_index(drop=True)

        temas_validos = [t for t in rango_topics_original if t <= len(corpus_bow) - 1]
        if not temas_validos:
            print(f"  ⚠️  No hay temas válidos para '{sentimiento}' (corpus={len(corpus_bow)} docs).")
            continue

        config_lda_actual = config_lda.copy()
        config_lda_actual["n_components_range"] = temas_validos

        todos_los_modelos = barrido_lda(
            corpus_bow, diccionario, textos_limpios,
            config_lda_actual, sentimiento,
            output_dir=dir_graficos,
        )

        print(f"  -> Guardando resultados en subcarpetas de: {dir_sentimiento}")

        resumen_coherencias = []

        for res in todos_los_modelos:
            k      = res["n_topics"]
            modelo = res["modelo"]
            coh    = res["coherencia"]

            guardar_resumen_temas(
                modelo, k, corpus_bow, sub_df_valido, col_texto,
                dir_resumen, n_palabras=10, n_top_resenas=10,
            )
            guardar_distribucion_docs(
                modelo, k, corpus_bow, sub_df_valido,
                dir_distribucion,
            )
            resumen_coherencias.append({"k": k, "coherencia": round(coh, 4)})

        df_resumen_coh = pd.DataFrame(resumen_coherencias)
        df_resumen_coh.to_csv(
            os.path.join(dir_sentimiento, "resumen_coherencias.csv"),
            index=False, encoding="utf-8-sig",
        )
        print(f"  -> Resumen de coherencias guardado en: {dir_sentimiento}")

    print(f"\n✅ Proceso finalizado. Revisa '{carpeta_base}/'")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python clustering.py <data.csv> <config_file.json>")
        sys.exit(1)

    csv_data  = sys.argv[1]
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