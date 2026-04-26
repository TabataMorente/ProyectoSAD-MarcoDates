# -*- coding: utf-8 -*-
import sys
import signal
import re
import pandas as pd
import numpy as np
import nltk
import os

# 1. Forzamos la entrada de la carpeta de librerías al principio de la lista
ruta_venv = os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages')
if ruta_venv not in sys.path:
    sys.path.insert(0, ruta_venv)

# 2. Ahora intentamos el import
try:
    import emoji
    print("✅ Módulo 'emoji' cargado con éxito mediante bypass de ruta.")
except ImportError:
    # Si lo anterior falla, probamos con la ruta absoluta que verificamos antes
    sys.path.insert(0, r'C:\Users\tabat\PycharmProjects\ProyectoSAD-MarcoDates\.venv\Lib\site-packages')
    import emoji
    print("✅ Módulo 'emoji' cargado mediante ruta absoluta.")
from scipy.sparse import hstack, csr_matrix

# Herramientas de Scikit-Learn e Imblearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from nltk import SnowballStemmer

# ==========================================
# CONFIGURACIÓN INICIAL DE NLTK Y SEÑALES
# ==========================================
try:
    nltk.data.find('corpora/stopwords')
except (LookupError, AttributeError):
    print("📥 Descargando diccionario de stopwords...")
    nltk.download('stopwords', quiet=True)

def signal_handler(sig, frame):
    print("\n🛑 Saliendo del programa...")
    sys.exit(0)


# ==========================================
# 1. ELIMINAR DUPLICADOS
# ==========================================
def eliminar_duplicados(df_train, df_test, df_dev, config):
    if config.get("drop_duplicates"):
        df_train = df_train.drop_duplicates()
        df_test = df_test.drop_duplicates()
        df_dev = df_dev.drop_duplicates()
        print(" -> Filas duplicadas eliminadas.")
    return df_train, df_test, df_dev


# ==========================================
# 2. ELIMINAR COLUMNAS
# ==========================================
def eliminar_columnas(df_train, df_test, df_dev, config):
    columnas_a_borrar = config.get("drop_columns", [])
    if columnas_a_borrar:
        df_train = df_train.drop(columns=[c for c in columnas_a_borrar if c in df_train.columns])
        df_test = df_test.drop(columns=[c for c in columnas_a_borrar if c in df_test.columns])
        df_dev = df_dev.drop(columns=[c for c in columnas_a_borrar if c in df_dev.columns])
        print(f" -> Columnas eliminadas: {columnas_a_borrar}")
    return df_train, df_test, df_dev


# ==========================================
# 3. ASIGNAR TIPOS EXPLÍCITOS
# ==========================================
def asignar_tipos(df_train, df_test, df_dev, config):
    tipos_json = config.get("categoria", [])
    if not tipos_json:
        return df_train, df_test, df_dev

    for col, tipo in zip(df_train.columns, tipos_json):
        if tipo == "int":
            df_train[col] = df_train[col].astype('Int64')
            df_test[col] = df_test[col].astype('Int64')
            df_dev[col] = df_dev[col].astype('Int64')
        elif tipo == "double":
            df_train[col] = df_train[col].astype('float64')
            df_test[col] = df_test[col].astype('float64')
            df_dev[col] = df_dev[col].astype('float64')
        elif tipo in ("string", "category"):
            df_train[col] = df_train[col].astype('category')
            df_test[col] = df_test[col].astype('category')
            df_dev[col] = df_dev[col].astype('category')
        elif tipo == "text":
            df_train[col] = df_train[col].astype('string')
            df_test[col] = df_test[col].astype('string')
            df_dev[col] = df_dev[col].astype('string')

    print(" -> Tipos de datos asignados según config.")
    return df_train, df_test, df_dev


# ==========================================
# 4. VALORES ERRÓNEOS
# ==========================================
def tratar_valores_erroneos(df_train, df_test, df_dev, config):
    config_err = config.get("erroneous_values")
    if not config_err or config_err.get("action") == "none":
        return df_train, df_test, df_dev

    accion_global = config_err.get("action")
    reglas = config_err.get("rules", {})

    def aplicar_limpieza(df, df_ref):
        df_res = df.copy()
        for col, regla in reglas.items():
            if col not in df_res.columns:
                continue
            condiciones = regla.get("conditions", [])
            estrategia = regla.get("strategy", "none")
            mask = pd.Series(False, index=df_res.index)

            for cond in condiciones:
                t, v = cond.get("type"), cond.get("value")
                if t == "less_than":      mask |= (df_res[col] < v)
                elif t == "greater_than": mask |= (df_res[col] > v)
                elif t == "equals":       mask |= (df_res[col] == v)
                elif t == "in_list":      mask |= (df_res[col].isin(v))
                elif t == "regex":        mask |= df_res[col].astype(str).str.contains(v, na=False, regex=True)
                elif t == "has_decimals":
                    if pd.api.types.is_numeric_dtype(df_res[col]) and v is True:
                        mask |= (df_res[col].notna() & (df_res[col] % 1 != 0))

            if mask.any():
                if accion_global == "delete":
                    df_res = df_res[~mask]
                elif accion_global == "impute":
                    if estrategia == "mean" and pd.api.types.is_numeric_dtype(df_ref[col]):
                        val = int(round(df_ref[col].mean())) if pd.api.types.is_integer_dtype(df_ref[col]) else df_ref[col].mean()
                        df_res.loc[mask, col] = val
                    elif estrategia == "median" and pd.api.types.is_numeric_dtype(df_ref[col]):
                        val = int(round(df_ref[col].median())) if pd.api.types.is_integer_dtype(df_ref[col]) else df_ref[col].median()
                        df_res.loc[mask, col] = val
                    elif estrategia == "mode":
                        if not df_ref[col].mode().empty:
                            df_res.loc[mask, col] = df_ref[col].mode()[0]
        return df_res

    df_train = aplicar_limpieza(df_train, df_train)
    df_test  = aplicar_limpieza(df_test, df_train)
    df_dev   = aplicar_limpieza(df_dev, df_train)
    print(" -> Valores erróneos tratados.")
    return df_train, df_test, df_dev


# ==========================================
# 5. CODIFICAR OBJETIVO
# ==========================================
def codificar_objetivo(df_train, df_test, df_dev, config):
    target = config.get("target")
    if target and target in df_train.columns:
        if not pd.api.types.is_numeric_dtype(df_train[target]):
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            df_test[target] = df_test[target].map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )
            df_dev[target] = df_dev[target].map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )
            print(f" -> Variable objetivo '{target}' codificada a números.")
    return df_train, df_test, df_dev


# ==========================================
# 6. TRADUCCIÓN DE EMOJIS A TEXTO
# ==========================================
def traducir_emojis(df_train, df_test, df_dev, config, target):
    text_cols = config.get('text_features', [])
    if not text_cols:
        return df_train, df_test, df_dev

    def limpiar_texto(texto):
        if not isinstance(texto, str):
            return ""
        return emoji.demojize(texto, delimiters=(" ", " "))

    for col in text_cols:
        if col in df_train.columns and col != target:
            print(f" ✨ Traduciendo emojis a texto en: {col}")
            df_train[col] = df_train[col].apply(limpiar_texto)
            df_test[col]  = df_test[col].apply(limpiar_texto)
            df_dev[col]   = df_dev[col].apply(limpiar_texto)
    return df_train, df_test, df_dev


# ==========================================
# NUEVO: FEATURES DE TEXTO ANTES DE LIMPIAR
# Extrae longitud y conteo de caracteres mientras
# el texto todavía es legible (antes de stemming).
# ==========================================
def extraer_features_texto(df_train, df_test, df_dev, config, target):
    """
    Extrae features numéricas derivadas del texto crudo:
    - word_count: número de palabras
    - char_count: número de caracteres
    Estas features se añaden como columnas numéricas antes de vectorizar.
    """
    text_cols = config.get('text_features', [])
    if not text_cols:
        return df_train, df_test, df_dev

    for col in text_cols:
        if col not in df_train.columns or col == target:
            continue
        for df in (df_train, df_test, df_dev):
            texto = df[col].fillna("").astype(str)
            df[f"{col}_word_count"] = texto.str.split().str.len()
            df[f"{col}_char_count"] = texto.str.len()

    print(" -> Features de longitud de texto extraídas (word_count, char_count).")
    return df_train, df_test, df_dev


# ==========================================
# NUEVO: FEATURES DE FECHA
# Extrae mes, año y día de semana de columnas de fecha.
# ==========================================
def extraer_features_fecha(df_train, df_test, df_dev, config):
    """
    Lee 'date_features' del config (lista de columnas de fecha)
    y extrae: año, mes, día de semana como features numéricas.
    Elimina la columna de fecha original tras la extracción.
    Ejemplo en JSON: "date_features": ["at"]
    """
    date_cols = config.get('date_features', [])
    if not date_cols:
        return df_train, df_test, df_dev

    for col in date_cols:
        for df in (df_train, df_test, df_dev):
            if col not in df.columns:
                continue
            parsed = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_year"]    = parsed.dt.year.fillna(0).astype(int)
            df[f"{col}_month"]   = parsed.dt.month.fillna(0).astype(int)
            df[f"{col}_weekday"] = parsed.dt.dayofweek.fillna(0).astype(int)
            df.drop(columns=[col], inplace=True)

    print(f" -> Features de fecha extraídas de: {date_cols}")
    return df_train, df_test, df_dev


# ==========================================
# 7. LIMPIEZA DE TEXTO (sin stemming para TF-IDF)
# ==========================================
def limpiar_y_normalizar_texto(df_train, df_test, df_dev, config, target):
    text_cols = config.get('text_features', [])
    if not text_cols:
        return df_train, df_test, df_dev

    lang = config.get('language', 'english')
    usar_stemming = config.get('use_stemming', False)

    # Stopwords de dominio general + dominio de apps de citas
    DOMAIN_STOPWORDS = {
        "app", "use", "like", "get", "make", "time", "really", "just",
        "would", "one", "also", "even", "still", "well", "much", "many",
        "good", "great", "nice", "love", "hate", "bad", "awful",
    }

    try:
        stop_words = set(stopwords.words(lang))
        # Preservamos negaciones: son críticas para sentimiento
        negaciones = {"no", "not", "nor", "against", "isn't", "aren't", "didn't",
                      "won't", "doesn't", "don't", "can't", "couldn't", "wouldn't"}
        stop_words = (stop_words - negaciones) | DOMAIN_STOPWORDS
        stemmer = SnowballStemmer('spanish' if lang == 'spanish' else 'english') if usar_stemming else None
    except Exception:
        stop_words = set()
        stemmer = None

    def procesar_celda(texto):
        if not isinstance(texto, str):
            return ""
        texto = texto.lower().strip()
        # Elimina puntuación pero conserva apóstrofes (isn't, don't)
        texto = re.sub(r"[^\w\s']", '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()

        palabras = texto.split()
        if stemmer:
            # Con stemming: reduce palabras al lexema
            palabras = [stemmer.stem(p) for p in palabras if p not in stop_words]
        else:
            # Sin stemming: solo quita stopwords, preserva morfología natural
            # Esto permite que TF-IDF capture bigramas reales como "not good", "highly recommend"
            palabras = [p for p in palabras if p not in stop_words and len(p) > 1]

        return " ".join(palabras)

    for col in text_cols:
        if col in df_train.columns and col != target:
            modo = "Stemming" if usar_stemming else "Limpieza (sin stemming)"
            print(f" 🧹 {modo} en: {col}")
            df_train[col] = df_train[col].apply(procesar_celda)
            df_test[col]  = df_test[col].apply(procesar_celda)
            df_dev[col]   = df_dev[col].apply(procesar_celda)

    return df_train, df_test, df_dev


# ==========================================
# 8. VECTORIZACIÓN DE TEXTO (TF-IDF / BoW)
# ==========================================
def procesar_texto(df_train, df_test, df_dev, config, target):
    text_cols = config.get('text_features', [])
    if not text_cols:
        return df_train, df_test, df_dev

    method = config.get('text_process_method', 'tf-idf')
    ngram_range_list = config.get('ngram_range', [1, 1])
    ngram_setting = tuple(ngram_range_list)

    limite_raw = config.get('limite_palabras', None)
    if str(limite_raw).lower() == 'none' or limite_raw is None:
        limite_palabras = None
    else:
        limite_palabras = int(limite_raw)

    # min_df configurable (antes estaba hardcodeado a 3)
    min_df = config.get('min_df', 2)

    # Activar vectorizador de subwords (char_wb) para capturar errores ortográficos
    usar_charwb = config.get('use_char_vectorizer', False)

    for col in text_cols:
        if col not in df_train.columns or col == target:
            continue

        textos_train = df_train[col].fillna("").astype(str)
        textos_test  = df_test[col].fillna("").astype(str)
        textos_dev   = df_dev[col].fillna("").astype(str)

        if method == 'tf-idf':
            vec_word = TfidfVectorizer(
                max_features=limite_palabras,
                min_df=min_df,
                ngram_range=ngram_setting,
                sublinear_tf=True,       # log(1+tf) — mejora para texto largo
            )
        elif method == 'bow':
            vec_word = CountVectorizer(
                max_features=limite_palabras,
                min_df=min_df,
                ngram_range=ngram_setting,
            )
        elif method == 'one-hot':
            vec_word = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            print(f" ⚠️  Método de vectorización desconocido: {method}. Usando tf-idf.")
            vec_word = TfidfVectorizer(max_features=limite_palabras, min_df=min_df, ngram_range=ngram_setting, sublinear_tf=True)

        # --- Vectorización de palabras ---
        if method == 'one-hot':
            t_train = vec_word.fit_transform(df_train[[col]])
            t_test  = vec_word.transform(df_test[[col]])
            t_dev   = vec_word.transform(df_dev[[col]])
        else:
            t_train = vec_word.fit_transform(textos_train)
            t_test  = vec_word.transform(textos_test)
            t_dev   = vec_word.transform(textos_dev)

        if usar_charwb and method in ('tf-idf', 'bow'):
            limite_char = int(limite_palabras * 0.3) if limite_palabras else 5000
            vec_char = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=limite_char,
                min_df=min_df,
                sublinear_tf=True,
            )
            c_train = vec_char.fit_transform(textos_train)
            c_test  = vec_char.transform(textos_test)
            c_dev   = vec_char.transform(textos_dev)

            # Concatenamos ambas matrices dispersas antes de densificar
            t_train = hstack([t_train, c_train])
            t_test  = hstack([t_test,  c_test])
            t_dev   = hstack([t_dev,   c_dev])
            print(f" -> Vectorizador char_wb (subwords) añadido: {c_train.shape[1]} features.")

        if hasattr(t_train, "toarray"):
            t_train = t_train.toarray()
            t_test  = t_test.toarray()
            t_dev   = t_dev.toarray()

        n_feats = t_train.shape[1]
        cols_names = [f"{col}_{i}" for i in range(n_feats)]
        df_t_train = pd.DataFrame(t_train, columns=cols_names, index=df_train.index)
        df_t_test  = pd.DataFrame(t_test,  columns=cols_names, index=df_test.index)
        df_t_dev   = pd.DataFrame(t_dev,   columns=cols_names, index=df_dev.index)

        df_train = pd.concat([df_train.drop(columns=[col]), df_t_train], axis=1)
        df_test  = pd.concat([df_test.drop(columns=[col]),  df_t_test],  axis=1)
        df_dev   = pd.concat([df_dev.drop(columns=[col]),   df_t_dev],   axis=1)

        print(f" -> Texto '{col}' vectorizado con {method} | ngram={ngram_setting} | features={n_feats} | max_features={limite_palabras}")

    return df_train, df_test, df_dev


# ==========================================
# 9. CODIFICACIÓN CATEGÓRICA Y BOOLEANA
# ==========================================
def codificar_variables(df_train, df_test, df_dev, config, target):
    estrategia = config.get("categorical_encoding", "none")
    # Solo columnas categóricas/object que NO sean columnas de texto a vectorizar
    text_cols = config.get('text_features', [])
    categorical_cols = df_train.select_dtypes(include=['category', 'object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target and c not in text_cols]

    if estrategia == "none" or not categorical_cols:
        return df_train, df_test, df_dev

    if estrategia == "one-hot":
        df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True, dtype=int)
        df_test  = pd.get_dummies(df_test,  columns=categorical_cols, drop_first=True, dtype=int)
        df_dev   = pd.get_dummies(df_dev,   columns=categorical_cols, drop_first=True, dtype=int)
        # Alineación: test y dev heredan exactamente las columnas de train
        df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)
        df_train, df_dev  = df_train.align(df_dev,  join='left', axis=1, fill_value=0)

    elif estrategia == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            df_test[col]  = df_test[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )
            df_dev[col]   = df_dev[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )

    print(f" -> Variables categóricas codificadas usando: {estrategia}.")
    return df_train, df_test, df_dev


# ==========================================
# 10. IMPUTACIÓN DE NULOS
# ==========================================
def tratar_nulos(df_train, df_test, df_dev, config, target):
    accion = config.get('missing_values')
    if accion == 'impute':
        strategy = config.get('impute_strategy', 'mean')

        # Primero eliminamos filas sin target (no hay nada que aprender)
        df_train = df_train.dropna(subset=[target])
        df_test  = df_test.dropna(subset=[target])
        df_dev   = df_dev.dropna(subset=[target])

        # Solo imputamos columnas numéricas (el texto ya fue manejado antes)
        cols_input = [
            c for c in df_train.columns
            if c != target and pd.api.types.is_numeric_dtype(df_train[c])
        ]
        if cols_input:
            imputer = SimpleImputer(strategy=strategy)
            df_train[cols_input] = imputer.fit_transform(df_train[cols_input])
            df_test[cols_input]  = imputer.transform(df_test[cols_input])
            df_dev[cols_input]   = imputer.transform(df_dev[cols_input])
        print(f" -> Nulos imputados en características numéricas (Estrategia: {strategy}).")

    elif accion == 'delete':
        df_train = df_train.dropna()
        df_test  = df_test.dropna()
        df_dev   = df_dev.dropna()
        print(" -> Filas con nulos eliminadas.")

    return df_train, df_test, df_dev


# ==========================================
# 11. OUTLIERS (Clipping con IQR)
# ==========================================
def tratar_outliers(df_train, df_test, df_dev, config, target):
    if config.get('outliers') == 'clip':
        text_cols = config.get('text_features', [])
        num_cols = df_train.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            # No clippeamos el target ni columnas derivadas del texto (ya normalizadas)
            if col == target or any(col.startswith(f"{t}_") for t in text_cols):
                continue
            Q1  = df_train[col].quantile(0.25)
            Q3  = df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            inf, sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_train[col] = np.clip(df_train[col], inf, sup)
            df_test[col]  = np.clip(df_test[col],  inf, sup)
            df_dev[col]   = np.clip(df_dev[col],   inf, sup)

        print(" -> Outliers tratados (Clipping IQR).")
    return df_train, df_test, df_dev


# ==========================================
# 12. DISCRETIZACIÓN Y ESCALADO
# ==========================================
def escalar_y_discretizar(df_train, df_test, df_dev, config, target):
    disc_cols = config.get('discretize_features', [])
    if disc_cols:
        n   = config.get('discretize_bins', 3)
        est = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='uniform')
        for col in disc_cols:
            if col in df_train.columns and col != target:
                df_train[col] = est.fit_transform(df_train[[col]])
                df_test[col]  = est.transform(df_test[[col]])
                df_dev[col]   = est.transform(df_dev[[col]])

    method = config.get('scaling')
    if method in ('min-max', 'max-min'):
        scaler = MinMaxScaler()
    elif method == 'z-score':
        scaler = StandardScaler()
    else:
        return df_train, df_test, df_dev

    text_cols    = config.get('text_features', [])
    cols_numericas = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cols_a_escalar = [
        c for c in cols_numericas
        if c != target and not any(c.startswith(f"{t}_") for t in text_cols)
    ]

    if cols_a_escalar:
        df_train[cols_a_escalar] = scaler.fit_transform(df_train[cols_a_escalar])
        df_test[cols_a_escalar]  = scaler.transform(df_test[cols_a_escalar])
        df_dev[cols_a_escalar]   = scaler.transform(df_dev[cols_a_escalar])
        print(f" -> Datos escalados con {method}.")

    return df_train, df_test, df_dev


# ==========================================
# 13. BALANCEO DE DATOS (Solo en Train)
# ==========================================
def balancear_clases(df_train, config, target):
    strat = config.get("sampling_strategy", "none")
    ratio = config.get("sampling_ratio", "auto")
    if isinstance(ratio, dict):
        ratio = {int(k): v for k, v in ratio.items()}

    if strat == "none" or not target or target not in df_train.columns:
        return df_train

    X = df_train.drop(columns=[target])
    y = df_train[target]

    if strat == "oversample":
        res = RandomOverSampler(random_state=42, sampling_strategy=ratio)
    elif strat == "undersample":
        res = RandomUnderSampler(random_state=42, sampling_strategy=ratio)
    elif strat == "SMOTE":
        res = SMOTE(random_state=42, sampling_strategy=ratio)
    else:
        return df_train

    X_res, y_res = res.fit_resample(X, y)
    df_train = pd.concat([X_res, y_res], axis=1)
    print(f" -> Balanceo aplicado en Train ({strat}).")
    return df_train


# ==========================================
# AGRUPAR TARGET EN SENTIMIENTOS
# ==========================================
def agrupar_sentimiento_target(df_train, df_test, df_dev, target):
    if not target or target not in df_train.columns:
        return df_train, df_test, df_dev

    def mapear_sentimiento(valor):
        try:
            v = int(float(valor))
            if v in [1, 2]:   return 'negativo'
            elif v == 3:       return 'neutro'
            elif v in [4, 5]:  return 'positivo'
            else:              return valor
        except Exception:
            return valor

    print(f" -> Agrupando target '{target}' en: negativo, neutro y positivo.")
    df_train[target] = df_train[target].apply(mapear_sentimiento)
    df_test[target]  = df_test[target].apply(mapear_sentimiento)
    df_dev[target]   = df_dev[target].apply(mapear_sentimiento)

    return df_train, df_test, df_dev


# ==========================================
# PIPELINE PRINCIPAL UNIFICADO
# ORDEN CORREGIDO:
#   1. Limpieza básica (duplicados, columnas, tipos, erróneos)
#   2. Target (agrupación + codificación)
#   3. Emojis → features de texto → features de fecha
#   4. Codificación categórica (DESPUÉS de emojis, ANTES del texto)
#   5. Limpieza de texto (stopwords, sin stemming por defecto)
#   6. Vectorización TF-IDF / BoW
#   7. Tratamiento estadístico (nulos sobre numéricas ya limpias)
#   8. Outliers → escalado
#   9. Balanceo (último, sobre el dataset completamente numérico)
# ==========================================
def pipeline_preprocesamiento(df_train, df_test, df_dev, config_full):
    target_global = config_full.get("target")
    config_prep   = config_full.get("preprocessing", {})

    signal.signal(signal.SIGINT, signal_handler)
    print("\n--- 🛠️ INICIANDO PIPELINE DE PREPROCESADO ---")

    # 1. Limpieza básica
    df_train, df_test, df_dev = eliminar_duplicados(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = eliminar_columnas(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = asignar_tipos(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = tratar_valores_erroneos(df_train, df_test, df_dev, config_prep)

    # 2. Target
    df_train, df_test, df_dev = agrupar_sentimiento_target(df_train, df_test, df_dev, target_global)
    df_train, df_test, df_dev = codificar_objetivo(df_train, df_test, df_dev, config_full)

    # 3. Texto crudo → emojis → features de longitud → features de fecha
    df_train, df_test, df_dev = traducir_emojis(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = extraer_features_texto(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = extraer_features_fecha(df_train, df_test, df_dev, config_prep)

    # 4. Codificación categórica (ANTES de vectorizar texto)
    df_train, df_test, df_dev = codificar_variables(df_train, df_test, df_dev, config_prep, target_global)

    # 5. Limpieza y normalización de texto (sin stemming por defecto)
    df_train, df_test, df_dev = limpiar_y_normalizar_texto(df_train, df_test, df_dev, config_prep, target_global)

    # 6. Vectorización
    df_train, df_test, df_dev = procesar_texto(df_train, df_test, df_dev, config_prep, target_global)

    # 7. Nulos (solo sobre columnas numéricas tras vectorización)
    df_train, df_test, df_dev = tratar_nulos(df_train, df_test, df_dev, config_prep, target_global)

    # 8. Outliers y escalado (nunca sobre features TF-IDF)
    df_train, df_test, df_dev = tratar_outliers(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = escalar_y_discretizar(df_train, df_test, df_dev, config_prep, target_global)

    # 9. Balanceo (último paso, sobre datos completamente numéricos)
    df_train = balancear_clases(df_train, config_prep, target_global)

    print("--- ✅ PIPELINE DE PREPROCESADO COMPLETADO ---\n")
    return df_train, df_test, df_dev