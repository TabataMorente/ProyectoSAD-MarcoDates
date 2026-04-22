# -*- coding: utf-8 -*-
import sys
import signal
import re
import pandas as pd
import numpy as np
import nltk
# import emoji

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
        # Solo borramos si la columna existe en el DataFrame
        df_train = df_train.drop(columns=[c for c in columnas_a_borrar if c in df_train.columns])
        df_test = df_test.drop(columns=[c for c in columnas_a_borrar if c in df_test.columns])
        df_dev = df_dev.drop(columns=[c for c in columnas_a_borrar if c in df_dev.columns])
        print(f" -> Columnas eliminadas: {columnas_a_borrar}")
    return df_train, df_test, df_dev


# ==========================================
# 3. ASIGNAR TIPOS EXPLICITOS (Nuevo)
# ==========================================
def asignar_tipos(df_train, df_test, df_dev, config):
    tipos_json = config.get("categoria", [])
    if not tipos_json: return df_train, df_test, df_dev

    for col, tipo in zip(df_train.columns, tipos_json):
        if tipo == "int":
            df_train[col] = df_train[col].astype('Int64')
            df_test[col] = df_test[col].astype('Int64')
            df_dev[col] = df_dev[col].astype('Int64')
        elif tipo == "double":
            df_train[col] = df_train[col].astype('float64')
            df_test[col] = df_test[col].astype('float64')
            df_dev[col] = df_dev[col].astype('float64')
        elif tipo == "string" or tipo == "category":
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
# 4. VALORES ERRÓNEOS (Mejorado con in_list y decimales)
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
            if col not in df_res.columns: continue

            condiciones = regla.get("conditions", [])
            estrategia = regla.get("strategy", "none")
            mask = pd.Series(False, index=df_res.index)

            for cond in condiciones:
                t, v = cond.get("type"), cond.get("value")
                if t == "less_than": mask |= (df_res[col] < v)
                elif t == "greater_than": mask |= (df_res[col] > v)
                elif t == "equals": mask |= (df_res[col] == v)
                elif t == "in_list": mask |= (df_res[col].isin(v))
                elif t == "regex": mask |= df_res[col].astype(str).str.contains(v, na=False, regex=True)
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
    df_test = aplicar_limpieza(df_test, df_train)
    df_dev = aplicar_limpieza(df_dev, df_train)
    print(" -> Valores erróneos tratados.")
    return df_train, df_test, df_dev


# ==========================================
# 5. CODIFICAR OBJETIVO (Nuevo - LabelEncoder para Target)
# ==========================================
def codificar_objetivo(df_train, df_test, df_dev, config):
    target = config.get("target")
    if target and target in df_train.columns:
        if not pd.api.types.is_numeric_dtype(df_train[target]):
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            # Aplicamos a test y dev. Si hay etiquetas nuevas, ponemos -1
            df_test[target] = df_test[target].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            df_dev[target] = df_dev[target].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            print(f" -> Variable objetivo '{target}' codificada a números.")
    return df_train, df_test, df_dev


# # ==========================================
# # 6. TRADUCCIÓN DE EMOJIS A TEXTO
# # ==========================================
# def traducir_emojis(df_train, df_test, df_dev, config, target):
#     text_cols = config.get('text_features', [])
#     if not text_cols: return df_train, df_test, df_dev
#
#     def limpiar_texto(texto):
#         if not isinstance(texto, str): return ""
#         return emoji.demojize(texto, delimiters=(" ", " "))
#
#     for col in text_cols:
#         if col in df_train.columns and col != target:
#             print(f" ✨ Traduciendo emojis a texto en: {col}")
#             df_train[col] = df_train[col].apply(limpiar_texto)
#             df_test[col] = df_test[col].apply(limpiar_texto)
#             df_dev[col] = df_dev[col].apply(limpiar_texto)
#     return df_train, df_test, df_dev


# ==========================================
# 7. LIMPIEZA DE TEXTO AVANZADA (Nuevo - Stemming y Regex)
# ==========================================
def limpiar_y_normalizar_texto(df_train, df_test, df_dev, config, target):
    text_cols = config.get('text_features', [])
    if not text_cols: return df_train, df_test, df_dev

    lang = config.get('language', 'english')
    try:
        stop_words = set(stopwords.words(lang))
        stemmer = SnowballStemmer(lang[:7] if lang == 'spanish' else 'english')
    except:
        stop_words = set()
        stemmer = None

    def procesar_celda(texto):
        if not isinstance(texto, str): return ""
        # Minúsculas y quitar puntuación
        texto = str(texto).lower().strip()
        texto = re.sub(r'[^\w\s]', '', texto)

        # Stemming y Stopwords
        if stemmer:
            palabras = texto.split()
            palabras_limpias = [stemmer.stem(p) for p in palabras if p not in stop_words]
            texto = " ".join(palabras_limpias)
        return texto

    for col in text_cols:
        if col in df_train.columns and col != target:
            print(f" 🧹 Normalizando y aplicando Stemming en: {col}")
            df_train[col] = df_train[col].apply(procesar_celda)
            df_test[col] = df_test[col].apply(procesar_celda)
            df_dev[col] = df_dev[col].apply(procesar_celda)

    return df_train, df_test, df_dev


# ==========================================
# 8. VECTORIZACIÓN DE TEXTO (TF-IDF / BoW)
# ==========================================
def procesar_texto(df_train, df_test, df_dev, config, target):
    text_cols = config.get('text_features', [])
    if not text_cols: return df_train, df_test, df_dev

    method = config.get('text_process_method', 'tf-idf')

    if method == 'tf-idf':
        vec = TfidfVectorizer()
    elif method == 'bow':
        vec = CountVectorizer()
    else:
        vec = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    for col in text_cols:
        if col not in df_train.columns or col == target: continue

        if method == 'one-hot':
            t_train = vec.fit_transform(df_train[[col]])
            t_test = vec.transform(df_test[[col]])
            t_dev = vec.transform(df_dev[[col]])
        else:
            t_train = vec.fit_transform(df_train[col].astype(str))
            t_test = vec.transform(df_test[col].astype(str))
            t_dev = vec.transform(df_dev[col].astype(str))

        if hasattr(t_train, "toarray"):
            t_train, t_test, t_dev = t_train.toarray(), t_test.toarray(), t_dev.toarray()

        cols_names = [f"{col}_{i}" for i in range(t_train.shape[1])]
        df_t_train = pd.DataFrame(t_train, columns=cols_names, index=df_train.index)
        df_t_test = pd.DataFrame(t_test, columns=cols_names, index=df_test.index)
        df_t_dev = pd.DataFrame(t_dev, columns=cols_names, index=df_dev.index)

        df_train = pd.concat([df_train.drop(columns=[col]), df_t_train], axis=1)
        df_test = pd.concat([df_test.drop(columns=[col]), df_t_test], axis=1)
        df_dev = pd.concat([df_dev.drop(columns=[col]), df_t_dev], axis=1)

    print(f" -> Texto vectorizado con {method}.")
    return df_train, df_test, df_dev


# ==========================================
# 9. CODIFICACIÓN CATEGÓRICA Y BOOLEANA
# ==========================================
def codificar_variables(df_train, df_test, df_dev, config, target):
    estrategia = config.get("categorical_encoding", "none")
    # Buscamos columnas de tipo 'category' u 'object'
    categorical_cols = df_train.select_dtypes(include=['category', 'object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target]

    if estrategia == "none" or not categorical_cols: return df_train, df_test, df_dev

    if estrategia == "one-hot":
        df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True, dtype=int)
        df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True, dtype=int)
        df_dev = pd.get_dummies(df_dev, columns=categorical_cols, drop_first=True, dtype=int)
        # Alineación para igualar columnas
        df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)
        df_train, df_dev = df_train.align(df_dev, join='left', axis=1, fill_value=0)

    elif estrategia == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            df_test[col] = df_test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            df_dev[col] = df_dev[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    print(f" -> Variables categóricas codificadas usando: {estrategia}.")
    return df_train, df_test, df_dev


# ==========================================
# 10. IMPUTACIÓN DE NULOS
# ==========================================
def tratar_nulos(df_train, df_test, df_dev, config, target):
    accion = config.get('missing_values')
    if accion == 'impute':
        strategy = config.get('impute_strategy', 'mean')
        imputer = SimpleImputer(strategy=strategy)

        df_train = df_train.dropna(subset=[target])
        df_test = df_test.dropna(subset=[target])
        df_dev = df_dev.dropna(subset=[target])

        cols_input = [c for c in df_train.columns if c != target]
        df_train[cols_input] = imputer.fit_transform(df_train[cols_input])
        df_test[cols_input] = imputer.transform(df_test[cols_input])
        df_dev[cols_input] = imputer.transform(df_dev[cols_input])
        print(f" -> Nulos imputados en características (Estrategia: {strategy}).")

    elif accion == 'delete':
        df_train, df_test, df_dev = df_train.dropna(), df_test.dropna(), df_dev.dropna()
        print(" -> Filas con nulos eliminadas.")

    return df_train, df_test, df_dev


# ==========================================
# 11. OUTLIERS (Clipping con IQR)
# ==========================================
def tratar_outliers(df_train, df_test, df_dev, config, target):
    if config.get('outliers') == 'clip':
        num_cols = df_train.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col == target: continue

            Q1 = df_train[col].quantile(0.25)
            Q3 = df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            inf, sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            df_train[col] = np.clip(df_train[col], inf, sup)
            df_test[col] = np.clip(df_test[col], inf, sup)
            df_dev[col] = np.clip(df_dev[col], inf, sup)

        print(" -> Outliers tratados (Clipping IQR).")
    return df_train, df_test, df_dev


# ==========================================
# 12. DISCRETIZACIÓN Y ESCALADO
# ==========================================
def escalar_y_discretizar(df_train, df_test, df_dev, config, target):
    disc_cols = config.get('discretize_features', [])
    if disc_cols:
        n = config.get('discretize_bins', 3)
        est = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='uniform')
        for col in disc_cols:
            if col in df_train.columns and col != target:
                df_train[col] = est.fit_transform(df_train[[col]])
                df_test[col] = est.transform(df_test[[col]])
                df_dev[col] = est.transform(df_dev[[col]])

    method = config.get('scaling')
    if method in ['max-min', 'z-score']:
        scaler = MinMaxScaler() if method == 'max-min' else StandardScaler()
        cols_a_escalar = [c for c in df_train.columns if c != target]

        df_train[cols_a_escalar] = scaler.fit_transform(df_train[cols_a_escalar])
        df_test[cols_a_escalar] = scaler.transform(df_test[cols_a_escalar])
        df_dev[cols_a_escalar] = scaler.transform(df_dev[cols_a_escalar])
        print(f" -> Datos escalados con {method}.")

    return df_train, df_test, df_dev


# ==========================================
# 13. BALANCEO DE DATOS (Solo en Train)
# ==========================================
def balancear_clases(df_train, config, target):
    strat = config.get("sampling_strategy", "none")
    if strat == "none" or not target or target not in df_train.columns: return df_train

    X = df_train.drop(columns=[target])
    y = df_train[target]

    if strat == "oversample":
        res = RandomOverSampler(random_state=42)
    elif strat == "undersample":
        res = RandomUnderSampler(random_state=42)
    elif strat == "SMOTE":
        res = SMOTE(random_state=42)
    else:
        return df_train

    X_res, y_res = res.fit_resample(X, y)
    df_train = pd.concat([X_res, y_res], axis=1)
    print(f" -> Balanceo aplicado en Train ({strat}).")
    return df_train


# ==========================================
# PIPELINE PRINCIPAL UNIFICADO
# ==========================================
def pipeline_preprocesamiento(df_train, df_test, df_dev, config_full):
    target_global = config_full.get("target")
    config_prep = config_full.get("preprocessing", {})

    # Activamos la señal de interrupción para poder pararlo con Ctrl+C sin romper nada
    signal.signal(signal.SIGINT, signal_handler)

    print("\n--- 🛠️ INICIANDO PIPELINE DE PREPROCESADO ---")

    # 1. Limpieza básica y Tipos
    df_train, df_test, df_dev = eliminar_duplicados(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = eliminar_columnas(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = asignar_tipos(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = tratar_valores_erroneos(df_train, df_test, df_dev, config_prep)

    # 2. Codificación del Target (Crucial para Sentiment Analysis)
    df_train, df_test, df_dev = codificar_objetivo(df_train, df_test, df_dev, config_full)

    # 3. Procesamiento de Texto Completo
    # df_train, df_test, df_dev = traducir_emojis(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = limpiar_y_normalizar_texto(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = procesar_texto(df_train, df_test, df_dev, config_prep, target_global)

    # 4. Variables Categóricas
    df_train, df_test, df_dev = codificar_variables(df_train, df_test, df_dev, config_prep, target_global)

    # 5. Tratamiento estadístico
    df_train, df_test, df_dev = tratar_nulos(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = tratar_outliers(df_train, df_test, df_dev, config_prep, target_global)

    # 6. Ajustes finales
    df_train, df_test, df_dev = escalar_y_discretizar(df_train, df_test, df_dev, config_prep, target_global)
    df_train = balancear_clases(df_train, config_prep, target_global)

    print("--- ✅ PIPELINE DE PREPROCESADO COMPLETADO ---\n")
    return df_train, df_test, df_dev