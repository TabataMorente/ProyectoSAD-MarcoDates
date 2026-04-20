# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
import emoji

# Herramientas de Scikit-Learn e Imblearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords

# ==========================================
# CONFIGURACIÓN INICIAL DE NLTK
# ==========================================
try:
    nltk.data.find('corpora/stopwords')
except (LookupError, AttributeError):
    print("📥 Descargando diccionario de stopwords...")
    nltk.download('stopwords', quiet=True)


# ==========================================
# 1. ELIMINAR DUPLICADOS
# ==========================================
def eliminar_duplicados(df_train, df_test, df_dev, config):
    if config.get("drop_duplicates"):
        df_train = df_train.drop_duplicates()
        # Nota: Normalmente solo se eliminan duplicados en Train para no perder filas de evaluación,
        # pero seguimos la instrucción de limpiar ambos si el config lo pide.
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
# 3. VALORES ERRÓNEOS (Lógica de Negocio)
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

            # Detectar errores según condiciones del JSON
            for cond in condiciones:
                t, v = cond.get("type"), cond.get("value")
                if t == "less_than":
                    mask |= (df_res[col] < v)
                elif t == "greater_than":
                    mask |= (df_res[col] > v)
                elif t == "equals":
                    mask |= (df_res[col] == v)
                elif t == "regex":
                    mask |= df_res[col].astype(str).str.contains(v, na=False)

            if mask.any():
                if accion_global == "delete":
                    df_res = df_res[~mask]
                elif accion_global == "impute":
                    # Imputación manual basada en estadísticas de la referencia (Train)
                    if estrategia == "mean":
                        val = df_ref[col].mean()
                    elif estrategia == "median":
                        val = df_ref[col].median()
                    elif estrategia == "mode":
                        val = df_ref[col].mode()[0]
                    df_res.loc[mask, col] = val
        return df_res

    df_train = aplicar_limpieza(df_train, df_train)
    df_test = aplicar_limpieza(df_test, df_train)
    df_dev = aplicar_limpieza(df_dev, df_train)
    print(" -> Valores erróneos tratados.")
    return df_train, df_test, df_dev


# ==========================================
# 3.5. TRADUCCIÓN DE EMOJIS A TEXTO
# ==========================================
def traducir_emojis(df_train, df_test, df_dev, config, target):
    """
    Convierte emojis en texto descriptivo para que los vectorizadores puedan procesarlos.
    Requiere: pip install emoji
    """
    # Verificamos si la opción está activa y si hay columnas de texto
    text_cols = config.get('text_features', [])

    if not text_cols:
        return df_train, df_test, df_dev

    def limpiar_texto(texto):
        if not isinstance(texto, str):
            return ""
        # Reemplaza emojis por su nombre en texto usando espacios en lugar de dos puntos
        return emoji.demojize(texto, delimiters=(" ", " "))

    for col in text_cols:
        if col in df_train.columns and col != target:
            print(f" ✨ Traduciendo emojis a texto en: {col}")
            df_train[col] = df_train[col].apply(limpiar_texto)
            df_test[col] = df_test[col].apply(limpiar_texto)
            df_dev[col] = df_dev[col].apply(limpiar_texto)

    return df_train, df_test, df_dev


# ==========================================
# 4. PREPROCESADO DE TEXTO (NLP)
# ==========================================
# Añadimos 'target' como argumento para protegerlo
def procesar_texto(df_train, df_test, df_dev, config, target):
    text_cols = config.get('text_features', [])
    if not text_cols: return df_train, df_test, df_dev

    method = config.get('text_process_method', 'tf-idf')
    lang = config.get('language', 'english')

    # Intentamos cargar stopwords, si falla (por el examen) usamos None
    try:
        sw = stopwords.words(lang)
    except:
        sw = None

    # 1. Elegimos vectorizador según el metodo del JSON
    if method == 'tf-idf':
        vec = TfidfVectorizer(stop_words=sw)
    elif method == 'bow':
        vec = CountVectorizer(stop_words=sw)
    else:
        # handle_unknown='ignore' es vital para que el test no explote con palabras nuevas
        vec = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    for col in text_cols:
        # PROTECCIÓN: Si la columna no existe o es el target, saltamos
        if col not in df_train.columns or col == target:
            continue

        # 2. ENTRENAMIENTO (Fit) solo en Train y APLICACIÓN (Transform) en todos
        if method == 'one-hot':
            # One-Hot necesita DataFrame 2D -> [[col]]
            t_train = vec.fit_transform(df_train[[col]])
            t_test = vec.transform(df_test[[col]])
            t_dev = vec.transform(df_dev[[col]])
        else:
            # TF-IDF/BoW necesitan Series de texto -> [col]
            t_train = vec.fit_transform(df_train[col].astype(str))
            t_test = vec.transform(df_test[col].astype(str))
            t_dev = vec.transform(df_dev[col].astype(str))

        # 3. CONVERSIÓN A MATRIZ DENSA (Scikit-Learn suele devolver matrices sparse)
        if hasattr(t_train, "toarray"):
            t_train = t_train.toarray()
            t_test = t_test.toarray()
            t_dev = t_dev.toarray()

        # 4. CREACIÓN DE NUEVOS DATAFRAMES
        # Usamos el index original para que al concatenar no se desalineen las filas
        cols_names = [f"{col}_{i}" for i in range(t_train.shape[1])]
        df_t_train = pd.DataFrame(t_train, columns=cols_names, index=df_train.index)
        df_t_test = pd.DataFrame(t_test, columns=cols_names, index=df_test.index)
        df_t_dev = pd.DataFrame(t_dev, columns=cols_names, index=df_dev.index)

        # 5. REEMPLAZO DE COLUMNAS
        # Borramos la de texto original y pegamos las nuevas columnas numéricas
        df_train = pd.concat([df_train.drop(columns=[col]), df_t_train], axis=1)
        df_test = pd.concat([df_test.drop(columns=[col]), df_t_test], axis=1)
        df_dev = pd.concat([df_dev.drop(columns=[col]), df_t_dev], axis=1)

    print(f" -> Texto procesado con {method} en las columnas: {text_cols}.")
    return df_train, df_test, df_dev


# ==========================================
# 5. CODIFICACIÓN CATEGÓRICA Y BOOLEANA
# ==========================================
def codificar_variables(df_train, df_test, df_dev, config, target):
    # Categorías Ordinales (Mapeo numérico)
    cat_cols = config.get('categorical_features', [])
    for col in cat_cols:
        if col in df_train.columns and col != target:  # Protección target
            mapping = {val: i for i, val in enumerate(df_train[col].unique())}
            df_train[col] = df_train[col].map(mapping)
            # Si en test aparece algo que no estaba en train, le ponemos -1
            df_test[col] = df_test[col].map(mapping).fillna(-1)
            df_dev[col] = df_dev[col].map(mapping).fillna(-1)

    # Booleanos (Normalización de texto a 0/1)
    bool_cols = config.get('boolean_features', [])
    mapeo_bool = {'true': 1, 'false': 0, 'sí': 1, 'no': 0, 'yes': 1, '1': 1, '0': 0}
    for col in bool_cols:
        if col in df_train.columns and col != target:
            for df in [df_train, df_test, df_dev]:
                df[col] = df[col].astype(str).str.lower().map(mapeo_bool).fillna(0)

    print(" -> Variables categóricas/booleanas codificadas.")
    return df_train, df_test, df_dev


# ==========================================
# 6. IMPUTACIÓN DE NULOS
# ==========================================
def tratar_nulos(df_train, df_test, df_dev, config, target):
    accion = config.get('missing_values')

    if accion == 'impute':
        strategy = config.get('impute_strategy', 'mean')
        imputer = SimpleImputer(strategy=strategy)

        # SEPARAMOS EL TARGET: No queremos imputar la "respuesta"
        # Si faltan etiquetas en el target, lo mejor es borrar esas filas
        df_train = df_train.dropna(subset=[target])
        df_test = df_test.dropna(subset=[target])
        df_dev = df_dev.dropna(subset=[target])

        # Imputamos solo las características (X)
        cols_input = [c for c in df_train.columns if c != target]

        df_train[cols_input] = imputer.fit_transform(df_train[cols_input])
        df_test[cols_input] = imputer.transform(df_test[cols_input])
        df_dev[cols_input] = imputer.transform(df_dev[cols_input])

        print(f" -> Nulos imputados en características (Estrategia: {strategy}).")

    elif accion == 'delete':
        df_train = df_train.dropna()
        df_test = df_test.dropna()
        df_dev = df_dev.dropna()
        print(" -> Filas con nulos eliminadas.")

    return df_train, df_test, df_dev


# ==========================================
# 7. OUTLIERS (Clipping con IQR)
# ==========================================
def tratar_outliers(df_train, df_test, df_dev, config, target):
    if config.get('outliers') == 'clip':
        num_cols = df_train.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            if col == target: continue  # ESCUDO: No recortamos la respuesta

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
# 8. DISCRETIZACIÓN Y ESCALADO
# ==========================================
def escalar_y_discretizar(df_train, df_test, df_dev, config, target):
    # 1. Discretización
    disc_cols = config.get('discretize_features', [])
    if disc_cols:
        n = config.get('discretize_bins', 3)
        est = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='uniform')
        for col in disc_cols:
            if col in df_train.columns and col != target:
                df_train[col] = est.fit_transform(df_train[[col]])
                df_test[col] = est.transform(df_test[[col]])
                df_dev[col] = est.transform(df_dev[[col]])

    # 2. Escalado Final
    method = config.get('scaling')
    if method in ['max-min', 'z-score']:
        scaler = MinMaxScaler() if method == 'max-min' else StandardScaler()

        # ESCALAMOS SOLO X (las preguntas), NO y (la respuesta)
        cols_a_escalar = [c for c in df_train.columns if c != target]

        df_train[cols_a_escalar] = scaler.fit_transform(df_train[cols_a_escalar])
        df_test[cols_a_escalar] = scaler.transform(df_test[cols_a_escalar])
        df_dev[cols_a_escalar] = scaler.transform(df_dev[cols_a_escalar])

        print(f" -> Datos escalados con {method} (Target protegido).")

    return df_train, df_test, df_dev


# ==========================================
# 9. BALANCEO DE DATOS (Solo en Train)
# ==========================================
def balancear_clases(df_train, config, target):
    # Aquí 'target' ya viene de fuera, no del config
    strat = config.get("sampling_strategy", "none")

    if strat == "none" or not target or target not in df_train.columns:
        return df_train

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
# PIPELINE PRINCIPAL
# ==========================================
def pipeline_preprocesamiento(df_train, df_test, df_dev, config_full):
    # 1. Extraer info del JSON
    target_global = config_full.get("target")
    config_prep = config_full.get("preprocessing", {})

    # 2. Limpieza básica (No suelen necesitar el target)
    df_train, df_test, df_dev = eliminar_duplicados(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = eliminar_columnas(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = tratar_valores_erroneos(df_train, df_test, df_dev, config_prep)

    # 3. Transformación (El texto y las variables necesitan saber cuál es el target para no tocarlo)
    df_train, df_test, df_dev = traducir_emojis(df_train, df_test, df_dev, config_prep)
    df_train, df_test, df_dev = procesar_texto(df_train, df_test, df_dev, config_prep, target_global)
    df_train, df_test, df_dev = codificar_variables(df_train, df_test, df_dev, config_prep, target_global)

    # 4. Tratamiento estadístico (Aquí el target es el "escudo")
    df_train, df_test, df_dev = tratar_nulos(df_train, df_test, df_dev, config_prep,target_global)
    df_train, df_test, df_dev = tratar_outliers(df_train, df_test, df_dev, config_prep, target_global)

    # 5. Ajustes finales
    df_train, df_test, df_dev = escalar_y_discretizar(df_train, df_test, df_dev, config_prep, target_global)
    df_train = balancear_clases(df_train, config_prep, target_global)

    return df_train, df_test, df_dev