# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk

# Herramientas de Scikit-Learn para limpiar y preprocesar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import KBinsDiscretizer # Para discretizar
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except (LookupError, AttributeError):
    print("📥 Descargando diccionario de stopwords...")
    nltk.download('stopwords')

try:
    sw_list = stopwords.words('spanish') + stopwords.words('english')
except:
    sw_list = None  # Backup por si falla la descarga en el examen

def apply_preprocessing(X_train, X_dev, X_test, config):
    """
    FLUJO DE PREPROCESADO COMPLETO:
    1. Texto -> 2. Categorías -> 3. Booleanos -> 4. Imputación -> 5. Outliers -> 6. Escalado
    """
    prep_cfg = config['preprocessing'] # Leer la configuración de preprocesado del JSON

    # IMPORTANTE: Convertimos a DataFrame para poder usar nombres de columnas y tipos
    # Usamos los nombres originales de las columnas del CSV

    train_df = pd.DataFrame(X_train).reset_index(drop=True)
    dev_df = pd.DataFrame(X_dev).reset_index(drop=True)
    test_df = pd.DataFrame(X_test).reset_index(drop=True)

    # --- 1. PREPROCESADO DE TEXTO (TF-IDF / BoW / One-Hot) ---
    text_cols = prep_cfg.get('text_features', [])  # Busca si hay columnas de texto definidas en el JSON
    if text_cols:
        method = prep_cfg.get('text_process', 'tf-idf')  # Obtiene el metodo de procesado

        # Leemos el idioma del JSON (por defecto 'spanish' o 'english')
        lang = prep_cfg.get('language', 'spanish')
        try:
            sw_list = stopwords.words(lang)
        except:
            sw_list = None  # Si el idioma no existe en NLTK, no filtramos nada

        # Elegimos la técnica de vectorización según indique el JSON
        if method == 'tf-idf':
            vec = TfidfVectorizer()
        elif method == 'bow':
            vec = CountVectorizer()
        else:  # Si es OneHot encoding
            vec = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Texto como categoria

        for col in text_cols:
            # 2. TRANSFORMACIÓN: Aquí está el truco de las dimensiones
            if method == 'one-hot':
                # One-Hot NECESITA 2D (DataFrame) -> [[col]]
                t_train = vec.fit_transform(train_df[[col]])
                t_dev = vec.transform(dev_df[[col]])
                t_test = vec.transform(test_df[[col]])
            else:
                # TF-IDF/BoW NECESITAN 1D (Series/Texto) -> [col]
                t_train = vec.fit_transform(train_df[col].astype(str))
                t_dev = vec.transform(dev_df[col].astype(str))
                t_test = vec.transform(test_df[col].astype(str))

            # Si el resultado es una matriz dispersa (sparse), la convertimos a densa (array)
            if hasattr(t_train, "toarray"):
                t_train = t_train.toarray()
                t_dev = t_dev.toarray()
                t_test = t_test.toarray()

            # Definimos nombres de columnas (ej: message_0, message_1...) para que Pandas no se pierda
            col_names = [f"{col}_{i}" for i in range(t_train.shape[1])]

            # Convertimos a DataFrame especificando el tipo de dato (float) desde el principio
            t_train_df = pd.DataFrame(t_train, columns=col_names, dtype=float).reset_index(drop=True)
            t_dev_df = pd.DataFrame(t_dev, columns=col_names, dtype=float).reset_index(drop=True)
            t_test_df = pd.DataFrame(t_test, columns=col_names, dtype=float).reset_index(drop=True)

            # Borramos la columna de texto original y concatenamos las nuevas columnas numéricas
            train_df = pd.concat([train_df.drop(columns=[col]).reset_index(drop=True), t_train_df], axis=1)
            dev_df = pd.concat([dev_df.drop(columns=[col]).reset_index(drop=True), t_dev_df], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]).reset_index(drop=True), t_test_df], axis=1)

    # --- 2. CATEGORIALES (Reemplazo por número/Ordinal) ---
    cat_cols = prep_cfg.get('categorical_features', [])
    for col in cat_cols:
        # Crea un mapa: cada categoría única recibe un número (0, 1, 2...)
        categorias = train_df[col].unique() # Obtiene los valores únicos (ej: 'rojo', 'azul')
        mapeo_cat = {val: i for i, val in enumerate(categorias)} # Crea mapa {'rojo': 0, 'azul': 1}
        # Transforma las palabras en números usando el mapa anterior
        train_df[col] = train_df[col].map(mapeo_cat)
        dev_df[col] = dev_df[col].map(mapeo_cat)
        test_df[col] = test_df[col].map(mapeo_cat)

    # --- 3. BOOLEANOS (Conversión de Texto a 0/1) ---
    # Solo se procesan los que vienen en el JSON (formato texto)
    bool_cols = prep_cfg.get('boolean_features', [])
    # Diccionario de traducción para normalizar diferentes formas de escribir booleanos
    mapeo_bool = {'true': 1, 'false': 0, 'sí': 1, 'no': 0, 'yes': 1, 'si': 1, '1': 1, '0': 0}
    for col in bool_cols:
        # Convierte a minúsculas, traduce según el mapa y guarda como número
        train_df[col] = train_df[col].astype(str).str.lower().map(mapeo_bool)
        dev_df[col] = dev_df[col].astype(str).str.lower().map(mapeo_bool)
        test_df[col] = test_df[col].astype(str).str.lower().map(mapeo_bool)

    # --- 4. GESTIÓN DE MISSING VALUES ---
    # Ahora que todas es número, imputamos
    if prep_cfg.get('missing_values') == 'impute':
        strategy = prep_cfg.get('impute_strategy', 'mean') # 'mean' rellenará con la media
        imputer = SimpleImputer(strategy=strategy) # Configura el imputador
        cols_nombres = train_df.columns # Guarda los nombres de las columnas
        # Rellenamos huecos (NaN) usando la estrategia (media, moda...) calculada en TRAIN
        # fit aprende las medias de TRAIN; transform las aplica para rellenar huecos en todos
        train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=cols_nombres)
        dev_df = pd.DataFrame(imputer.transform(dev_df), columns=cols_nombres)
        test_df = pd.DataFrame(imputer.transform(test_df), columns=cols_nombres)

    # --- 5. GESTIÓN DE OUTLIERS (IQR Clipping) ---
    # Solo actúa en columnas numéricas
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        # Calculamos los límites estadísticos (Cuartiles)
        Q1 = train_df[col].quantile(0.25) # Primer cuartil (percentil 25)
        Q3 = train_df[col].quantile(0.75) # Tercer cuartil (percentil 75)
        IQR = Q3 - Q1 # Rango intercuartílico (la "anchura" de la caja)
        lower_limit = Q1 - 1.5 * IQR # Límite inferior
        upper_limit = Q3 + 1.5 * IQR # Límite superior

        # Recortamos los valores en los 3 conjuntos usando los límites de TRAIN
        train_df[col] = np.clip(train_df[col], lower_limit, upper_limit)
        dev_df[col] = np.clip(dev_df[col], lower_limit, upper_limit)
        test_df[col] = np.clip(test_df[col], lower_limit, upper_limit)

    # --- 5.5 DISCRETIZACIÓN (Opcional) ---
    # Convertimos números continuos en "cajones" o categorías (bins)

    disc_cols = prep_cfg.get('discretize_features', [])
    if disc_cols:
        n_bins = prep_cfg.get('discretize_bins', 3)
        # encode='ordinal' para que devuelva 0, 1, 2...
        # strategy='uniform' para que los rangos tengan el mismo ancho
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)

        for col in disc_cols:
            if col in train_df.columns:
                # 1. Ajustamos con los datos de TRAIN
                train_df[col] = discretizer.fit_transform(train_df[[col]])
                # 2. Aplicamos la misma frontera a DEV y TEST
                dev_df[col] = discretizer.transform(dev_df[[col]])
                test_df[col] = discretizer.transform(test_df[[col]])
                print(f"INFO: Columna '{col}' discretizada en {n_bins} intervalos.")

    # --- 6. ESCALADO FINAL  ---
    # Normalizamos los rangos de los números para que el KNN funcione bien
    if prep_cfg.get('scaling') == 'max-min':
        scaler = MinMaxScaler() # Escala al rango entre 0 y 1
    elif prep_cfg.get('scaling') == 'z-score':
        scaler = StandardScaler() # Centra los datos (media 0, desviación 1)
    elif prep_cfg.get('scaling') == 'max':
        scaler = StandardScaler() # Centra los datos (media 0, desviación 1)
    else:
        scaler = None

    # Ajustamos el escalador con TRAIN y transformamos los tres conjuntos
    if scaler is not None:
        # Si hay un escalador configurado, escalamos los datos
        X_train_final = scaler.fit_transform(train_df)
        X_dev_final = scaler.transform(dev_df)
        X_test_final = scaler.transform(test_df)
    else:
        # Si no queremos escalar (scaler es None), dejamos los datos igual
        X_train_final = train_df.copy()
        X_dev_final = dev_df.copy()
        X_test_final = test_df.copy()

    return X_train_final, X_dev_final, X_test_final