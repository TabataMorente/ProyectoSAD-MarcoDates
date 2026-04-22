# -*- coding: utf-8 -*-
import sys
import os
import json
import joblib
import pandas as pd

# --- IMPORTACIÓN DEL NUEVO PREPROCESADO ---
from preprocessing import pipeline_preprocesamiento

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Función para abrir y leer el archivo de configuración JSON
def load_config(json_path):
    with open(json_path, 'r') as f: # Abre el archivo en modo lectura
        return json.load(f) # Convierte el contenido del JSON en un diccionario de Python

def train():
    # 1. Validación (Ahora solo pedimos un CSV y el JSON)
    if len(sys.argv) < 3:
        print("Uso: python train.py <data.csv> <config_file.json>")
        sys.exit(1)

    # 2. Carga de configuración y datos originales
    config = load_config(sys.argv[2])
    target_global = config.get('target')
    df_raw = pd.read_csv(sys.argv[1])

    # 3. DIVISIÓN TRIPLE (70/15/15)
    # Paso A: Separar el 70% para Train y el 30% para lo demás (Dev + Test)
    df_train_raw, df_temp = train_test_split(
        df_raw,
        test_size=0.30,
        random_state=42
    )

    # Paso B: Del 30% restante, dividimos a la mitad para sacar Dev (15%) y Test (15%)
    df_dev_raw, df_test_raw = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=42
    )

    print(f"📊 Reparto: Train={len(df_train_raw)} | Dev={len(df_dev_raw)} | Test={len(df_test_raw)}")

    # 4. APLICACIÓN DEL PIPELINE MODULAR
    # Ahora enviamos los 3 trozos recién creados a tu preprocessing.py
    df_train_p, df_test_p, df_dev_p = pipeline_preprocesamiento(
        df_train_raw.copy(),
        df_test_raw.copy(),
        df_dev_raw.copy(),
        config
    )

    # 5. SEPARACIÓN FINAL X e y (Usando tus nombres de variables)
    X_train_p = df_train_p.drop(columns=[target_global])
    y_train = df_train_p[target_global]

    X_dev_p = df_dev_p.drop(columns=[target_global])
    y_dev = df_dev_p[target_global]

    X_test_p = df_test_p.drop(columns=[target_global])
    y_test = df_test_p[target_global]

    # --- INICIO DEL ENTRENAMIENTO ---
    method = config.get('method', 'knn')
    task = config.get('task', 'classification')
    csv_id = os.path.basename(sys.argv[1]).split('.')[0]
    eval_strat = config.get('evaluation', 'macro')  # Estrategia de evaluación (macro/micro) del JSON
    #lista para guardar las predicciones de los modelos
    dict_predicciones = {'Valor_Real': y_dev.values}
    #carpeta para los csv
    folder_path = os.path.join("csv", csv_id)
    os.makedirs(folder_path, exist_ok=True)

    print(f"🚀 Iniciando entrenamiento con método: {method}...")

    # --- CASO NAIVE BAYES (MULTIMODELO) ---
    if method == 'bayes':
        params_cfg = config.get('hyperparameters_bayes', {})
        b_type = params_cfg.get('bayes_type', 'gaussian')

        # Carpeta común para todos los Bayes
        folder_path = os.path.join("modelos", csv_id, method)
        os.makedirs(folder_path, exist_ok=True)

        # 1. Elegimos el "sabor" de Bayes
        if b_type == 'multinomial':
            # Tipo de algoritmo: 'multinomial' es ideal cuando discretizamos o usamos frecuencias
            for a in params_cfg.get('alpha', [1.0]):
                # Parámetro Alpha: Es el 'Suavizado de Laplace'. Suma un pequeño valor para que ninguna probabilidad sea 0%.
                model = MultinomialNB(alpha=a)
                model.fit(X_train_p, y_train)
                y_pred = model.predict(X_dev_p)  # Generamos predicción para comparar
                score_dev = f1_score(y_dev, y_pred, average=eval_strat)

                model_name = f"bayes_multi_alpha={a}.sav"
                dict_predicciones[model_name] = y_pred

                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

        elif b_type == 'bernoulli':
            for a in params_cfg.get('alpha', [1.0]):
                model = BernoulliNB(alpha=a)
                model.fit(X_train_p, y_train)
                y_pred = model.predict(X_dev_p)
                score_dev = f1_score(y_dev, y_pred, average=eval_strat)

                model_name = f"bayes_bern_alpha={a}.sav"
                dict_predicciones[model_name] = y_pred

                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

        else:  # GAUSSIAN
            for sm in params_cfg.get('var_smoothing', [1e-9]):
                # Suavizado de varianza: Ayuda al modelo GaussianNB a no ser tan rígido (evita divisiones por cero)
                model = GaussianNB(var_smoothing=sm)
                model.fit(X_train_p, y_train)
                y_pred = model.predict(X_dev_p)
                score_dev = f1_score(y_dev, y_pred, average=eval_strat)

                model_name = f"bayes_gauss_sm={sm}.sav"
                dict_predicciones[model_name] = y_pred

                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

    # --- CASO KNN ---
    elif method == 'knn':
        # Sacamos las listas del JSON. Si no existen, ponemos unas por defecto []
        params_cfg = config.get('hyperparameters_knn', {})

        # .get(clave, valor_por_defecto)
        k_min, k_max = params_cfg.get('k_range', [1, 5])  # IMPORTANTE: Aquí se pone un limite maximo y minimo
        lista_p = params_cfg.get('p', [1, 2])
        lista_w = params_cfg.get('weights', ["uniform", "distance"])
        step = params_cfg.get('step', 2)  # Va de 2 en 2, o el numero que sea

        # GENERAMOS LA LISTA DINÁMICA
        lista_k = list(range(k_min, k_max + 1, step))

        # BARRIDO DE HIPERPARÁMETROS: Probamos combinaciones de k, p y pesos
        for k in lista_k:  # k: número de vecinos a consultar
            for p in lista_p:  # p=1 es distancia Manhattan, p=2 es distancia Euclídea
                for w in lista_w:  # w: peso de la distancia (uniforme o ponderado)

                    # 1. ELEGIMOS EL ALGORITMO SEGÚN LA TAREA
                    if task == 'regression':
                        model = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
                        model.fit(X_train_p, y_train)
                        y_pred = model.predict(X_dev_p)
                        score_dev = r2_score(y_dev, y_pred)
                        metric = "R2"
                    else:
                        model = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
                        model.fit(X_train_p, y_train)
                        y_pred = model.predict(X_dev_p)
                        score_dev = f1_score(y_dev, y_pred, average=eval_strat)
                        metric = "F1"

                    # GUARDAMOS TODOS LOS MODELOS GENERADOS
                    folder_path = os.path.join("modelos", csv_id, method)
                    os.makedirs(folder_path, exist_ok=True)

                    params_str = f"k={k}_p={p}_w={w}"
                    model_name = f"knn_{params_str}.sav"

                    # --- GUARDADO DE RESULTADOS CSV ---
                    dict_predicciones[model_name] = y_pred

                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    # --- CASO ARBOLES DE DECISION ---
    elif method == 'tree':
        # 1. Extraemos los hiperparámetros del JSON
        params_cfg = config.get('hyperparameters_tree', {})
        lista_depth = params_cfg.get('max_depth', [None, 5, 10])  ## valores de profundidad por defecto
        lista_crit = params_cfg.get('criterio', ['gini', 'entropy'])

        # 2. Barrido de hiperparámetros
        for depth in lista_depth:
            for crit in lista_crit:

                # 3. Ajuste según la tarea (Clasificación o Regresión)
                if task == 'regression':
                    # En regresión, el criterio suele ser 'squared_error' o 'absolute_error'
                    c_reg = 'squared_error' if crit == 'gini' else 'absolute_error'
                    model = DecisionTreeRegressor(max_depth=depth, criterion=c_reg,
                                                  random_state=42)  # crea el modelo
                    model.fit(X_train_p, y_train)  # estudia los datos X y aprende a llegar al target Y
                    y_pred = model.predict(X_dev_p)
                    score_dev = r2_score(y_dev, y_pred)  # calcula R2 score
                    metric = "R2"
                else:
                    model = DecisionTreeClassifier(max_depth=depth, criterion=crit, random_state=42)
                    model.fit(X_train_p, y_train)
                    y_pred = model.predict(X_dev_p)
                    score_dev = f1_score(y_dev, y_pred, average=eval_strat)
                    metric = "F1"

                # 4. Guardado del modelo
                folder_path = os.path.join("modelos", csv_id, method)
                os.makedirs(folder_path, exist_ok=True)

                params_str = f"depth={depth}_crit={crit}"
                model_name = f"tree_{params_str}.sav"

                # --- GUARDADO DE RESULTADOS CSV ---
                dict_predicciones[model_name] = y_pred

                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    # --- CASO RANDOM FOREST ---
    elif method == 'forest':
        # 1. Extraemos los hiperparámetros del JSON
        params_cfg = config.get('hyperparameters_forest', {})
        lista_n_estimators = params_cfg.get('n_estimators', [50, 100])  # Número de árboles
        lista_depth = params_cfg.get('max_depth', [None, 5, 10])
        lista_features = params_cfg.get('max_features', ['sqrt', 'log2'])  # Cuántas variables ve cada árbol

        print(f"🌲🌲 Iniciando entrenamiento de Random Forest...")

        # 2. Triple barrido de hiperparámetros
        for n_est in lista_n_estimators:
            for depth in lista_depth:
                for feat in lista_features:

                    # 3. Ajuste según la tarea
                    if task == 'regression':
                        model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, max_features=feat,
                                                      random_state=42, n_jobs=-1)
                        model.fit(X_train_p, y_train)
                        y_pred = model.predict(X_dev_p)
                        score_dev = r2_score(y_dev, y_pred)
                        metric = "R2"
                    else:
                        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, max_features=feat,
                                                       random_state=42, n_jobs=-1)
                        model.fit(X_train_p, y_train)
                        y_pred = model.predict(X_dev_p)
                        score_dev = f1_score(y_dev, y_pred, average=eval_strat)
                        metric = "F1"

                    # 4. Guardado
                    folder_path = os.path.join("modelos", csv_id, method)
                    os.makedirs(folder_path, exist_ok=True)

                    params_str = f"n={n_est}_d={depth}_f={feat}"
                    model_name = f"forest_{params_str}.sav"

                    # --- GUARDADO DE RESULTADOS CSV ---
                    dict_predicciones[model_name] = y_pred

                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    # --- CASO LOGISTIC REGRESSION ---
    elif method == 'logistic':
        # 1. Extraemos los hiperparámetros del JSON
        params_cfg = config.get('hyperparameters_logistic', {})
        # C es el inverso de la fuerza de regularización (valores más pequeños = mayor regularización)
        lista_C = params_cfg.get('C', [0.1, 1.0, 10.0])
        # Multi_class 'multinomial' es ideal para el proyecto (positivo, negativo, neutro)
        lista_solver = params_cfg.get('solver', ['lbfgs', 'liblinear'])

        print(f"📈 Iniciando entrenamiento de Logistic Regression...")

        for c_val in lista_C:
            for solv in lista_solver:
                # La regresión logística solo se usará para clasificación en este proyecto
                if task == 'classification':
                    # Usamos un max_iter alto (1000) para evitar errores de convergencia con texto
                    model = LogisticRegression(C=c_val, solver=solv, max_iter=1000, random_state=42)
                    model.fit(X_train_p, y_train)
                    y_pred = model.predict(X_dev_p)

                    # Evaluamos usando la estrategia del JSON (Macro-Fscore solicitado)
                    score_dev = f1_score(y_dev, y_pred, average=eval_strat)
                    metric = "F1"

                    # 4. Guardado del modelo y resultados
                    folder_path = os.path.join("modelos", csv_id, method)
                    os.makedirs(folder_path, exist_ok=True)

                    params_str = f"C={c_val}_solver={solv}"
                    model_name = f"{csv_id}_{task}_logistic_{params_str}.sav"

                    # Guardamos la predicción en el diccionario unificado para tu tabla comparativa
                    dict_predicciones[model_name] = y_pred

                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")



    # --- SELECCIÓN DEL MEJOR MODELO DEL ENTRENAMIENTO ---
    import glob

    # Buscamos todos los modelos que acabamos de guardar en la carpeta del metodo
    model_files = glob.glob(os.path.join(folder_path, "*.sav"))
    best_model_path = None
    max_score = -1

    # Pequeño truco: como no queremos re-entrenar, leemos los resultados que imprimimos antes
    # O mejor, cargamos cada uno y probamos rápido en Dev
    for m_path in model_files:
        tmp_model = joblib.load(m_path)
        y_pred = tmp_model.predict(X_dev_p)

        if task == 'regression':
            current_score = r2_score(y_dev, y_pred)
        else:
            current_score = f1_score(y_dev, y_pred, average=eval_strat)

        if current_score > max_score:
            max_score = current_score
            best_model_path = m_path

        # --- GUARDADO DEL MODELO GANADOR Y SUS MÉTRICAS ---
        if best_model_path:
            # 1. Crear carpetas necesarias
            os.makedirs("modelos_finales", exist_ok=True)
            metrics_folder = "resultados_finales"
            os.makedirs(metrics_folder, exist_ok=True)

            # 2. Gestionar nombres de archivos
            detalles_params = os.path.basename(best_model_path)
            final_name = f"MEJOR_{csv_id}_{detalles_params}"
            final_path = os.path.join("modelos_finales", final_name)

            # 3. Guardar el modelo físico (.sav)
            mejor_modelo = joblib.load(best_model_path)
            joblib.dump(mejor_modelo, final_path)

            # 4. Generar predicciones para el reporte de métricas
            y_pred_mejor = mejor_modelo.predict(X_dev_p)

            # 5. Crear el diccionario de métricas
            metrics_dict = {
                'dataset': [csv_id],
                'algoritmo': [method],
                'configuracion': [detalles_params],
                'accuracy': [accuracy_score(y_dev, y_pred_mejor)]
            }

            if task == 'classification':
                metrics_dict['precision'] = [precision_score(y_dev, y_pred_mejor, average=eval_strat)]
                metrics_dict['recall'] = [recall_score(y_dev, y_pred_mejor, average=eval_strat)]
                metrics_dict['f1_score'] = [f1_score(y_dev, y_pred_mejor, average=eval_strat)]
            else:
                metrics_dict['r2_score'] = [r2_score(y_dev, y_pred_mejor)]

            # 6. Guardar el CSV de métricas
            df_metrics = pd.DataFrame(metrics_dict)
            ruta_csv_metrics = os.path.join(metrics_folder, f"metricas_{method}_{csv_id}.csv")
            df_metrics.to_csv(ruta_csv_metrics, index=False)

            # --- SALIDA POR CONSOLA ---
            print("\n" + "⭐" * 40)
            print(f"🏆 GANADOR: {final_name}")
            print(f"📊 PUNTUACIÓN (DEV): {max_score:.4f}")

            if task == 'classification':
                print("\n📋 REPORTE DETALLADO:")
                print(classification_report(y_dev, y_pred_mejor))

            print(f"📁 Modelo en: {final_path}")
            print(f"📄 Métricas en: {ruta_csv_metrics}")
            print("⭐" * 40)


    # --- GENERACIÓN DEL CSV UNIFICADO DE PREDICCIONES ---
    df_predicciones_final = pd.DataFrame(dict_predicciones)
    ruta_unificada = os.path.join("csv", csv_id, f"comparativa_predicciones_{method}.csv")
    df_predicciones_final.to_csv(ruta_unificada, index=False)

    print("\n" + "📊 " * 10)
    print(f"TABLA UNIFICADA CREADA: {ruta_unificada}")
    print("📊 " * 10 + "\n")


    # --- EXPORTACIÓN DE DATOS PARA EL SCRIPT DE EVALUACIÓN ---
    # Creamos una carpeta para los datos ya limpios y procesados
    processed_data_path = os.path.join("datos_preprocesados", csv_id)
    os.makedirs(processed_data_path, exist_ok=True)

    # Guardamos el Test preprocesado (X y etiquetas)
    # Lo guardamos con índice para no perder la referencia si hubo dropna
    df_test_final_p = pd.concat([X_test_p, y_test], axis=1)

    ruta_test_p = os.path.join(processed_data_path, f"{csv_id}_test_ready.csv")
    df_test_final_p.to_csv(ruta_test_p, index=False)

    # Guardamos el Train preprocesado (X y etiquetas)
    # Lo guardamos con índice para no perder la referencia si hubo dropna
    df_train_final_p = pd.concat([X_train_p, y_train], axis=1)

    ruta_test_p = os.path.join(processed_data_path, f"{csv_id}_train_ready.csv")
    df_train_final_p.to_csv(ruta_test_p, index=False)

    # Guardamos el Dev preprocesado (X y etiquetas)
    # Lo guardamos con índice para no perder la referencia si hubo dropna
    df_dev_final_p = pd.concat([X_dev_p, y_dev], axis=1)

    ruta_test_p = os.path.join(processed_data_path, f"{csv_id}_dev_ready.csv")
    df_dev_final_p.to_csv(ruta_test_p, index=False)

    print(f"📦 DATOS LISTOS: El CSV para evaluación se ha guardado en: {ruta_test_p}")


if __name__ == "__main__":
    train()