# -*- coding: utf-8 -*-
import sys
import os
import json
import joblib
import pandas as pd

from preprocessing import pipeline_preprocesamiento

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, r2_score,
    accuracy_score, precision_score, recall_score, classification_report,
)


def load_config(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def train():
    if len(sys.argv) < 3:
        print("Uso: python train.py <data.csv> <config_file.json>")
        sys.exit(1)

    config        = load_config(sys.argv[2])
    target_global = config.get('target')
    df_raw        = pd.read_csv(sys.argv[1])

    # ── División 70 / 15 / 15 ────────────────────────────────────────────────
    df_train_raw, df_temp = train_test_split(
        df_raw, test_size=0.30, random_state=42,
        stratify=df_raw[target_global],
    )
    df_temp = df_temp.reset_index(drop=True)

    df_dev_raw, df_test_raw = train_test_split(
        df_temp, test_size=0.50, random_state=42,
        stratify=df_temp[target_global],
    )

    print(f"📊 Reparto: Train={len(df_train_raw)} | Dev={len(df_dev_raw)} | Test={len(df_test_raw)}")

    # ── Preprocesado ─────────────────────────────────────────────────────────
    df_train, df_test, df_dev = pipeline_preprocesamiento(
        df_train_raw, df_test_raw, df_dev_raw, config
    )

    print(f"📊 Reparto tras preprocesado: Train={len(df_train)} | Dev={len(df_dev)} | Test={len(df_test)}")

    # ── Separación X / y ─────────────────────────────────────────────────────
    train_features = df_train.drop(columns=[target_global])
    train_target   = df_train[target_global]

    dev_features = df_dev.drop(columns=[target_global])
    dev_target   = df_dev[target_global]

    test_features = df_test.drop(columns=[target_global])
    test_target   = df_test[target_global]

    # ── Configuración general ─────────────────────────────────────────────────
    method     = config.get('method', 'knn')
    task       = config.get('task', 'classification')
    csv_id     = os.path.basename(sys.argv[1]).split('.')[0]
    eval_strat = config.get('evaluation', 'macro')

    folder_path = os.path.join("resultados_clasificacion", csv_id, "modelos", method)
    os.makedirs(folder_path, exist_ok=True)

    # ── Registro de modelos entrenados en ESTA ejecución ─────────────────────
    # Clave: evitar evaluar modelos de ejecuciones anteriores que queden en disco.
    modelos_esta_ejecucion = {}   # { ruta_pkl: score_dev }

    print(f"🚀 Iniciando entrenamiento con método: {method}...")

    # ════════════════════════════════════════════════════════════════════════
    # FUNCIÓN AUXILIAR: entrenar, evaluar y registrar
    # ════════════════════════════════════════════════════════════════════════
    def registrar_modelo(model, model_name):
        """Entrena, evalúa en dev, guarda en .pkl y registra en el dict de ejecución."""
        model.fit(train_features, train_target)
        y_pred = model.predict(dev_features)

        if task == 'regression':
            score_dev = r2_score(dev_target, y_pred)
            metric    = "R2"
        else:
            score_dev = f1_score(dev_target, y_pred, average=eval_strat)
            metric    = "F1"

        # Guardamos en .pkl (antes era .sav)
        pkl_name = model_name.replace(".sav", ".pkl")
        pkl_path = os.path.join(folder_path, pkl_name)
        joblib.dump(model, pkl_path)

        modelos_esta_ejecucion[pkl_path] = score_dev
        print(f"✅ Guardado: {pkl_name} | {metric}-Dev: {score_dev:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # NAIVE BAYES
    # ════════════════════════════════════════════════════════════════════════
    if method == 'bayes':
        params_cfg = config.get('hyperparameters_bayes', {})
        b_type     = params_cfg.get('bayes_type', 'gaussian')

        if b_type == 'multinomial' and (train_features.values < 0).any():
            print("⚠️  MultinomialNB detectó valores negativos. Usando ComplementNB en su lugar.")
            b_type = 'complement'

        if b_type == 'multinomial':
            for a in params_cfg.get('alpha', [1.0]):
                for sm in params_cfg.get('var_smoothing', [1e-9]):
                    registrar_modelo(MultinomialNB(alpha=a),
                                     f"bayes_multi_alpha={a}_sm={sm}.pkl")

        elif b_type == 'complement':
            for a in params_cfg.get('alpha', [1.0]):
                for sm in params_cfg.get('var_smoothing', [1e-9]):
                    registrar_modelo(ComplementNB(alpha=a),
                                     f"bayes_complement_alpha={a}_sm={sm}.pkl")

        elif b_type == 'bernoulli':
            for a in params_cfg.get('alpha', [1.0]):
                for sm in params_cfg.get('var_smoothing', [1e-9]):
                    registrar_modelo(BernoulliNB(alpha=a),
                                     f"bayes_bern_alpha={a}_sm={sm}.pkl")

        elif b_type == 'gaussian':
            for sm in params_cfg.get('var_smoothing', [1e-9]):
                registrar_modelo(GaussianNB(var_smoothing=sm),
                                 f"bayes_gauss_sm={sm}.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # KNN
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'knn':
        params_cfg   = config.get('hyperparameters_knn', {})
        k_min, k_max = params_cfg.get('k_range', [1, 5])
        lista_p      = params_cfg.get('p', [1, 2])
        lista_w      = params_cfg.get('weights', ["uniform", "distance"])
        step         = params_cfg.get('step', 2)
        lista_k      = list(range(k_min, k_max + 1, step))

        for k in lista_k:
            for p in lista_p:
                for w in lista_w:
                    if task == 'regression':
                        model = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
                    else:
                        model = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
                    registrar_modelo(model, f"knn_k={k}_p={p}_w={w}.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # DECISION TREE
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'tree':
        params_cfg  = config.get('hyperparameters_tree', {})
        lista_depth = params_cfg.get('max_depth', [None, 5, 10])
        lista_crit  = params_cfg.get('criterio', ['gini'])

        for depth in lista_depth:
            for crit in lista_crit:
                if task == 'regression':
                    model = DecisionTreeRegressor(max_depth=depth, criterion='squared_error',
                                                  random_state=42)
                else:
                    model = DecisionTreeClassifier(max_depth=depth, criterion=crit,
                                                   class_weight='balanced', random_state=42)
                registrar_modelo(model, f"tree_d={depth}_c={crit}.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # RANDOM FOREST
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'forest':
        params_cfg         = config.get('hyperparameters_forest', {})
        lista_n_estimators = params_cfg.get('n_estimators', [100])
        lista_depth        = params_cfg.get('max_depth', [None, 10])
        lista_features     = params_cfg.get('max_features', ['sqrt', 'log2'])

        print("🌲🌲 Iniciando entrenamiento de Random Forest...")

        for n_est in lista_n_estimators:
            for depth in lista_depth:
                for feat in lista_features:
                    if task == 'regression':
                        model = RandomForestRegressor(
                            n_estimators=n_est, max_depth=depth, max_features=feat,
                            random_state=42, n_jobs=-1,
                        )
                    else:
                        model = RandomForestClassifier(
                            n_estimators=n_est, max_depth=depth, max_features=feat,
                            random_state=42, n_jobs=-1, class_weight='balanced',
                        )
                    registrar_modelo(model, f"forest_n={n_est}_d={depth}_f={feat}.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # LOGISTIC REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'logistic':
        params_cfg   = config.get('hyperparameters_logistic', {})
        lista_C      = params_cfg.get('C', [0.1, 1.0, 10.0])
        lista_solver = params_cfg.get('solver', ['lbfgs'])

        print("📈 Iniciando entrenamiento de Logistic Regression...")

        for c_val in lista_C:
            for solv in lista_solver:
                model = LogisticRegression(
                    C=c_val, solver=solv,
                    max_iter=1000, random_state=42,
                    class_weight='balanced',
                )
                registrar_modelo(model, f"logistic_C={c_val}_solver={solv}.pkl")

    else:
        print(f"❌ Método '{method}' no reconocido.")
        sys.exit(1)

    # ════════════════════════════════════════════════════════════════════════
    # SELECCIÓN DEL MEJOR MODELO (solo de esta ejecución)
    # ════════════════════════════════════════════════════════════════════════
    if not modelos_esta_ejecucion:
        print("⚠️ No se entrenó ningún modelo en esta ejecución.")
        sys.exit(1)

    best_model_path = max(modelos_esta_ejecucion, key=modelos_esta_ejecucion.get)
    max_score       = modelos_esta_ejecucion[best_model_path]

    dir_mejor    = os.path.join("resultados_clasificacion", csv_id, "mejor_modelo")
    dir_metricas = os.path.join("resultados_clasificacion", csv_id, "metricas")
    os.makedirs(dir_mejor,    exist_ok=True)
    os.makedirs(dir_metricas, exist_ok=True)

    detalles_params = os.path.basename(best_model_path)
    final_name      = f"MEJOR_{csv_id}_{detalles_params}"
    final_path      = os.path.join(dir_mejor, final_name)

    mejor_modelo = joblib.load(best_model_path)
    joblib.dump(mejor_modelo, final_path)

    y_pred_mejor = mejor_modelo.predict(dev_features)

    metrics_dict = {
        'dataset':       [csv_id],
        'algoritmo':     [method],
        'configuracion': [detalles_params],
        'accuracy':      [accuracy_score(dev_target, y_pred_mejor)],
    }

    if task == 'classification':
        metrics_dict['precision'] = [precision_score(dev_target, y_pred_mejor, average=eval_strat)]
        metrics_dict['recall']    = [recall_score(dev_target,    y_pred_mejor, average=eval_strat)]
        metrics_dict['f1_score']  = [f1_score(dev_target,        y_pred_mejor, average=eval_strat)]
    else:
        metrics_dict['r2_score']  = [r2_score(dev_target, y_pred_mejor)]

    df_metrics = pd.DataFrame(metrics_dict)
    ruta_csv   = os.path.join(dir_metricas, f"metricas_{method}_{csv_id}.csv")
    df_metrics.to_csv(ruta_csv, index=False)

    print("\n" + "⭐" * 40)
    print(f"🏆 MODELO GANADOR: {final_name}")
    print(f"📊 MEJOR F1-SCORE (DEV): {max_score:.4f}")

    if task == 'classification':
        print("\n📋 REPORTE FINAL DE CLASIFICACIÓN:")
        print(classification_report(dev_target, y_pred_mejor))

    print(f"📁 Modelo guardado en: {final_path}")
    print(f"📄 CSV de métricas:    {ruta_csv}")
    print("⭐" * 40)

    # ── Exportar datos de test preprocesados para evaluación externa ──────────
    processed_path = os.path.join("preprocesado", csv_id)
    os.makedirs(processed_path, exist_ok=True)

    pd.concat([test_features, test_target], axis=1).to_csv(
        os.path.join(processed_path, f"{csv_id}_test_ready.csv"), index=False
    )
    print(f"📦 Datos preprocesados guardados en: {processed_path}/")

    # ── Escribir la ruta exacta del mejor modelo para que evaluar.py la use ──
    # Así el evaluador no tiene que buscar en disco: lee esta ruta directamente.
    ruta_mejor_txt = os.path.join(dir_mejor, "ultimo_mejor_modelo.txt")
    with open(ruta_mejor_txt, 'w', encoding='utf-8') as f:
        f.write(final_path)
    print(f"📝 Ruta del mejor modelo escrita en: {ruta_mejor_txt}")


if __name__ == "__main__":
    train()