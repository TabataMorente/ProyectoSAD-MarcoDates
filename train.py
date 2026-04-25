# -*- coding: utf-8 -*-
import sys
import os
import json
import joblib
import glob
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

    config       = load_config(sys.argv[2])
    target_global = config.get('target')
    df_raw       = pd.read_csv(sys.argv[1])

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

    folder_path = os.path.join("modelos", csv_id, method)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join("csv", csv_id), exist_ok=True)

    dict_predicciones = {'Valor_Real': dev_target.values}

    print(f"🚀 Iniciando entrenamiento con método: {method}...")

    # ════════════════════════════════════════════════════════════════════════
    # NAIVE BAYES
    # ════════════════════════════════════════════════════════════════════════
    if method == 'bayes':
        params_cfg = config.get('hyperparameters_bayes', {})
        b_type     = params_cfg.get('bayes_type', 'gaussian')

        if b_type == 'multinomial':
            # Verificación: MultinomialNB necesita features no negativas
            if (train_features.values < 0).any():
                print("⚠️  MultinomialNB detectó valores negativos. Usando ComplementNB en su lugar.")
                b_type = 'complement'
            else:
                for a in params_cfg.get('alpha', [1.0]):
                    model      = MultinomialNB(alpha=a)
                    model.fit(train_features, train_target)
                    y_pred     = model.predict(dev_features)
                    score_dev  = f1_score(dev_target, y_pred, average=eval_strat)
                    model_name = f"bayes_multi_alpha={a}.sav"
                    dict_predicciones[model_name] = y_pred
                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

        # ComplementNB: ideal para texto con clases desbalanceadas
        if b_type == 'complement':
            for a in params_cfg.get('alpha', [1.0]):
                model      = ComplementNB(alpha=a)
                model.fit(train_features, train_target)
                y_pred     = model.predict(dev_features)
                score_dev  = f1_score(dev_target, y_pred, average=eval_strat)
                model_name = f"bayes_complement_alpha={a}.sav"
                dict_predicciones[model_name] = y_pred
                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

        elif b_type == 'bernoulli':
            for a in params_cfg.get('alpha', [1.0]):
                model      = BernoulliNB(alpha=a)
                model.fit(train_features, train_target)
                y_pred     = model.predict(dev_features)
                score_dev  = f1_score(dev_target, y_pred, average=eval_strat)
                model_name = f"bayes_bern_alpha={a}.sav"
                dict_predicciones[model_name] = y_pred
                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

        elif b_type == 'gaussian':
            for sm in params_cfg.get('var_smoothing', [1e-9]):
                model      = GaussianNB(var_smoothing=sm)
                model.fit(train_features, train_target)
                y_pred     = model.predict(dev_features)
                score_dev  = f1_score(dev_target, y_pred, average=eval_strat)
                model_name = f"bayes_gauss_sm={sm}.sav"
                dict_predicciones[model_name] = y_pred
                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | F1-Dev: {score_dev:.4f}")

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
                        model     = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
                        model.fit(train_features, train_target)
                        y_pred    = model.predict(dev_features)
                        score_dev = r2_score(dev_target, y_pred)
                        metric    = "R2"
                    else:
                        model     = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
                        model.fit(train_features, train_target)
                        y_pred    = model.predict(dev_features)
                        score_dev = f1_score(dev_target, y_pred, average=eval_strat)
                        metric    = "F1"

                    params_str = f"k={k}_p={p}_w={w}"
                    model_name = f"knn_{params_str}.sav"
                    dict_predicciones[model_name] = y_pred
                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # ÁRBOL DE DECISIÓN
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'tree':
        params_cfg   = config.get('hyperparameters_tree', {})
        lista_depth  = params_cfg.get('max_depth', [5, 10])
        lista_crit   = params_cfg.get('criterion', params_cfg.get('criterio', ['gini']))

        for depth in lista_depth:
            for crit in lista_crit:
                if task == 'regression':
                    model     = DecisionTreeRegressor(max_depth=depth, criterion='squared_error', random_state=42)
                    model.fit(train_features, train_target)
                    y_pred    = model.predict(dev_features)
                    score_dev = r2_score(dev_target, y_pred)
                    metric    = "R2"
                else:
                    model     = DecisionTreeClassifier(max_depth=depth, criterion=crit,
                                                        class_weight='balanced', random_state=42)
                    model.fit(train_features, train_target)
                    y_pred    = model.predict(dev_features)
                    score_dev = f1_score(dev_target, y_pred, average=eval_strat)
                    metric    = "F1"

                params_str = f"d={depth}_c={crit}"
                model_name = f"tree_{params_str}.sav"
                dict_predicciones[model_name] = y_pred
                joblib.dump(model, os.path.join(folder_path, model_name))
                print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # RANDOM FOREST
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'forest':
        params_cfg        = config.get('hyperparameters_forest', {})
        lista_n_estimators = params_cfg.get('n_estimators', [100])
        lista_depth        = params_cfg.get('max_depth', [None, 10])
        lista_features     = params_cfg.get('max_features', ['sqrt', 'log2'])

        print("🌲🌲 Iniciando entrenamiento de Random Forest...")

        for n_est in lista_n_estimators:
            for depth in lista_depth:
                for feat in lista_features:
                    if task == 'regression':
                        model     = RandomForestRegressor(
                            n_estimators=n_est, max_depth=depth, max_features=feat,
                            random_state=42, n_jobs=-1,
                        )
                        model.fit(train_features, train_target)
                        y_pred    = model.predict(dev_features)
                        score_dev = r2_score(dev_target, y_pred)
                        metric    = "R2"
                    else:
                        model     = RandomForestClassifier(
                            n_estimators=n_est, max_depth=depth, max_features=feat,
                            random_state=42, n_jobs=-1, class_weight='balanced',
                        )
                        model.fit(train_features, train_target)
                        y_pred    = model.predict(dev_features)
                        score_dev = f1_score(dev_target, y_pred, average=eval_strat)
                        metric    = "F1"

                    params_str = f"n={n_est}_d={depth}_f={feat}"
                    model_name = f"forest_{params_str}.sav"
                    dict_predicciones[model_name] = y_pred
                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # LOGISTIC REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    elif method == 'logistic':
        params_cfg  = config.get('hyperparameters_logistic', {})
        lista_C     = params_cfg.get('C', [0.1, 1.0, 10.0])
        lista_solver = params_cfg.get('solver', ['lbfgs'])

        print("📈 Iniciando entrenamiento de Logistic Regression...")

        for c_val in lista_C:
            for solv in lista_solver:
                if task == 'classification':
                    model = LogisticRegression(
                        C=c_val, solver=solv,
                        max_iter=1000, random_state=42,
                        class_weight='balanced',
                    )
                    model.fit(train_features, train_target)
                    y_pred    = model.predict(dev_features)
                    score_dev = f1_score(dev_target, y_pred, average=eval_strat)
                    metric    = "F1"

                    params_str = f"C={c_val}_solver={solv}"
                    model_name = f"logistic_{params_str}.sav"
                    dict_predicciones[model_name] = y_pred
                    joblib.dump(model, os.path.join(folder_path, model_name))
                    print(f"✅ Guardado: {model_name} | {metric}-Dev: {score_dev:.4f}")

    else:
        print(f"❌ Método '{method}' no reconocido.")
        sys.exit(1)

    # ════════════════════════════════════════════════════════════════════════
    # SELECCIÓN DEL MEJOR MODELO
    # ════════════════════════════════════════════════════════════════════════
    model_files    = glob.glob(os.path.join(folder_path, "*.sav"))
    best_model_path = None
    max_score       = -1

    for m_path in model_files:
        tmp_model = joblib.load(m_path)
        y_pred    = tmp_model.predict(dev_features)

        if task == 'regression':
            current_score = r2_score(dev_target, y_pred)
        else:
            current_score = f1_score(dev_target, y_pred, average=eval_strat)

        if current_score > max_score:
            max_score       = current_score
            best_model_path = m_path

    if best_model_path:
        os.makedirs("modelos_finales",    exist_ok=True)
        os.makedirs("resultados_finales", exist_ok=True)

        detalles_params  = os.path.basename(best_model_path)
        final_name       = f"MEJOR_{csv_id}_{detalles_params}"
        final_path       = os.path.join("modelos_finales", final_name)

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
        ruta_csv   = os.path.join("resultados_finales", f"metricas_{method}_{csv_id}.csv")
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

        # ── Exportar datos preprocesados para evaluación externa ────────────
        processed_path = os.path.join("datos_preprocesados", csv_id)
        os.makedirs(processed_path, exist_ok=True)

        pd.concat([test_features,  test_target],  axis=1).to_csv(
            os.path.join(processed_path, f"{csv_id}_test_ready.csv"),  index=False)
        pd.concat([train_features, train_target], axis=1).to_csv(
            os.path.join(processed_path, f"{csv_id}_train_ready.csv"), index=False)
        pd.concat([dev_features,   dev_target],   axis=1).to_csv(
            os.path.join(processed_path, f"{csv_id}_dev_ready.csv"),   index=False)

        print(f"📦 Datos preprocesados guardados en: {processed_path}/")
    else:
        print("⚠️ No se encontraron archivos .sav para evaluar.")


if __name__ == "__main__":
    train()