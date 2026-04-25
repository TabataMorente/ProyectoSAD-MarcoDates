# -*- coding: utf-8 -*-
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import (
    f1_score, r2_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, classification_report
)


# ==========================================
# FUNCIONES DE MÉTRICAS AVANZADAS
# ==========================================
def print_advanced_metrics(y_test, y_pred):
    """
    Calcula e imprime Accuracy, Precision, Recall, Specificity y F1 por cada clase.
    """
    labels = sorted(list(set(y_test) | set(y_pred)))

    print(f"\n -> Accuracy Global (Exactitud): {accuracy_score(y_test, y_pred):.4f}")

    # Obtener métricas base por cada clase
    precision = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    # Cálculo manual de Especificidad (Specificity) por clase
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    # Evitar división por cero
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)

    # Formateo de tabla
    print(f"\n{'Clase':<12} | {'Precision':<10} | {'Recall':<10} | {'Specif.':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for i, label in enumerate(labels):
        print(
            f"{str(label):<12} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {specificity[i]:<10.4f} | {f1[i]:<10.4f}")

    print("-" * 70)
    # Medias Globales
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    print(f"{'MEDIA MACRO':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {f1_macro:<10.4f}")
    print(f"{'MEDIA MICRO':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {f1_micro:<10.4f}")


# ==========================================
# FUNCIÓN PRINCIPAL DE EVALUACIÓN
# ==========================================
def evaluar():
    # 1. Validación de entrada
    if len(sys.argv) < 3:
        print("Uso: python eval.py <config.json> <nombre_dataset>")
        print("Ejemplo: python eval.py config.json ventas")
        sys.exit(1)

    config_path = sys.argv[1]
    csv_id = sys.argv[2]  # El nombre que usaste en train (ej: 'ventas')

    # Cargar Configuración
    with open(config_path, 'r') as f:
        config = json.load(f)

    target = config.get('target')
    method = config.get('method')  # 'knn', 'forest', 'tree', 'bayes'
    task = config.get('task', 'classification')

    # 2. CARGA DE DATOS PREPROCESADOS
    # El archivo generado por train.py en la carpeta de datos preprocesados
    ruta_test_ready = os.path.join("preprocesado", csv_id, f"{csv_id}_test_ready.csv")

    if not os.path.exists(ruta_test_ready):
        print(f"❌ ERROR: No se encuentra el CSV listo en: {ruta_test_ready}")
        print("Asegúrate de haber ejecutado train.py primero.")
        return

    df_test = pd.read_csv(ruta_test_ready)

    # AÑADE ESTA LÍNEA para limpiar la columna fantasma si existe:
    if 'Unnamed: 0' in df_test.columns:
        df_test = df_test.drop(columns=['Unnamed: 0'])

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # 3. BÚSQUEDA DEL MODELO GANADOR (.sav)
    # Buscamos en 'modelos_finales' el archivo que coincida con dataset y algoritmo
    search_pattern = os.path.join("resultados_clasificacion", "Tinder","mejor_modelo", f"MEJOR_{csv_id}_{method}*.sav")
    modelos_encontrados = glob.glob(search_pattern)

    if not modelos_encontrados:
        print(f"❌ ERROR: No se encontró el modelo final para '{method}' en {search_pattern}")
        return

    # Cargamos el primer modelo que coincida con el patrón
    ruta_modelo = modelos_encontrados[0]
    print("\n" + "=" * 65)
    print(f" 🚀 INICIANDO AUDITORÍA FINAL")
    print(f" 🏆 MODELO CARGADO: {os.path.basename(ruta_modelo)}")
    print("=" * 65)

    # 4. PREDICCIÓN
    modelo = joblib.load(ruta_modelo)
    y_pred = modelo.predict(X_test)

    # 5. GENERACIÓN DE REPORTES
    if task == 'regression':
        r2 = r2_score(y_test, y_pred)
        print(f"\n📊 RESULTADO FINAL (R2-Score): {r2:.4f}")
    else:
        # Matriz de Confusión visual
        print("\n[ MATRIZ DE CONFUSIÓN ]")
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(list(set(y_test) | set(y_pred)))
        df_cm = pd.DataFrame(
            cm,
            index=[f"Real {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels]
        )
        print(df_cm)

        # Informe estadístico detallado
        print("\n[ MÉTRICAS DE CLASIFICACIÓN ]")
        print_advanced_metrics(y_test, y_pred)

    # 6. EXPORTAR PREDICCIONES DE AUDITORÍA
    salida_dir = os.path.join("resultados_clasificacion", csv_id, "evaluacion")
    os.makedirs(salida_dir, exist_ok=True)

    df_audit = pd.DataFrame({
        'Valor_Real': y_test,
        'Prediccion': y_pred
    })

    ruta_salida = os.path.join(salida_dir, f"audit_final_{method}.csv")
    df_audit.to_csv(ruta_salida, index=False)

    print(f"PROCESO TERMINADO. Predicciones guardadas en: {ruta_salida}")


if __name__ == "__main__":
    evaluar()