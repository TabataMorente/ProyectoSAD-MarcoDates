# -*- coding: utf-8 -*-
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from sklearn.metrics import (
    f1_score, r2_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, classification_report
)


# ==========================================
# FUNCIONES DE MÉTRICAS AVANZADAS
# ==========================================
def build_metrics_text(y_test, y_pred):
    """
    Calcula Accuracy, Precision, Recall, Specificity y F1 por cada clase.
    Devuelve el bloque completo como string (para imprimir y para guardar).
    """
    labels = sorted(list(set(y_test) | set(y_pred)))
    lines = []

    accuracy = accuracy_score(y_test, y_pred)
    lines.append(f"\n -> Accuracy Global (Exactitud): {accuracy:.4f}")

    precision = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    recall    = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    f1        = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    # Especificidad por clase
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)

    lines.append(f"\n{'Clase':<12} | {'Precision':<10} | {'Recall':<10} | {'Specif.':<10} | {'F1-Score':<10}")
    lines.append("-" * 70)
    for i, label in enumerate(labels):
        lines.append(
            f"{str(label):<12} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {specificity[i]:<10.4f} | {f1[i]:<10.4f}"
        )
    lines.append("-" * 70)

    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    lines.append(f"{'MEDIA MACRO':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {f1_macro:<10.4f}")
    lines.append(f"{'MEDIA MICRO':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {f1_micro:<10.4f}")

    return "\n".join(lines)


def print_advanced_metrics(y_test, y_pred):
    """Imprime las métricas por consola."""
    print(build_metrics_text(y_test, y_pred))


# ==========================================
# CONSTRUCCIÓN DEL TÍTULO CON PARÁMETROS
# ==========================================
def build_titulo(config, csv_id, nombre_modelo):
    """
    Genera el bloque de título que se escribirá en el .txt.
    Incluye: dataset, mé_todo, hiperparámetros activos y timestamp.
    """
    method    = config.get('method', 'unknown')
    task      = config.get('task', 'classification')
    target    = config.get('target', 'unknown')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Hiperparámetros del mé_todo usado
    hparam_key = f"hyperparameters_{method}"
    hparams    = config.get(hparam_key, {})

    lines = []
    lines.append("=" * 70)

    if hparams:
        lines.append(f"  Hiperparámetros ({hparam_key}):")
        for k, v in hparams.items():
            lines.append(f"    · {k}: {v}")

    # Preprocesado relevante (resumen compacto)
    prep = config.get('preprocessing', {})
    if prep:
        lines.append("  Preprocesado:")
        campos_interes = [
            'scaling', 'sampling_strategy', 'categorical_encoding',
            'text_process_method', "ngram_range", "limite_palabras"
        ]
        for campo in campos_interes:
            if campo in prep:
                lines.append(f"    · {campo}: {prep[campo]}")

    return "\n".join(lines)


# ==========================================
# EXPORTAR AL .TXT ACUMULATIVO
# ==========================================
def exportar_metricas_txt(salida_dir, csv_id, method, titulo, metricas_texto):
    """
    Añade al archivo metricas_historial.txt el bloque de la evaluación actual.
    Si el archivo no existe, lo crea.
    """
    ruta_txt = os.path.join(salida_dir, "metricas_historial.txt")

    bloque = []
    bloque.append("\n")
    bloque.append(titulo)
    bloque.append("\n[ MÉTRICAS DE CLASIFICACIÓN ]")
    bloque.append(metricas_texto)
    bloque.append("\n")

    contenido = "\n".join(bloque)

    with open(ruta_txt, 'a', encoding='utf-8') as f:
        f.write(contenido)

    print(f"  → Métricas añadidas al historial: {ruta_txt}")


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
    csv_id      = sys.argv[2]

    # Cargar Configuración
    with open(config_path, 'r') as f:
        config = json.load(f)

    target = config.get('target')
    method = config.get('method')
    task   = config.get('task', 'classification')

    # 2. CARGA DE DATOS PREPROCESADOS
    ruta_test_ready = os.path.join("preprocesado", csv_id, f"{csv_id}_test_ready.csv")

    if not os.path.exists(ruta_test_ready):
        print(f"❌ ERROR: No se encuentra el CSV listo en: {ruta_test_ready}")
        print("Asegúrate de haber ejecutado train.py primero.")
        return

    df_test = pd.read_csv(ruta_test_ready)

    if 'Unnamed: 0' in df_test.columns:
        df_test = df_test.drop(columns=['Unnamed: 0'])

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # 3. BÚSQUEDA DEL MODELO GANADOR (.pkl)
    search_pattern   = os.path.join("resultados_clasificacion", csv_id, "mejor_modelo", f"MEJOR_{csv_id}_{method}*.pkl")
    modelos_encontrados = glob.glob(search_pattern)

    if not modelos_encontrados:
        print(f"❌ ERROR: No se encontró el modelo final para '{method}' en {search_pattern}")
        return

    ruta_modelo  = modelos_encontrados[0]
    nombre_modelo = os.path.basename(ruta_modelo)

    print("\n" + "=" * 65)
    print(f" 🚀 INICIANDO AUDITORÍA FINAL")
    print(f" 🏆 MODELO CARGADO: {nombre_modelo}")
    print("=" * 65)

    # 4. PREDICCIÓN
    modelo = joblib.load(ruta_modelo)
    y_pred = modelo.predict(X_test)

    # 5. GENERACIÓN DE REPORTES
    salida_dir = os.path.join("resultados_clasificacion", csv_id, "evaluacion")
    os.makedirs(salida_dir, exist_ok=True)

    if task == 'regression':
        r2 = r2_score(y_test, y_pred)
        print(f"\n📊 RESULTADO FINAL (R2-Score): {r2:.4f}")

        # Exportar al .txt también para regresión
        titulo = build_titulo(config, csv_id, nombre_modelo)
        bloque_r2 = f"\n R2-Score: {r2:.4f}\n"
        ruta_txt = os.path.join(salida_dir, "metricas_historial.txt")
        with open(ruta_txt, 'a', encoding='utf-8') as f:
            f.write("\n" + titulo + bloque_r2 + "\n")
        print(f"  → Métrica R2 añadida al historial: {ruta_txt}")

    else:
        # --- Métricas ---
        print("\n[ MÉTRICAS DE CLASIFICACIÓN ]")
        metricas_texto = build_metrics_text(y_test, y_pred)
        print(metricas_texto)

        # --- Título con parámetros ---
        titulo = build_titulo(config, csv_id, nombre_modelo)

        # --- Exportar al .txt acumulativo ---
        exportar_metricas_txt(salida_dir, csv_id, method, titulo, metricas_texto)

    # 6. EXPORTAR PREDICCIONES DE AUDITORÍA (CSV, igual que antes)
    df_audit = pd.DataFrame({
        'Valor_Real': y_test,
        'Prediccion': y_pred
    })
    ruta_salida = os.path.join(salida_dir, f"audit_final_{method}.csv")
    df_audit.to_csv(ruta_salida, index=False)

    print(f"\nPROCESO TERMINADO. Predicciones guardadas en: {ruta_salida}")


if __name__ == "__main__":
    evaluar()