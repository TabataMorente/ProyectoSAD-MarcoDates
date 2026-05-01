# -*- coding: utf-8 -*-
import sys
import subprocess
import os
import json


def main():
    # 1. Validación de argumentos
    if len(sys.argv) < 3:
        print("🚀 Uso: python main.py <data.csv> <config_file.json>")
        sys.exit(1)

    csv_data = sys.argv[1]
    json_file = sys.argv[2]

    dataset_name = os.path.basename(csv_data).split('.')[0]

    # 2. Leer el JSON para saber qué tarea ejecutar
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error al leer el archivo JSON: {e}")
        sys.exit(1)

    task = config.get("task", "classification").lower()

    print("\n" + "═" * 60)
    print(f" 🏁 INICIANDO PIPELINE COMPLETO: {dataset_name.upper()}")
    print(f" 🎯 TAREA SELECCIONADA: {task.upper()}")
    print("═" * 60)

    if task == "classification":
        # --- FASE 1: ENTRENAMIENTO ---
        print("\n--- 🛠️  FASE 1: ENTRENAMIENTO Y PREPROCESADO ---")
        res_train = subprocess.run([r"python", "train.py", csv_data, json_file])

        if res_train.returncode != 0:
            print("❌ Error en la fase de entrenamiento. Abortando...")
            sys.exit(1)

        # --- FASE 2: EVALUACIÓN (AUDITORÍA) ---
        # Leemos la ruta exacta del mejor modelo que train.py escribió en disco.
        # Así evaluar.py evalúa siempre el modelo de ESTA ejecución, no uno antiguo.
        ruta_mejor_txt = os.path.join(
            "resultados_clasificacion", dataset_name, "mejor_modelo", "ultimo_mejor_modelo.txt"
        )

        if not os.path.exists(ruta_mejor_txt):
            print(f"❌ No se encontró el archivo de ruta del mejor modelo: {ruta_mejor_txt}")
            sys.exit(1)

        with open(ruta_mejor_txt, 'r', encoding='utf-8') as f:
            ruta_mejor_modelo = f.read().strip()

        print(f"\n📌 Mejor modelo de esta ejecución: {ruta_mejor_modelo}")

        print("\n--- 📊 FASE 2: EVALUACIÓN FINAL (AUDITORÍA) ---")
        # Pasamos la ruta exacta del .pkl en lugar del dataset_name genérico,
        # para que evaluar.py no tenga que buscar en disco por su cuenta.
        res_eval = subprocess.run(
            [r"python", "evaluar.py", json_file, dataset_name, ruta_mejor_modelo]
        )

        if res_eval.returncode != 0:
            print("❌ Error en la fase de evaluación.")
            sys.exit(1)

    elif task == "clustering":
        # --- FASE ÚNICA: CLUSTERING / TOPIC MODELING ---
        print("\n--- 🔍 FASE: CLUSTERING (LDA) ---")
        res_clust = subprocess.run(["python", "clustering.py", csv_data, json_file])

        if res_clust.returncode != 0:
            print("❌ Error en la fase de clustering. Abortando...")
            sys.exit(1)

    else:
        print(f"❌ Tarea no reconocida en el JSON: '{task}'. Usa 'classification' o 'clustering'.")
        sys.exit(1)

    print("\n" + "═" * 60)
    print(" ✅ PIPELINE FINALIZADO CON ÉXITO")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()