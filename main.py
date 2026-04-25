# -*- coding: utf-8 -*-
import sys
import subprocess
import os
import json


def main():
    # 1. Validación de argumentos (Ahora solo necesitamos 2: el CSV y el JSON)
    if len(sys.argv) < 3:
        print("🚀 Uso: python main.py <data.csv> <config_file.json>")
        sys.exit(1)

    csv_data = sys.argv[1]
    json_file = sys.argv[2]

    # Extraemos el nombre del dataset (ej: de 'ventas.csv' sacamos 'ventas')
    # El evaluador buscará el modelo en resultados_clasificacion/<dataset>/mejor_modelo/
    # y los datos en preprocesado/<dataset>/
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
        # train.py ahora recibe: el csv original y el json
        # Se encarga de dividir 70/15/15, preprocesar, entrenar y guardar el test listo
        print("\n--- 🛠️  FASE 1: ENTRENAMIENTO Y PREPROCESADO ---")
        res_train = subprocess.run(["python", "train.py", csv_data, json_file])

        if res_train.returncode != 0:
            print("❌ Error en la fase de entrenamiento. Abortando...")
            sys.exit(1)

        # --- FASE 2: EVALUACIÓN (AUDITORÍA) ---
        # eval.py ahora recibe: el json y el nombre del dataset
        # Se encarga de cargar el modelo 'MEJOR_...' y el CSV '_test_ready.csv'
        print("\n--- 📊 FASE 2: EVALUACIÓN FINAL (AUDITORÍA) ---")
        res_eval = subprocess.run(["python", "evaluar.py", json_file, dataset_name])

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