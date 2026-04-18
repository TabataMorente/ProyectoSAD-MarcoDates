# -*- coding: utf-8 -*-
import sys
import subprocess
import os

def main():
    # 1. Validación de argumentos (Ahora solo necesitamos 2: el CSV y el JSON)
    if len(sys.argv) < 3:
        print("🚀 Uso: python main.py <data.csv> <config_file.json>")
        sys.exit(1)

    csv_data = sys.argv[1]
    json_file = sys.argv[2]

    # Extraemos el nombre del dataset (ej: de 'ventas.csv' sacamos 'ventas')
    # Lo necesitamos porque el evaluador busca en carpetas con ese nombre
    dataset_name = os.path.basename(csv_data).split('.')[0]

    print("\n" + "═"*60)
    print(f" 🏁 INICIANDO PIPELINE COMPLETO: {dataset_name.upper()}")
    print("═"*60)

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

    print("\n" + "═"*60)
    print(" ✅ PIPELINE FINALIZADO CON ÉXITO")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()