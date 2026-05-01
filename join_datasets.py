import sys
import pandas as pd
import os

def add_instances(dataset_file, new_instances_dir, result_dir):
    # pre: Se da  por hecho que la carpeta new_instances_dir solo contiene .csv
    result_dataset = pd.read_csv(dataset_file)
    instances_dir = os.listdir(new_instances_dir)

    os.makedirs(result_dir, exist_ok=True)

    for current_file_name in instances_dir:
        full_path = os.path.join(new_instances_dir, current_file_name)
        if os.path.isfile(full_path):
            current_dataset = pd.read_csv(full_path)
            result_dataset = pd.concat([result_dataset, current_dataset])

    print(result_dataset)

    result_file_name = os.path.join(result_dir, "new_Tinder.csv")

    result_dataset.to_csv(result_file_name, index=False)

def get_parameters():
    result = None

    if len(sys.argv) > 1:
        result = sys.argv[1:]

    return result

if __name__ == "__main__":
    """
    Parametro 1: carpeta con los archivos con las instancias generadas
    Parametro 2: Datos/Tinder.csv
    Parametro 3: nombre del archivo resultado
    """
    parameters = get_parameters()

    enough_parameters = True
    new_instances_dir = ""
    result_dataset_file = ""
    result_dataset_dir = ""

    if parameters == None:
        enough_parameters = False
    elif len(parameters) > 2:
        new_instances_dir = parameters[0]
        result_dataset_file = parameters[1]
        result_dataset_dir = parameters[2]
    else:
        enough_parameters = False

    if enough_parameters:
        add_instances(result_dataset_file, new_instances_dir, result_dataset_dir)
    else:
        print("No hay suficientes parametros")
