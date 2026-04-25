from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_classic.evaluation import ExactMatchStringEvaluator
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
import sys
import json
import pandas

def evaluate(config, examples_collection):
    """
    config: Diccionario con las opciones del prompt engineering
    plantilla: pregunta plantilla: No se si ainadir o no
    type:
    config: Dict
    plantilla: str
    """
    plantilla = """Respond ONLY with the sentiment label (Worst, Bad, Neutral, Good or Best) and nothing else.
    ## TASK
    Comment:{comment} Sentiment:
    """
    prompt = PromptTemplate.from_template(plantilla)

    model = OllamaLLM(
        model=config.get("model", "gemma2:2b"),
        temperature=config.get("temperature", 0),
        repeat_penalty=config.get("repeat_penalty", 0),
        num_predict=config.get("num_predict", 1),
        top_k=config.get("top_k", 10),
        top_p=config.get("top_p", 0.5)
    )

    chain = prompt | model

    evaluator = ExactMatchStringEvaluator()  # Por ahora se queda este evaluador
    ok = 0
    wrongOut = 0

    possible_answers = config.get("possible_answers",  ["Worst", "Bad", "Neutral", "Good", "Best"] )

    for current_examples in examples_collection:
        print(current_examples)

    """
    for row_number, instance in dataset.iterrows():
        answer = chain.invoke({"comment" : instance["content"]}).strip()

        if not answer in possible_answers: wrongOut += 1

        score = evaluator.evaluate_strings(
            prediction=answer,
            reference=possible_answers[instance["score"] - 1]
        )['score']

        # evaluate_strings SOLO devuelve 1 o 0.
        if score == 1.0: ok += 1
        acc = round(100 * ok / (row_number + 1), 2)
        print("| " + model.model + "| row: " + str(row_number + 1) + " | acc: " + str(acc) + " | inc: " + str(wrongOut) + " |")
    """

def get_parameters():
    result = None

    if len(sys.argv) > 1:
        result = sys.argv[1:]

    return result

def split_dataset(prompt_config, shot, dataframe):
    result = None
    split_type = prompt_config.get("split_type")
    seed = prompt_config.get("seed", 42)

    if split_type == "First":
        result = dataframe.iloc[:shot]
    elif split_type == "Random":
        result, _ = train_test_split(dataframe, train_size=shot, shuffle=True, random_state=seed)
    elif split_type == "Stratified":
        result, _ = train_test_split(dataframe, stratify=dataframe["score"], train_size=shot, shuffle=True, random_state=seed)
    elif split_type == "Manual":
        manual_split_indexes = prompt_config.get("manual_split_indexes")
        result = dataframe.iloc[manual_split_indexes]
    else:
        print(f"El split {split_type} no es valido")

    return result

def split_dataset_by_shots(prompt_config, dataframe):
    shots = prompt_config.get("shots")

    result = []
    dataframe_length = len(dataframe)

    for current_shot in shots:
        if current_shot < 0:
            print("El shot no puede ser negativo")
        elif current_shot == 0:
            result.append(None)
        elif current_shot <= dataframe_length:
            subsample = split_dataset(prompt_config, current_shot, dataframe)
            result.append(subsample)
        else:
            print("El shot no puede ser mayor que la longitud del dataframe")

    return result

if __name__ == "__main__":
    parameters = get_parameters()
    prompt_config = None
    data_file = "./Datos/Tinder.csv"

    if parameters == None:
        print("No hay parametros suficientes")
    elif len(parameters) == 1:
        config_file = parameters[0]
        with open(config_file, "r") as file:
            config_dict = json.load(file)
            prompt_config = config_dict.get("prompt_engineering")
    else:
        data_file = parameters[1]

    if prompt_config != None:
        mode = prompt_config.get("mode")
        if mode == "sentiment_analysis":
            examples_collection = split_dataset_by_shots(prompt_config.get("sentiment_analysis"), pandas.read_csv(data_file))
            evaluate(prompt_config, examples_collection)
        elif mode == "nose":
            pass
        else:
            print(f"El modo {mode} no es valido")
    else:
        print("No hay opciones para el prompt engineering")
