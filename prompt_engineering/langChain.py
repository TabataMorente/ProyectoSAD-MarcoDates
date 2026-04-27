from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_classic.evaluation import ExactMatchStringEvaluator
from sklearn.model_selection import train_test_split
import sys
import json
import pandas as pd

class Log_generator:
    def __init__(self, path, debug=False):
        self.debug = debug
        self.path = path

    def get_debug(self):
        return self.debug

    def to_csv(self, output_dict):
        if self.debug:
            for clave in output_dict.keys():
                print(clave + ": " + str(len(output_dict[clave])))

        pd.DataFrame(output_dict).to_csv(self.path, index=False)

class Classification_log_generator(Log_generator):
    def __init__(self, path, debug=False):
        self.question = []
        self.examples = []
        self.id_examples = []
        self.actual_answer = []
        self.llm_answer = []
        self.task_name = []
        super().__init__(path, debug)

    def add_no_zero_shots_info(self, current_question, current_examples, current_question_indexes):
        self.question.append(current_question)
        self.examples.append(current_examples)
        self.id_examples.append(current_question_indexes)

        if super().get_debug():
            print(current_question, end="")

    def add_zero_shots_info(self, current_question):
        self.question.append(current_question)
        self.examples.append(None)
        self.id_examples.append(None)

        if super().get_debug():
            print(current_question)

    def add_answers(self, task_name, llm_answer, actual_answer):
        self.llm_answer.append(llm_answer)
        self.actual_answer.append(actual_answer)
        self.task_name.append(task_name)

    def to_dict(self):
        return {
            "Question" : self.question,
            "Examples": self.examples,
            "Id-Examples": self.id_examples,
            "Task name": self.task_name,
            "Actual-answer": self.actual_answer,
            "LLM-answer" : self.llm_answer
        }

    def to_csv(self):
        super().to_csv(self.to_dict())

class Oversampling_log_generator(Log_generator):
    def __init__(self, path, debug=False):
        super().__init__(path, debug)
        self.examples = []
        self.answers = []
        self.instructions = []
        self.models = []

    def add_model(self, model):
        self.models.append(model)
        if super().get_debug():
            print(model)

    def add_examples(self, example_string):
        self.examples.append(example_string)
        if super().get_debug():
            print(example_string)

    def add_instruction(self, instruction):
        self.instructions.append(instruction)
        if super().get_debug():
            print(instruction)

    def add_answer(self, answer):
        self.answers.append(answer)
        if super().get_debug():
            print(answer)

    def to_dict(self):
        return {
            "model": self.models,
            "instructions": self.instructions,
            "examples": self.examples,
            "answers": self.answers
        }

    def to_csv(self):
        super().to_csv(self.to_dict())

def create_examples(current_example_list, plantilla_no_zero_shots, task_string, log_generator):
    result = ""

    if len(current_example_list) > 0:
        examples_index = []

        for current_index, current_example in current_example_list.iterrows():
            formated_example = f"Comment:\"{current_example["content"]}\" Sentiment:{current_example["score"]}"
            result = result + formated_example + "\n"
            examples_index.append(current_index)

        formed_question = plantilla_no_zero_shots.format(examples=result, task=task_string)
        log_generator.add_no_zero_shots_info(formed_question, result, examples_index)

    return result

def create_chain(model, plantilla):
    current_prompt = PromptTemplate.from_template(plantilla)
    return ( current_prompt | model )

def evaluate(config, example_list_collection, tasks_collection):
    """
    config: Diccionario con las opciones del prompt engineering
    example_list_collection: Collecion de ejemplos ainadir en el prompt engineering
    plantilla: pregunta plantilla: No se si ainadir o no
    type:
    config: Dict
    example_list_collection: List[pd.DataFrame]
    plantilla: str
    """
    plantilla_zero_shots = """Respond ONLY with the sentiment label (\"negative\", \"neutral\" or \"positive\") and nothing else.
## TASK
{task}"""

    plantilla_no_zero_shots = """Respond ONLY with the sentiment label (\"negative\", \"neutral\" or \"positive\") for each task and nothing else.
## Examples
{examples}
## TASK
{task}"""

    model = OllamaLLM(
        model=config.get("model", "gemma2:2b"),
        temperature=config.get("temperature", 0),
        repeat_penalty=config.get("repeat_penalty", 0),
        num_predict=config.get("num_predict", 1),
        top_k=config.get("top_k", 10),
        top_p=config.get("top_p", 0.5)
    )

    chain_no_zero_shots = create_chain(model, plantilla_no_zero_shots)
    chain_zero_shots = create_chain(model, plantilla_zero_shots)

    evaluator = ExactMatchStringEvaluator()  # Por ahora se queda este evaluador
    ok = 0
    wrongOut = 0

    possible_answers = config.get("possible_answers",  ["positive", "neutral", "negative"] )

    log_generator = Classification_log_generator("classification.csv", True)

    for current_task in tasks_collection:
        task_string = "Comment:" + "\""+ current_task["content"] + "\" Sentiment: " + "\n"
        actual_answer = str(current_task["score"])

        for current_index, current_example_list in enumerate(example_list_collection):
            examples_text = create_examples(current_example_list, plantilla_no_zero_shots, task_string, log_generator)

            answer = None
            if len(examples_text) > 0:
                answer = chain_no_zero_shots.invoke({"examples": examples_text, "task":task_string}).strip()
            else:
                log_generator.add_zero_shots_info(plantilla_zero_shots.format(task=task_string))
                answer = chain_zero_shots.invoke({"task":task_string}).strip()

            log_generator.add_answers(task_string, answer, actual_answer)

            if not answer in possible_answers: wrongOut += 1

            score = evaluator.evaluate_strings(
                prediction=answer,
                reference=actual_answer
            )['score']

            print(answer)

            # evaluate_strings SOLO devuelve 1 o 0.
            if score == 1.0: ok += 1
            acc = round(100 * ok / (current_index + 1), 2)
            print("| " + model.model + "| row: " + str(current_index + 1) + " | acc: " + str(acc) + " | inc: " + str(wrongOut) + " |")

    log_generator.to_csv()

def oversample(config, examples_collection):
    plantilla_zero_shots = f"""Generate a comment with the sentiment label \"{config["score_to_extract"]}\" and nothing else.
Comment: """

    plantilla_no_zero_shots = "Generate a comment with the sentiment label \"" + config["score_to_extract"] + """"\", following the examples
below and nothing else.
## Examples
{examples}
Comment: """

    model = OllamaLLM(
        model=config.get("model", "gemma2:2b"),
        temperature=config.get("temperature", 0),
        repeat_penalty=config.get("repeat_penalty", 0),
        num_predict=config.get("num_predict", 1),
        top_k=config.get("top_k", 10),
        top_p=config.get("top_p", 0.5)
    )

    chain_no_zero_shots = create_chain(model, plantilla_no_zero_shots)
    chain_zero_shots = create_chain(model, plantilla_zero_shots)

    log_generator = Oversampling_log_generator("oversample.csv", True)

    for current_example_list_index, current_example_list in enumerate(examples_collection):
        example_string = ""
        for current_index, current_example in current_example_list.iterrows():
            example_string += "Comment:" + current_example["content"] + "\n"

        answer = None
        if len(example_string) > 0:
            log_generator.add_instruction(plantilla_no_zero_shots.format(examples=example_string))
            answer = chain_no_zero_shots.invoke({"examples": example_string}).strip()
        else:
            log_generator.add_instruction(plantilla_zero_shots)
            answer = chain_zero_shots.invoke({}).strip()

        log_generator.add_examples(example_string)
        log_generator.add_model(model.model)
        log_generator.add_answer(answer)

    log_generator.to_csv()

def get_parameters():
    result = None

    if len(sys.argv) > 1:
        result = sys.argv[1:]

    return result

def split_dataset(prompt_config, shot, dataframe):
    result = []
    split_collection = prompt_config.get("split")
    split_collection = [split for split in split_collection if split != "Manual"] # Lo limpiamos aqui para tener mejor log en el bucle
    seed = prompt_config.get("seed", 42)

    classes_length = len(prompt_config["possible_answers"])

    for split in split_collection:
        if split == "First":
            result.append(dataframe.iloc[:shot])
        elif split == "Random":
            subsample, _ = train_test_split(dataframe, train_size=shot, shuffle=True, random_state=seed)
            result.append(subsample)
        elif split == "Stratified" and shot >= classes_length:
            subsample, _ = train_test_split(dataframe, stratify=dataframe["score"], train_size=shot, shuffle=True, random_state=seed)
            result.append(subsample)
        else:
            print(f"El split {split} no es valido con el shot {shot}")

    return result

def extract_rows_manually(index_list, dataframe):
    result = []
    for current_manual_split in index_list:
        result.append(dataframe.iloc[current_manual_split])

    return result

def split_dataset_manually(prompt_config, dataframe):
    split = prompt_config.get("split")
    result = None
    if "Manual" in split:
        manual_split_indexes = prompt_config.get("manual_split_indexes")
        result = extract_rows_manually(manual_split_indexes, dataframe)

    return result

def split_dataset_by_shots(prompt_config, dataframe):
    shots = prompt_config.get("shots")

    result = []
    dataframe_length = len(dataframe)

    for current_shot in shots:
        if current_shot < 0:
            print("El shot no puede ser negativo")
        elif current_shot == 0:
            result.append(pd.DataFrame({})) # Se ainade vacio para poder identificar cuando ha sido ZERO shots
        elif current_shot <= dataframe_length:
            split_collection = split_dataset(prompt_config, current_shot, dataframe)
            for subsample in split_collection:
                result.append(subsample)
        else:
            print("El shot no puede ser mayor que la longitud del dataframe")

    manual_splits = split_dataset_manually(prompt_config, dataframe)
    if manual_splits != None:
        for current_split in manual_splits:
            result.append(current_split)

    return result

def get_test_collection(test_config, dataframe):
    return extract_rows_manually(test_config.get("questions_by_id"), dataframe)

def number_to_sentiment(value):
    result = ""
    if value == 1 or value == 2: result = "negative"
    elif value == 3: result = "neutral"
    elif value == 4 or value == 5: result = "positive"

    return result

def load_dataset(data_file):
    result = pd.read_csv(data_file)
    result.score = result.score.map(number_to_sentiment)
    return result

if __name__ == "__main__":
    """
    Parametro 1: ruta a config_file.json
    Parametro 2: ruta a el dataset
    """
    parameters = get_parameters()
    prompt_config = None
    data_file = "./Datos/Tinder.csv"

    if parameters == None:
        print("No hay parametros suficientes")
    elif len(parameters) >= 1:
        config_file = parameters[0]

        with open(config_file, "r") as file:
            config_dict = json.load(file)
            prompt_config = config_dict.get("prompt_engineering")

        if len(parameters) > 1:
            data_file = parameters[1]

    if prompt_config != None:
        mode = prompt_config.get("mode")
        if mode == "clasificacion":
            dataset = load_dataset(data_file)
            classification_config = prompt_config.get("clasificacion")
            examples_collection = split_dataset_by_shots(classification_config, dataset)
            tasks_collection = get_test_collection(classification_config.get("test_questions"), dataset)
            evaluate(classification_config, examples_collection, tasks_collection)
        elif mode == "oversampling":
            dataset = load_dataset(data_file)
            oversampling_config = prompt_config.get("oversampling")
            examples_collection = dataset.loc[lambda df : df["score"] == oversampling_config["score_to_extract"]]
            examples_collection = split_dataset_by_shots(oversampling_config, examples_collection)
            oversample(oversampling_config, examples_collection)
        else:
            print(f"El modo {mode} no es valido")
    else:
        print("No hay opciones para el prompt engineering")
