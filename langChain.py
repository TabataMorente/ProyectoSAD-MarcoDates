from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.model_selection import train_test_split
import sys
import json
import pandas as pd
import os
import evaluar

class Log_generator:
    def __init__(self, path, debug=False):
        self.debug = debug
        self.path = path
        os.makedirs(path, exist_ok=True)

    def get_debug(self):
        return self.debug

    def clean(self):
        pass

    def to_csv(self, output_dict, file_name):

        export_path = os.path.join(self.path, file_name)

        if self.debug:
            for clave in output_dict.keys():
                print(clave + ": " + str(len(output_dict[clave])))

        pd.DataFrame(output_dict).to_csv(export_path, index=False)

class Classification_log_generator(Log_generator):
    def __init__(self, path, debug=False):
        self.question = []
        self.examples = []
        self.id_examples = []
        self.actual_answer = []
        self.llm_answer = []
        self.task_name = []
        self.task_id = []
        self.models = []
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

    def add_models(self, model_name):
        self.models.append(model_name)

    def add_answers(self, task_name, llm_answer, actual_answer, task_id):
        self.llm_answer.append(llm_answer)
        self.actual_answer.append(actual_answer)
        self.task_name.append(task_name)
        self.task_id.append(task_id)

    def to_dict(self):
        return {
            "Model": self.models,
            "Question" : self.question,
            "Examples": self.examples,
            "Id-Examples": self.id_examples,
            "Task name": self.task_name,
            "Actual-answer": self.actual_answer,
            "LLM-answer" : self.llm_answer,
            "Task-id": self.task_id
        }

    def clean(self):
        self.question = []
        self.examples = []
        self.id_examples = []
        self.actual_answer = []
        self.llm_answer = []
        self.task_name = []
        self.task_id = []
        self.models = []

    def print_evaluation(self):
        evaluar.print_advanced_metrics(self.actual_answer, self.llm_answer)

    def to_csv(self, file_name):
        super().to_csv(self.to_dict(), file_name)

class Dataset_like_log_generator(Log_generator):
    def __init__(self, path, debug=False):
        super().__init__(path, debug)
        self.dict = {
            "content" : [],
            "score": [],
            "gender": [],
            "location": [],
            "date": []
        }

    def add(self, answer):
        my_keys = self.dict.keys()

        print(answer)

        for current_dict in answer:
            for current_key in my_keys:
                print(current_key + ":" + str(current_dict.get(current_key, None)))
                self.dict[current_key].append(current_dict.get(current_key, None))

    def to_csv(self, file_name):
        super().to_csv(self.dict, file_name)

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

    def clean(self):
        self.models = []
        self.examples = []
        self.answers = []
        self.instructions = []

    def to_dict(self):
        return {
            "model": self.models,
            "instructions": self.instructions,
            "examples": self.examples,
            "answers": self.answers
        }

    def to_csv(self, file_name):
        super().to_csv(self.to_dict(), file_name)

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

def sort_by_length(matrix):
    matrix_length = len(matrix)
    array_index = 0

    while array_index < matrix_length:
        smallest = array_index
        comparison_index = array_index

        while comparison_index < matrix_length:
            if len(matrix[comparison_index]) < len(matrix[smallest]):
                smallest = comparison_index

            comparison_index += 1

        aux = matrix[array_index]
        matrix[array_index] = matrix[smallest]
        matrix[smallest] = aux

        array_index += 1

def parse_answer_to_df(text):
    result = []
    line_list = text.split("\n")

    for key_value_list in line_list:
        current_dict = {}
        split_key_value_list = key_value_list.split(";")
        for key_value_pair in split_key_value_list:
            split_key_value = key_value_pair.split(":")

            if len(split_key_value) >= 2:
                current_dict[split_key_value[0].lower()] = split_key_value[1]

        result.append(current_dict)

    return result

def add_key(dict_list, new_key, new_value):
    for current_dict in dict_list:
        current_dict[new_key] = new_value

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

    log_generator = Classification_log_generator("classification_results", True)

    sort_by_length(example_list_collection)

    previous_shot = -1

    for current_index, current_example_list in enumerate(example_list_collection):
        shot = len(current_example_list)

        if previous_shot == -1:
            previous_shot = shot
        elif previous_shot != shot:
            log_generator.print_evaluation()
            log_generator.to_csv(str(previous_shot) + " shots.csv")
            log_generator.clean()
            previous_shot = shot

        for task_index, current_task in tasks_collection.iterrows():
            task_string = "Comment:" + "\"" + current_task["content"] + "\" Sentiment: " + "\n"
            actual_answer = str(current_task["score"])
            examples_text = create_examples(current_example_list, plantilla_no_zero_shots, task_string, log_generator)

            answer = None
            if len(examples_text) > 0:
                answer = chain_no_zero_shots.invoke({"examples": examples_text, "task": task_string}).strip().lower()
            else:
                log_generator.add_zero_shots_info(plantilla_zero_shots.format(task=task_string))
                answer = chain_zero_shots.invoke({"task": task_string}).strip().lower()

            log_generator.add_models(model.model)
            log_generator.add_answers(task_string, answer, actual_answer, task_index)

    print(previous_shot)
    log_generator.print_evaluation()
    log_generator.to_csv(str(previous_shot) + " shots.csv")
    log_generator.clean()

def oversample(config, examples_collection):
    for current_sentiment in config["score_to_extract"]:
        plantilla_zero_shots = f"""Generate a comment, including the location the comment was sent, the date and the user's gender, with the sentiment label \"{current_sentiment}\" and nothing else.
    Follow the next format:
    content:\"\";gender:male/female;location;date:;"""

        plantilla_no_zero_shots = "Generate a comment, including the user's gender, the location the comment was sent and the date, with the sentiment label \"" + str(current_sentiment) + """"\", following the examples
    below and nothing else.
    ## Examples
    {examples}
    content:\"\";gender:;location;date:;"""

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

        log_generator = Oversampling_log_generator("oversample_results", False)
        log_generator2 = Dataset_like_log_generator("oversample", True)

        previous_shot = -1

        sort_by_length(examples_collection)

        for current_example_list_index, current_example_list in enumerate(examples_collection):
            shot = len(current_example_list)

            if previous_shot == -1:
                previous_shot = shot
            elif previous_shot != shot:
                log_generator.to_csv(str(previous_shot) + " shots.csv")
                log_generator.clean()
                previous_shot = shot

            example_string = ""
            for current_index, current_example in current_example_list.iterrows():
                example_string += "Content:\"" + current_example["content"] + "\";Gender:" + current_example["gender"] \
                + ";Location:" + current_example["location"] + ";Date:" + current_example["date"] + ";\n"

            answer = None
            if len(example_string) > 0:
                log_generator.add_instruction(plantilla_no_zero_shots.format(examples=example_string))
                answer = chain_no_zero_shots.invoke({"examples": example_string}).strip()

            else:
                log_generator.add_instruction(plantilla_zero_shots)
                answer = chain_zero_shots.invoke({}).strip()


            parsed_answer = parse_answer_to_df(answer)
            add_key(parsed_answer, "score", current_sentiment)

            log_generator2.add(parsed_answer)

            log_generator.add_examples(example_string)
            log_generator.add_model(model.model)
            log_generator.add_answer(answer)

        log_generator.to_csv(str(previous_shot) + " shots.csv")
        log_generator.clean()

    log_generator2.to_csv("Tinder_oversample.csv")

def get_parameters():
    result = None

    if len(sys.argv) > 1:
        result = sys.argv[1:]

    return result

def split_dataset_given_split(prompt_config, split, split_size, dataframe, classes_length):
    result = None
    seed = prompt_config.get("seed", 42)
    if split == "First":
        result = dataframe.iloc[:split_size]
    elif split == "Random":
        result, _ = train_test_split(dataframe, train_size=split_size, shuffle=True, random_state=seed)
    elif split == "Stratified" and split_size >= classes_length:
        result, _ = train_test_split(dataframe, stratify=dataframe["score"], train_size=split_size, shuffle=True, random_state=seed)
    else:
        print(f"El split {split} no es valido con el split_size {split_size}")

    return result

def split_dataset(prompt_config, shot, dataframe):
    result = []
    split_collection = prompt_config.get("split")
    split_collection = [split for split in split_collection if split != "Manual"] # Lo limpiamos aqui para tener mejor log en el bucle

    classes_length = len(prompt_config["possible_answers"])

    for split in split_collection:
        subsample = split_dataset_given_split(prompt_config, split, shot, dataframe, classes_length)

        if subsample is not None:
            result.append(subsample)
        else:
            print(f"El split {split} no es valido con el shot {shot}")

    return result

def extract_rows_manually(index_list, dataframe):
    new_index_list = []

    for current_index in index_list:
        if type(current_index) != int:
            index_range = range(current_index[0], current_index[1])

            for current_index_in_range in index_range:
                new_index_list.append(current_index_in_range)
        else:
            new_index_list.append(current_index)

    print(new_index_list)

    result = dataframe.iloc[new_index_list].sample(frac=1)

    return result

def split_dataset_manually(prompt_config, dataframe):
    split = prompt_config.get("split")
    result = None
    if "Manual" in split:
        manual_split_indexes = prompt_config.get("manual_split_indexes")
        result = []

        for current_split in manual_split_indexes:
            result.append(extract_rows_manually(current_split, dataframe))

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

def get_test_collection(test_config, dataframe, classes_length):
    split_type = test_config.get("split")
    split_size = test_config.get("split_size")

    if split_type == "Manual":
        index_list = test_config.get("manual_split_indexes")
        result = extract_rows_manually(index_list, dataframe)
    else:
        result = split_dataset_given_split(test_config, split_type, split_size, dataframe, classes_length)

    return result

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
        data_file = parameters[0]

        if len(parameters) > 1:
            config_file = parameters[1]

            with open(config_file, "r") as file:
                config_dict = json.load(file)
                prompt_config = config_dict.get("prompt_engineering")

    if prompt_config != None:
        mode = prompt_config.get("mode")
        if mode == "clasificacion":
            dataset = load_dataset(data_file)
            classification_config = prompt_config.get("clasificacion")
            examples_collection = split_dataset_by_shots(classification_config, dataset)
            tasks_collection = get_test_collection(classification_config.get("test_questions"), dataset, len(classification_config.get("possible_answers")))
            evaluate(classification_config, examples_collection, tasks_collection)
        elif mode == "oversampling":
            dataset = pd.read_csv(data_file)
            oversampling_config = prompt_config.get("oversampling")
            examples_collection = dataset.loc[lambda df : df["score"].isin(oversampling_config["score_to_extract"])]
            examples_collection = split_dataset_by_shots(oversampling_config, examples_collection)

            oversample(oversampling_config, examples_collection)
        else:
            print(f"El modo {mode} no es valido")
    else:
        print("No hay opciones para el prompt engineering")
