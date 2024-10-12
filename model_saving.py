import keras
import json
import numpy as np
import pandas as pd
from keras.models import model_from_json


def save_json_model(input_model: any):
    """
    Saves the structure and weights of a machine learning model in JSON format.
    :param input_model: any - learning model to be saved. The model must have 'to_json()' and 'get_weights()' methods
    :return: None
    """
    model_json = input_model.to_json()
    with open('model_json.json', 'w') as json_file:
        json_file.write(model_json)
    weights = input_model.get_weights()
    weights_list = [weights_i.tolist() for weights_i in weights]
    with open('model_json_weights.json', 'w') as json_file:
        json.dump(weights_list, json_file)


def load_json_model(weights_path: str, arh_path: str):
    """
    Loads a machine learning model from JSON files containing its architecture and weights.

    :param weights_path: str - The file path to the JSON file containing the model's weights.
    :param arh_path: str - The file path to the JSON file containing the model's architecture.
    :return: keras.models.Model
    """
    with open(arh_path, 'r') as arh_file:
        arh = arh_file.read()
    with open(weights_path, 'r') as weights_file:
        weights_list = json.load(weights_file)
    weights = [np.array(weights_i) for weights_i in weights_list]
    return_model = model_from_json(arh)
    return_model.set_weights(weights)
    return return_model


if __name__ == '__main__':
    print(f'{"Начало программы":-^100}')
    model_path = r'D:\Work_Aubakirov\ML\Apha_version\PIT\model\Model_5'
    model = keras.models.load_model(model_path)
    print(model.summary())
    save_json_model(input_model=model)
    model = load_json_model(arh_path='model_json.json', weights_path='model_json_weights.json')
    print(model.summary())
