import tensorflow as tf
import os
import json
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum

DURATION_EACH_DATAPOINT = 1

class Habitat(Enum):
    AeCx = 1
    AnCx = 2
    
@dataclass
class ModelSetting:
    model_setting_path:str
    model_name: str
    habitat_setting: Habitat
    threshold: dict[str, float]

def load_model(model_setting: ModelSetting) -> tf.keras.Model:
    model_path = os.path.join(model_setting.model_setting_path, model_setting.model_name)
    model: tf.keras.Model = tf.keras.models.load_model(model_path, compile=False)
    
    # change the input layer to make sure that it does not specify batch size
    model_config = model.get_config()
    input_layer_name = model_config['layers'][0]['name']
    model_config['layers'][0] = {
        'name': input_layer_name,
        'class_name': 'InputLayer',
        'config': {
            'batch_input_shape': (None, 10, 8000, 1),
            'dtype': 'float32',
            'sparse': False,
            'name': input_layer_name
        },
        'inbound_nodes': []
    }
    new_model = model.__class__.from_config(model_config)
    weights = [layer.get_weights() for layer in model.layers[1:]]
    for layer, weight in zip(new_model.layers[1:], weights):
        layer.set_weights(weight)
    return new_model

def load_thres(model_setting: ModelSetting) -> npt.NDArray[np.floating]:
    class_labels = load_class_label(model_setting)
    thres = [0.]*len(class_labels)
    for l, v in class_labels.items():
        try:
            thres[v] = model_setting.threshold[l]
        except KeyError as e:
            print("checkout the settings.json, the keys in threshold might have difference name from the class labels")
            raise e
    return np.array(thres)

def load_class_label(model_setting: ModelSetting) -> dict[str, int]:
    if model_setting.habitat_setting == Habitat.AeCx:
        class_labels: dict[str, int] = {
                'Ae.Aegypti.M': 0,
                'Ae.Albopictus.M': 1,
                'Cx.Quin.M': 2,
                'Ae.Aegypti.F': 3,
                'Ae.Albopictus.F': 4,
                'Cx.Quin.F': 5
        }
    elif model_setting.habitat_setting == Habitat.AnCx:
        class_labels = {
                'An.Dirus.M': 0,
                'An.Minimus.M': 1,
                'Cx.Quin.M': 2,
                'An.Dirus.F': 3,
                'An.Minimus.F': 4,
                'Cx.Quin.F': 5,
        }
    else:
        assert False
    return class_labels


def load_model_setting(model_setting_path:str) -> ModelSetting:
    with open(os.path.join(model_setting_path, "settings.json"), ) as f:
        model_setting = json.load(f)
    assert "model_name" in model_setting and "habitat_setting" in model_setting and "thresholds" in model_setting
    if model_setting["habitat_setting"] == "AeCx":
        hab = Habitat.AeCx
    if model_setting["habitat_setting"] == "AnCx":
        hab = Habitat.AnCx
    return ModelSetting(model_setting_path, model_setting["model_name"], hab, model_setting["thresholds"])

def model_predict(model:tf.keras.Model, x:npt.NDArray) -> npt.NDArray:
    y_pred = model.predict(x, verbose='0')
    return tf.sigmoid(y_pred).numpy()