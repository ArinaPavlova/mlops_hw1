# import uvicorn
# import fastapi
import numpy as np
import pandas as pd
import logging
import pickle
import os

from pandas.core.frame import DataFrame
# from fastapi.responses import JSONResponse

from collections import defaultdict

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from typing import Dict, Union, Any

AVAILABLE_MODEL_LIST = {'regression': [Ridge, Lasso], 'classification': [LogisticRegression, GradientBoostingClassifier]}

class Models:
    def __init__(self):
        self.models = []
        self.fitted_models = []
        self.counter = 0
        self.ml_task = None
        self.available_models = defaultdict()

    #     self.models_file = 'models.pkl'
    #     self.fitted_models_file = 'fitted_models.pkl'
    #     self.counter = 0

    #     # Load models from file if available
    #     self.load_models()

    # def load_models(self):
    #     try:
    #         with open(self.models_file, 'rb') as file:
    #             self.models = pickle.load(file)
    #     except FileNotFoundError:
    #         self.models = []

    #     try:
    #         with open(self.fitted_models_file, 'rb') as file:
    #             self.fitted_models = pickle.load(file)
    #     except FileNotFoundError:
    #         self.fitted_models = []

    # def save_models(self):
    #     with open(self.models_file, 'wb') as file:
    #         pickle.dump(self.models, file)

    #     with open(self.fitted_models_file, 'wb') as file:
    #         pickle.dump(self.fitted_models, file)

    def available_model_list(self, task: str = '') -> str:
        """
        Получает на вход тип задачи и выводит список моделей доступных для ее решения 

        task: тип задачи
        """
        if task not in ['regression', 'classification']:
            logging.error(f"Invalid task type '{task}'. Available task types: 'regression', 'classification'")
            return "Invalid task type. Available task types: 'regression', 'classification'", 400  # Bad request
        self.ml_task = task
        self.available_models[self.ml_task] = {md.__name__: md for md in AVAILABLE_MODEL_LIST[self.ml_task]}
        to_print = [md.__name__ for md in AVAILABLE_MODEL_LIST[self.ml_task]]
        return f"ML task '{self.ml_task}':    Models: {to_print}", 200

    def get_model_by_id(self, model_id: int, fitted: bool = False) -> Dict:
        """
        Получает на вход id модели и возвращает ее

        model_id: id модели
        fitted: указывает, нужно ли получить подготовленную модель (True) или необученную модель (False).
        """
        models = self.fitted_models if fitted else self.models
        for model in models:
            if model['model_id'] == model_id:
                return model, 200
        logging.error(f"ML model {model_id} doesn't exist")
        return "ML model doesn't exist", 404  # Not found
    

    def create_model(self, model_name: str = '') -> Dict:
        """
        Получает на вход название модели и создает модель 

        model_name: название модели, которое выбирает пользователь

        return: {
            'model_id' - id модели
            'model_name' - название модели
            'ml_task' -  тип задачи
        }
        """
        self.counter += 1
        ml_model = {
            'model_id': self.counter,
            'model_name': None,
            'ml_task': self.ml_task,
        }

        fitted_model = {
            'model_id': self.counter,
        }

        if model_name in self.available_models[self.ml_task]:
            ml_model['model_name'] = model_name
        else:
            self.counter -= 1
            logging.error(f"Wrong model name {model_name}. Available models: {list(self.available_models[self.ml_task].keys())}")
            return "Wrong model name", 400  # Bad request
        
        self.models.append(ml_model)
        self.fitted_models.append(fitted_model)
        return ml_model, 201
    
    
    def update_model(self, model_dict: dict) -> None:
        """
        Получает на вход dict модели и обновляет его

        model_dict: dict модели
        """
        try:
            ml_model = self.get_model_by_id(model_dict['model_id'])
            ml_model.update(model_dict)
            return 200
        except (KeyError, TypeError):
            logging.error("Incorrect dictionary passed. Dictionary should be passed.")
            return 400  # Bad request

    def delete_model(self, model_id: int) -> None:
        """
        Получает на вход id и удаляет выбранную модель 

        model_id: id модели
        """
        try:
            model = self.get_model_by_id(model_id)[0]
            fitted_model = self.get_model_by_id(model_id, fitted=True)[0]
            self.fitted_models.remove(fitted_model)
            self.models.remove(model)
            return 200
        except ValueError:
            logging.error(f"ML model {model_id} doesn't exist")
            return 404  # Not found

    def fit(self, model_id, data_train, params) -> Dict:
        """
        Получает на вход id модели, данные для обучения и параметры, возвращает обученную модель

        model_id: id модели,
        data: данные (data_train и target)
        params: параметры для обучения
        """
        try:
            target = pd.DataFrame(data_train)[['target']]
            data_train = pd.DataFrame(data_train).drop(columns='target') 
        except Exception as e:
            logging.error(f"An error with input data: {e}")
            return "An error occurred with input data", 400  # Bad request
        
        try:
            model_dict = self.get_model_by_id(model_id)
            fitted_model = self.get_model_by_id(model_id, fitted=True)
        except ValueError:
            logging.error(f"ML model {model_id} doesn't exist")
            return "ML model doesn't exist", 404  # Not found
        
        try:
            ml_mod = self.available_models[self.ml_task][model_dict[0]['model_name']](**params)
        except TypeError:
            logging.error(f"Incorrect model parameters {params}.")
            return "Incorrect model parameters", 400  # Bad request
        
        try:
            ml_mod.fit(data_train, target)
        except Exception as e:
            logging.error(f"An error occurred during fitting: {e}")
            return "An error occurred during fitting", 500  # Internal server error
        
        try:
            fitted_model[0]['model'] = ml_mod
            return model_dict[0], 200
        except Exception as e:
            logging.error(f"Something wrong: {e}")
            return "Something wrong", 500  # Internal server error
        


    def predict(self, model_id, data_test) -> Union[DataFrame, Any]:
        """
        Получает на вход id модели и тестовую выборку, возвращает прогноз

        model_id: id модели,
        X: выборка для предсказания, без таргета
        """
        try:
            data_test = pd.DataFrame(data_test)
        except Exception as e:
            logging.error(f"An error with input data: {e}")
            return "An error occurred with input data", 400  # Bad request
        
        try:
            _ = self.get_model_by_id(model_id)
            fitted_model = self.get_model_by_id(model_id, fitted=True)
        except ValueError:
            logging.error(f"ML model {model_id} doesn't exist")
            return "ML model doesn't exist", 404  # Not found
        
        try:
            model = fitted_model[0]['model']
        except Exception as e:
                logging.error(f"Something wrong: {e}")
                return "Something wrong", 500  # Internal server error
        
        try:
            predict = model.predict(data_test)
            return pd.DataFrame(predict).to_dict(), 200
        except Exception as e:
                logging.error(f"An error occurred during prediction: {e}")
                return "An error occurred during prediction", 500  # Internal server error

