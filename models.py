import uvicorn
import fastapi
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from fastapi.responses import JSONResponse

from collections import defaultdict

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from typing import Dict, Union, Any

available_list = {'regression': [Ridge, Lasso], 'classification': [LogisticRegression, GradientBoostingClassifier]}

class Models:
    def __init__(self):
        self.models = []
        self.fitted_models = []
        self.counter = 0
        self.ml_task = None
        self.available_models = defaultdict()
        

    def available_model_list(self, task: str = '') -> str:
        """
        Получает на вход тип задачи и выводит список моделей доступных для ее решения 

        task: тип задачи
        """
        self.ml_task = task
        self.available_models[self.ml_task] = {md.__name__: md for md in available_list[self.ml_task]}
        to_print = [md.__name__ for md in available_list[self.ml_task]]
        return f"ML task '{self.ml_task}':    Models: {to_print}"

    def get_model_by_id(self, model_id: int, fitted: bool = False) -> Dict:
        """
        Получает на вход id модели и возвращает ее

        model_id: id модели
        fitted: указывает, нужно ли получить подготовленную модель (True) или необученную модель (False).
        """
        models = self.fitted_models if fitted else self.models
        for model in models:
            if model['model_id'] == model_id:
                return model
        JSONResponse(f"ML model {model_id} doesn't exist")

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
            JSONResponse(f"Wrong model name {model_name}. Available models: {list(self.available_models[self.ml_task].keys())}")   

        self.models.append(ml_model)
        self.fitted_models.append(fitted_model)
        return ml_model
    
    def update_model(self, model_dict: dict) -> None:
        """
        Получает на вход id и обновляет выбранную модель 

        model_id: id модели
        """
        try:
            ml_model = self.get_model_by_id(model_dict['model_id'])
            ml_model.update(model_dict)
        except (KeyError, TypeError):
            JSONResponse("Incorrect dictionary passed. Dictionary should be passed.")

    def delete_model(self, model_id: int) -> None:
        """
        Получает на вход id и удаляет выбранную модель 

        model_id: id модели
        """
        model = self.get_model_by_id(model_id)
        fitted_model = self.get_model_by_id(model_id, fitted=True)
        self.fitted_models.remove(fitted_model)
        self.models.remove(model)

    def fit(self, model_id, data, params, **kwargs) -> Dict:
        """
        Получает на вход id модели, данные для обучения и параметры, возвращает обученную модель

        model_id: id модели,
        data: данные (X_train и target)
        params: параметры для обучения
        """
        X = pd.DataFrame(data).drop(columns='target') 
        y = pd.DataFrame(data)[['target']]
        model_dict = self.get_model_by_id(model_id)
        fitted_model = self.get_model_by_id(model_id, fitted=True)
        ml_mod = self.available_models[self.ml_task][model_dict['model_name']](**params)
        ml_mod.fit(X, y)
        fitted_model['model'] = ml_mod
        return model_dict

    def predict(self, model_id, X, to_dict: bool = True, **kwargs) -> Union[DataFrame, Any]:
        """
        Получает на вход id модели и тестовую выборку, возвращает прогноз

        model_id: id модели,
        X: выборка для предсказания, без таргета
        """
        X = pd.DataFrame(X)
        _ = self.get_model_by_id(model_id)
        fitted_model = self.get_model_by_id(model_id, fitted=True)
        model = fitted_model['model']
        predict = model.predict(X)
        if to_dict:
            return pd.DataFrame(predict).to_dict()
        return predict