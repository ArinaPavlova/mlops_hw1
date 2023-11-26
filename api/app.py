import uvicorn
from fastapi import FastAPI, status, Response
from models import Models
from pydantic import BaseModel

app = FastAPI()

models = Models()

class ModelItem(BaseModel):
    model_name: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_name": "LogisticRegression"
                }
            ]
        }
    }

class ModelUpd(BaseModel):
    model_name: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                "model_name": 
                    {
                        'model_id': 1,
                        'model_name': 'LogisticRegression',
                        'ml_task': 'classification'
                    }
                }
            ]
        }
    }

class ModelFit(BaseModel):
    data_train : dict
    params : dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'data_train': 
                    {
                        'Name': 
                        {
                            0: 'Tom', 
                            1: 'Joseph', 
                            2: 'Krish', 
                            3: 'John'
                        },
                        'Age': 
                        {
                            0: 20, 
                            1: 21, 
                            2: 19, 
                            3: 18
                        }
                    },
                    'params': 
                    {
                        'random_state': 32
                    }
                }
            ]
        }
    }

class ModelPredict(BaseModel):
    data_test : dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'data_test': 
                    {
                        'Name': 
                        {
                            0: 'Marta', 
                            1: 'Den'
                        },
                        'Age': 
                        {
                            0: 22, 
                            1: 17
                        }
                    }
                }
            ]
        }
    }


@app.get("/api/model_list/{task}")
async def model_list(response: Response, task: str):
    """
    Возвращает список моделей доступных для решения полученной задачи
    """
    result, response.status_code = models.available_model_list(task)
    return result

@app.get("/api/get_models")
async def get_all_models():
    """
    Возвращает список всех моделей
    """
    return models.models

@app.get("/api/get_model_by_id/{model_id}")
def get_model_by_id(response: Response, model_id: int):
    """
    Возвращает модель по id
    """
    result, response.status_code = models.get_model_by_id(model_id)
    return result

@app.post("/api/create_model")
async def create_model(response: Response, request: ModelItem):
    """
    Создает модель
    """
    result, response.status_code = models.create_model(model_name=request.model_name)
    return result

@app.put("/api/update_model")
async def update_model(response: Response, request: ModelUpd, status_code=status.HTTP_200_OK):
    """
    Обновляет модели
    """
    status_code = models.update_model(model_name=request.model_name)
    if (status_code != 200):
        response.status_code = status_code

@app.delete("/api/delete_model/{model_id}")
def delete_model(response: Response, model_id: int):
    """
    Удаляет модель по id
    """
    status_code = models.delete_model(model_id)
    if (status_code != 200):
        response.status_code = status_code

@app.put("/api/fit/{model_id}")
async def fit(response: Response, model_id: int, request: ModelFit):
    """
    Обучает модель
    """
    model_id = model_id
    result, response.status_code = models.fit(model_id, request.data_train, request.params)
    return result

@app.put("/api/predict/{model_id}")
async def predict(response: Response, model_id: int, request: ModelPredict):
    """
    Возвращает прогноз модели
    """
    result, response.status_code = models.predict(model_id, request.data_test)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)