import uvicorn
from fastapi import FastAPI, Request, Response
from models import Models
# from pydantic import BaseModel

app = FastAPI()

models = Models()

# class ModelItem(BaseModel):
#     model_name: str

# class ModelUpd(BaseModel):
#     model_id : int
#     model_name: str
#     ml_task: str

# class Request(BaseModel):
#     data : dict
#     params : dict

# class ModelPredict(BaseModel):
#     data_test : dict


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
async def create_model(response: Response, request: Request):
    """
    Создает модель
    """
    model_name = (await request.json())['model_name']
    result, response.status_code = models.create_model(model_name=model_name)
    return result

@app.put("/api/update_model")
async def update_model(response: Response, request: Request):
    """
    Обновляет модели
    """
    model_name = (await request.json())['model_name']
    status_code = models.update_model(model_name=model_name)
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
async def fit(response: Response, model_id: int, request: Request):
    """
    Обучает модель
    """
    model_id = model_id
    data_train = (await request.json())['data_train']
    params = (await request.json())['params']
    result, response.status_code = models.fit(model_id, data_train, params)
    return result

@app.put("/api/predict/{model_id}")
async def predict(response: Response, model_id: int, request: Request):
    """
    Возвращает прогноз модели
    """
    data_test = (await request.json())['data_test']
    result, response.status_code = models.predict(model_id, data_test)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)