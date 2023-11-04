import uvicorn
from fastapi import FastAPI, Request
from models import Models

app = FastAPI()

models = Models()

@app.get("/api/model_list/{task}")
async def model_list(task: str):
    """
    Возвращает список моделей доступных для решения полученной задачи
    """
    return models.available_model_list(task)

@app.get("/api/get_models")
async def get_all_models():
    """
    Возвращает список всех моделей
    """
    return models.models

@app.get("/api/get_model_by_id/{model_id}")
def get_model_by_id_by_id(model_id: int):
    """
    Возвращает модель по id
    """
    return models.get_model_by_id(model_id)

@app.post("/api/create_model")
async def create_model(request: Request):
    """
    Создает модель
    """
    model_name = (await request.json())['model_name']
    models.create_model(model_name=model_name)
    return 'Модель создана'

@app.put("/api/update_model")
async def update_model(request: Request):
    """
    Обновляет модель
    """
    model_name = (await request.json())['model_name']
    models.update_model(model_name=model_name)
    return 'Модель обновлена'

@app.delete("/api/delete_model/{model_id}")
def delete_model(model_id: int):
    """
    Удаляет модель по id
    """
    models.delete_model(model_id)
    return 'Модель удалена'

@app.put("/api/fit/{model_id}")
async def fit(model_id: int, request: Request):
    """
    Обучает модель
    """
    models.fit(model_id, **await request.json())
    return 'Модель обучена'

@app.put("/api/predict/{model_id}")
async def predict(model_id: int, request: Request):
    """
    Возвращает прогноз модели
    """
    preds = models.predict(model_id, **await request.json())
    return preds

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)