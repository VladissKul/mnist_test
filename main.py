import os
import tempfile

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, Path
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
from pydantic import BaseModel

from model.classify_model import MNIST_Classify_Model, DataPreprocessing

device = torch.device('cpu')
SAVED_MODEL_PATH = "./model/model.pth"

CLASSIFY_MODEL = MNIST_Classify_Model().to(device)
CLASSIFY_MODEL.load_state_dict(torch.load(SAVED_MODEL_PATH))

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 28, 28, 1

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.tobytes()
    return image

# def preprocess_image(image_path):
#     # Попытка открыть изображение с помощью OpenCV
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     if image is None or image.size == 0:
#         # Если изображение не было успешно открыто или оно пусто, вернуть None
#         return None
#
#     # Изменяем размер изображения на 28x28 пикселей (стандартный размер MNIST)
#     image = cv2.resize(image, (28, 28))
#
#     # Инвертируем цвета (черное на белом вместо белого на черном)
#     image = 255 - image
#
#     # Бинаризуем изображение, преобразуя все пиксели в черный (0) или белый (1)
#     threshold = 128  # Задаем порог бинаризации
#     image = (image > threshold).astype(np.uint8)
#
#     # Нормализуем значения пикселей к диапазону [0, 1]
#     image = image.astype(np.float32) / 255.0
#
#     # Разворачиваем изображение в одномерный массив размером 28x28=784
#     image = image.reshape(784)
#
#     return image


class RequestInput(BaseModel):
    input: str


@app.get("/")
async def index():
    return {"Message": ["Hello Dev"]}


@app.post("/predict")
async def predict(image: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(await image.read())
        temp_image_path = temp_image.name

    preprocessed_image = preprocess_image(temp_image_path)
    request_input = DataPreprocessing(
        target_datatype=np.float32,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(preprocessed_image)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    os.remove(temp_image_path)

    preprocessed_image_array = np.frombuffer(preprocessed_image, dtype=np.uint8)
    preprocessed_image_array = preprocessed_image_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    plt.imshow(preprocessed_image_array, cmap='gray')
    plt.title(f'Predicted Digit: {prediction[0]}')
    plt.axis('off')

    plt.savefig('temp_plot.png')

    return {"prediction": prediction.tolist(), "plot_image_url": "temp_plot.png"}


@app.get("/predict_image/{image_path:path}")
async def predict_from_path(image_path: str = Path(..., description="Путь к изображению")):
    preprocessed_image = preprocess_image(image_path)

    request_input = DataPreprocessing(
        target_datatype=np.float32,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(preprocessed_image)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    return {"prediction": prediction.tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=1490)
