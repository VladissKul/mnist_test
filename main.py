import os
import tempfile

import cv2
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, UploadFile, Path
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
from pydantic import BaseModel

from logger.custom_logger import CustomLogger
from model.classify_model import MNIST_Classify_Model, DataPreprocessing

with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

device = torch.device(config["app"]["device"])
SAVED_MODEL_PATH = config["model"]["model_path"]

CLASSIFY_MODEL = MNIST_Classify_Model().to(device)
CLASSIFY_MODEL.load_state_dict(torch.load(SAVED_MODEL_PATH))

IMAGE_WIDTH = config["image"]["width"]
IMAGE_HEIGHT = config["image"]["height"]
IMAGE_CHANNEL = config["image"]["channel"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_logger = CustomLogger()


def preprocess_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    # image = image.tobytes()
    return image


# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     if image is None or image.size == 0:
#         return None
#
#     image = cv2.resize(image, (28, 28))
#
#     image = 255 - image
#
#     threshold = 128  # Задаем порог бинаризации
#     image = (image > threshold).astype(np.uint8)
#
#     image = image.astype(np.float32) / 255.0
#
#     image = image.reshape(784)
#
#     return image


class RequestInput(BaseModel):
    input: str


@app.get("/")
async def index():
    return {"Message": ["Hello Dev"]}


# @custom_logger.info
@app.post("/predict")
async def predict(image: UploadFile):
    try:
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

        custom_logger.info(f"Prediction: {prediction.tolist()}")
        custom_logger.info(f"Plot image URL: {config['image']['image_name']}")

        return {"prediction": prediction.tolist(), "plot_image_url": config["image"]["image_name"]}
    except Exception as e:
        custom_logger.error(f"An error occurred: {str(e)}")
        return {"error": str(e)}


@custom_logger.info(predict)
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
    uvicorn.run(app, host=config["app"]["host"], port=config["app"]["port"])
