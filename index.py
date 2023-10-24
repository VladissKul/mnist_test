import os
import tempfile
import tensorflow as tf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, UploadFile, Path
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from core.monitoring.logger.custom_logger import CustomLogger
from keras_model.predict import Predict

with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

device = torch.device(config["app"]["device"])

CLASSIFY_MODEL = tf.keras.models.load_model(config["model"]["model_path"])

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

prediction_counter = Counter("predictions_total", "Число предсказаний")

Instrumentator().instrument(app).expose(app)

"""
os.environ['root'] = '/root/work'
os.environ['ServiceConfigFile'] = f"{os.environ['root']}/model/secrets/config.yml"
from preprocessor.train import Train
from preprocessor.predict import predict

predict_data = Predict()
train_data = Train()

@app.post("/predict")
def predict(image: UploadFile)
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(image.read())
        result = predict_model.run(temp_image.name)
    return {"prediction": prediction.tolist()}

        
@app.post("/train")
def train():
    train_model.run()
"""

@app.get("/")
async def index():
    return {"Message": ["Hello Dev"]}    
    
# @custom_logger.info
@app.post("/predict")
async def predict(image: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(await image.read())
        temp_image_path = temp_image.name

    predict = Predict(CLASSIFY_MODEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL, device)
    preprocessed_image = predict.preprocess_image(temp_image_path)
    prediction = predict.predict_digit(preprocessed_image)
    os.remove(temp_image_path)
    predict.save_data_predict(preprocessed_image, prediction, 'temp_plot.png')
    prediction_counter.inc()
    return {"prediction": prediction.tolist(), "plot_image_url": config["image"]["image_name"]}


# @custom_logger.info(predict)
@app.get("/predict_image/{image_path:path}")
async def predict_from_path(image_path: str = Path(..., description="Путь к изображению")):
    predict = Predict(CLASSIFY_MODEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL, device)
    preprocessed_image = predict.preprocess_image(image_path)
    prediction = predict.predict_digit(preprocessed_image)
    prediction_counter.inc()
    return {"prediction": prediction.tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host=config["app"]["host"], port=config["app"]["port"])
