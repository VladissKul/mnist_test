import os
import sys
import tempfile
import torch
import uvicorn
# import yaml
from fastapi import FastAPI, UploadFile, Path
from fastapi.middleware.cors import CORSMiddleware

# from prometheus_client import Counter
# from prometheus_fastapi_instrumentator import Instrumentator


os.environ['root'] = '/root/work/projects/mnist_service_test'
os.environ['ServiceConfigFile'] = f"{os.environ['root']}/model/secrets/config.yml"
sys.path.append(os.environ['root'])
from preprocessor.train import Train
from preprocessor.predict import Predict
import nest_asyncio

nest_asyncio.apply()
# from core.monitoring.logger.custom_logger import CustomLogger


# with open("config.yml", "r") as config_file:
#   config = yaml.safe_load(config_file)

# device = torch.device(config["app"]["device"])

# CLASSIFY_MODEL = tf.keras.models.load_model(config["model"]["model_path"])

# IMAGE_WIDTH = config["image"]["width"]
# IMAGE_HEIGHT = config["image"]["height"]
# IMAGE_CHANNEL = config["image"]["channel"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# custom_logger = CustomLogger()

# prediction_counter = Counter("predictions_total", "Число предсказаний")

# Instrumentator().instrument(app).expose(app)


predict_data = Predict()
train_data = Train()


@app.post("/predict")
def predict(image: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(image.read())
        result = predict_model.run(temp_image.name)
        # prediction_counter.inc()
    return {"prediction": prediction.tolist()}


@app.post("/train")
def train():
    train_model.run()


@app.get("/")
async def index():
    return {"Message": ["Hello Dev"]}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=1481)