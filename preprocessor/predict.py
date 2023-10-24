import os
import pickle
import sys


class Predict:

    def __init__(self):
        self.load_config()

    def load_config(self):
        pass

    def load_model(self):
        sys.path.append(f"""{os.environ['root']}/model""")
        from model import Model
        with open(f"""{os.environ['root']}/model/artefacts/model.pkl""", 'rb') as f:
            self.model = pickle.load(f)

    def model_predict(self, raw_image):
        result = self.model.predict_pipeline(raw_image)
        return result

    def run(self, raw_image):
        self.load_model()
        self.model_predict(raw_image)
