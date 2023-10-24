import os
import sys
import cv2
import pickle


class Train:

    def __init__(self):
        self.load_config()

    def load_config(self):
        pass

    def init_model(self):
        sys.path.append(f"""{os.environ['root']}/model""")
        from model import Model
        self.model = Model()

    def model_fitting(self):
        self.model.train_pipeline()

    def run(self):
        self.init_model()
        self.model_fitting()
