import os
import pickle

import cv2
import numpy as np
from keras import models, layers
from tensorflow import keras
from core.config.custom_config import CustomConfig

class Model:


    def __init__(self):
        self.load_config()

    def init_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        self.model = model


    def load_data(self, file_image: str = f"""{os.environ['root']}/model/artefacts/mnist.npz""", preprocessed_type: str = 'train'):
        if preprocessed_type == 'train':
            (self.x_train, self.y_train), _ = keras.datasets.mnist.load_data(path=file_image)
        elif preprocessed_type == 'predict':
            self.raw_image = cv2.imread(file_image, cv2.IMREAD_GRAYSCALE)


    def load_config(self):
        self.config = {}
        self.config['image_height'] = 28
        self.config['image_width'] = 28
        self.config['epoch'] = 1
        self.config['batch_size'] = 32
        self.config['learning_rate'] = 0.003
        self.config['image_channel'] = 1
        # внести в secrets/config.yml

    def load_config(self):
        config = CustomConfig(os.environ['ServiceConfigFile'])
        config.load_full_config()
        self.config = config.load_section_config(self.class_name)


    def preprocessing_data(self, preprocessing_type: str = 'train'):
        if preprocessing_type == 'train':
            self.x_train = self.x_train.reshape((self.x_train.shape[0], self.config['image_height'], self.config['image_width'], 1))
            self.x_train = self.x_train / 255.0


        elif preprocessing_type == 'predict':
            self.preprocessed_image = cv2.resize(self.raw_image, (self.config['image_width'], self.config['image_height']))
            self.preprocessed_image = np.frombuffer(self.preprocessed_image, dtype=np.uint8)
            self.preprocessed_image = self.preprocessed_image.reshape(1, self.config['image_width'], self.config['image_height'], self.config['image_channel'])
            self.preprocessed_image = self.preprocessed_image.astype('float32') / 255.0


    def fit_model(self):
        self.mdl.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.mdl.train(x_train, y_train, batch_size=self.config['batch_size'], epochs=self.config['epoch'], validation_split=0.1)


    def predict(self):
        self.result = self.mdl.predict(self.preprocessed_image)


    def save_model(self):
        with open(f"""{os.environ['root']}/model/artefacts/model.pkl""", 'wb') as f:
            pickle.dump(self, f)


    def predict_pipeline(self, raw_image):
        self.load_data(raw_image, preprocessing_type = 'predict')
        self.preprocessing_data(preprocessing_type = 'predict')
        return self.result


    def train_pipeline(self):
        self.init_model()
        self.load_data(preprocessing_type = 'train')
        self.preprocessing_data(preprocessing_type = 'train')
        self.fit_model()
        self.save_model()


