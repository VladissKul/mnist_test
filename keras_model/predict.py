import cv2
import numpy as np
from matplotlib import pyplot as plt


class Predict:
    def __init__(self, CLASSIFY_MODEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL, device):
        self.CLASSIFY_MODEL = CLASSIFY_MODEL
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_CHANNEL = IMAGE_CHANNEL
        self.device = device

    def preprocess_image(self, image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        return image

    def create_request_input(self, preprocessed_image):
        preprocessed_image_array = np.frombuffer(preprocessed_image, dtype=np.uint8)
        preprocessed_image_array = preprocessed_image_array.reshape(1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT,
                                                                    self.IMAGE_CHANNEL)
        request_input = preprocessed_image_array.astype('float32') / 255.0
        return request_input

    def predict_digit(self, preprocessed_image):
        request_input = self.create_request_input(preprocessed_image)
        prediction = self.CLASSIFY_MODEL.predict(request_input)
        prediction = np.argmax(prediction, axis=1)
        return prediction

    def save_data_predict(self, preprocessed_image, prediction, image_path):
        preprocessed_image_array = np.frombuffer(preprocessed_image, dtype=np.uint8)
        preprocessed_image_array = preprocessed_image_array.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

        plt.imshow(preprocessed_image_array, cmap='gray')
        plt.title(f'Predicted Digit: {prediction[0]}')
        plt.axis('off')

        plt.savefig(image_path)
