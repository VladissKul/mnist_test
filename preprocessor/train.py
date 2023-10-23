import pickle

from keras_model.model import Model


class Train:

    def __init__(self):
        self.load_config()
        self.mdl = Model()

    def load_config(self):
        pass

    def create_model(self):
        self.mdl.init_model()

    def train_model(self, image):
        self.mdl.load_data(image, preprocessing_type='train')
        self.mdl.preprocessing_data(preprocessing_type='train')
        self.mdl.train_model()

    def save_model(self):
        with open('/artefacts/model.pkl', 'wb') as f:
            pickle.dumps(self.mdl, f)

    def run(self, image):
        self.create_model()
        self.train_model()
        self.save_model()
