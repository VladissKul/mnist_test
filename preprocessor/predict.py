
import pickle
from model import Model


class Predict:
    
    
    def __init__(self):
        self.load_config()
        
    
    def load_config(self):
        self.config['model_name']
    
    
    def load_model(self):
        with open(self.config['model_name'], 'rb') as f:
            self.mdl = pickle.loads(f)
    
    
    def predict_model(self, image):
        self.mdl.load_data(image, preprocessing_type = 'predict')
        self.mdl.preprocessing_data(preprocessing_type = 'predict')
        self.mdl.train_model()
        
        
    def save_preprocessed_data(self):
        self.load_model()
        self.predict_model()
        self.save_preprocessed_data()
        
        
    def run(self):
        self.load_model()
        self.predict_model()
        self.save_preprocessed_data()
        