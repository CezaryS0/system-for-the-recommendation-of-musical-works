from keras.models import Model, load_model
from Numpy.NumpyArray import NumpyArray
from Managers.DataManager import DataManager
import numpy as np
import os



class ModelManager:

    def __init__(self) -> None:
        self.model = None
        self.numpy = NumpyArray()
        self.dataManager = DataManager()

    def load_trained_model(self,model_path):
        self.model = load_model(model_path)
        self.model.set_weights(self.model.get_weights())

    def model_summary(self):
        if(self.model!=None):
            print(self.model.summary())
    
    def get_model_shape(self):
        if self.model!=None:
            return self.model.output.shape
        return None


    def model_predict(self,input):
        if self.model!=None:
            return self.model.predict(input)
        return None
