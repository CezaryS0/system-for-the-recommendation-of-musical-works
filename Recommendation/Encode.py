import os
from Numpy.NumpyArray import NumpyArray
from Managers.ModelManager import ModelManager

class Encode:

    def __init__(self) -> None:
        self.numpy = NumpyArray()
        self.model = ModelManager()


    def encode_classification(self,test_path,model_path):
        self.model.load_trained_model(model_path)
        representations = self.numpy.read_numpy_file(test_path,'representations.npy')
        #fusion = self.numpy.read_numpy_file(test_path,'representations.npy')
        buf_array = []
        for spectr in representations:
            buf_array.append(self.model.model_predict(spectr))
        