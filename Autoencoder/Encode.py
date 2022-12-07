from Managers.ModelManager import ModelManager
from Numpy.TrainingData import TrainingData
from Numpy.NumpyArray import NumpyArray
from Numpy.NumpyArray import np
from Audio.Audio import Audio
import numpy as np
class Encode:

    def __init__(self) -> None:
        self.encoder2D = ModelManager()
        self.encoder1D = ModelManager()
        self.prepare_models()
        self.numpy = NumpyArray()
        self.data = TrainingData()
        self.audio = Audio()

    def prepare_models(self):
        self.encoder1D.load_trained_model('Autoencoder_Saved/autoencoder_secondary.h5')
        self.encoder1D.discard_layers(-9)
        self.encoder2D.load_trained_model('Autoencoder_Saved/autoencoder.h5')
        self.encoder2D.discard_layers(-8)

    def encode(self,new_music_file_path):
        if self.audio.load_file(new_music_file_path,256)==True:
            details = self.audio.get_file_details()
            slices = self.audio.slice_spectrograms_in_memory(256)
            slices = self.numpy.expand_and_normalize(slices,3)
            representations = [self.encoder2D.model_predict(self.numpy.expand(x,0)) for x in slices]
            fusion = self.data.fuse_single_image(representations,details,'Train_Data')
            fusion = self.numpy.expand_and_normalize(fusion,2)
            fusion = [self.encoder1D.model_predict(self.numpy.expand(x,0)) for x in fusion]
            return fusion