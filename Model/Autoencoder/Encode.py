from Model.Managers.ModelManager import ModelManager
from Model.Numpy.TrainingData import TrainingData
from Model.Numpy.NumpyArray import NumpyArray
from Model.Numpy.NumpyArray import os
from Model.Audio.Audio import Audio
from Model.Utilities.Timer import Timer
from Model.Managers.DirectoryManager import DirectoryManager
class Encode:

    def __init__(self) -> None:
        self.encoder2D = ModelManager()
        self.encoder1D = ModelManager()
        self.timer = Timer()
        self.prepare_models()
        self.numpy = NumpyArray()
        self.data = TrainingData()
        self.audio = Audio()
        self.dm = DirectoryManager()
        self.result_path=""
    
    def prepare_models(self):
        self.encoder1D.load_trained_model('Autoencoder_Saved/autoencoder_secondary.h5')
        self.encoder1D.discard_layers(-15)
        self.encoder2D.load_trained_model('Autoencoder_Saved/autoencoder.h5')
        self.encoder2D.discard_layers(-14)

    def generate_slices(self,new_music_file_path):
        filename = os.path.basename(new_music_file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        self.timer.startTimer()
        details = self.audio.get_file_details(filename)
        
        self.audio.save_spectrogtram_mfcc('Recommendation Results/'+filename_no_ext+'/anchor_song/'+filename_no_ext+'.png')
        slices = self.audio.slice_spectrograms_in_memory(256)
        slices = self.numpy.expand_and_normalize(slices,3)
        self.timer.endTimer()
        self.timer.saveResults('Spectrograms generation',self.result_path,False)
        return slices,details

    def generate_representations(self,slices):
        self.timer.startTimer()
        representations = [self.encoder2D.model_predict(self.numpy.expand(x,0)) for x in slices]
        self.timer.endTimer()
        self.timer.saveResults('Spectrogram encoding',self.result_path,False)
        return representations

    def fuse_single_image(self,representations,details):
        self.timer.startTimer()
        fusion = self.data.fuse_single_image(representations,details,'Train_Data')
        self.timer.endTimer()
        self.timer.saveResults('Fusion generation',self.result_path,False)
        return fusion

    def encode_data_fusion(self,fusion):
        self.timer.startTimer()
        fusion = self.numpy.expand_and_normalize(fusion,2)
        fusion = [self.encoder1D.model_predict(self.numpy.expand(x,0)) for x in fusion]
        self.timer.endTimer()
        self.timer.saveResults('Fusion encoding',self.result_path,False)
        return fusion

    def encode(self,new_music_file_path):
        filename = os.path.basename(new_music_file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        self.dm.create_filename_dir('Recommendation Results',filename_no_ext,'anchor_song')
        self.result_path = 'Recommendation Results/'+filename_no_ext+'/results.txt'
        self.timer.startTimer()
        self.dm.create_main_dir('Recommendation Results')
        if self.audio.load_file(new_music_file_path,256)==True:
            self.timer.endTimer()
            self.timer.saveResults('Librosa load',self.result_path,True)
            slices,details = self.generate_slices(new_music_file_path)
            representations= self.generate_representations(slices)
            fusion = self.fuse_single_image(representations,details)
            return self.encode_data_fusion(fusion)