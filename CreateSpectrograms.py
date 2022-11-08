from Managers.DataManager import DataManager
from Numpy.NumpyArray import NumpyArray

dataset = 'dataset/fma_small'
csv_data = 'dataset/fma_metadata/tracks.csv'
dm = DataManager()
numpy = NumpyArray()
dm.create_and_slice_spectrograms("Train_Spectrogram_Images",dataset,csv_data)