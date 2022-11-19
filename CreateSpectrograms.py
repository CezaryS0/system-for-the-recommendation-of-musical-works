from Managers.DataManager import DataManager
from Numpy.NumpyArray import NumpyArray

#dataset = 'dataset/fma_small'
dataset = 'dataset_wav'
csv_data = 'dataset/fma_metadata/tracks.csv'
dm = DataManager()
numpy = NumpyArray()

dm.create_and_slice_spectrograms("Spectrograms",dataset,csv_data)