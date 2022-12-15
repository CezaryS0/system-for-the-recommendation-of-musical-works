from Model.Managers.DataManager import DataManager
from Model.Numpy.NumpyArray import NumpyArray

dataset = 'dataset/fma_full'
#dataset = 'dataset_wav'
csv_data = 'dataset/fma_metadata/tracks.csv'
dm = DataManager()
numpy = NumpyArray()

dm.create_and_slice_spectrograms("Spectrograms",dataset,csv_data,256)