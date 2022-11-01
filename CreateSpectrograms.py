from DataManager import DataManager

dataset = 'dataset/fma_small'
csv_data = 'dataset/fma_metadata/tracks.csv'
dm = DataManager()
dm.create_and_slice_spectrograms("Train_Spectrogram_Images",dataset,csv_data)