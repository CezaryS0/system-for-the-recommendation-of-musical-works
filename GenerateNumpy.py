from Numpy.NumpyArray import NumpyArray
from Managers.ModelManager import ModelManager
from Managers.GDManager import GDManager

input_id = '1eZ5SoAqW4c3RwNAM5m0FWJxzt3nrutVV'
output_folder = 'Train_Data'

googleDrive = GDManager()
numpy = NumpyArray()

encoder = ModelManager()
encoder.load_trained_model('Autoencoder_Saved/autoencoder.h5')
encoded_layers = encoder.get_number_of_layers()/2+1
encoder.discard_layers(-8)
numpy.save_dataset_to_numpy_files(encoder,"Spectrograms",output_folder)

#googleDrive.upload_directory_recursively(output_folder,input_id)
#googleDrive.search_folder(input_id,output_folder)