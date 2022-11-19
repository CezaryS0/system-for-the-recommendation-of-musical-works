from Numpy.NumpyArray import NumpyArray
from Managers.GDManager import GDManager

input_id = '1eZ5SoAqW4c3RwNAM5m0FWJxzt3nrutVV'
output_folder = 'Train_Data'

googleDrive = GDManager()
numpy = NumpyArray()
numpy.save_dataset_to_numpy_files("Spectrograms",output_folder)
googleDrive.upload_directory_recursively(output_folder,input_id)