from Numpy.NumpyArray import NumpyArray
from Managers.ModelManager import ModelManager
from Managers.GDManager import GDManager
class Controller:

    def __init__(self) -> None:
        self.numpy = NumpyArray()
        self.encoder2D = ModelManager()
        self.googleDrive = GDManager()
        self.input_id = '1eZ5SoAqW4c3RwNAM5m0FWJxzt3nrutVV'
        self.output_folder = 'Train_Data'

    def generate_and_upload_representations(self):
        self.encoder2D.load_trained_model('Autoencoder_Saved/autoencoder.h5')
        encoded_layers = self.encoder2D.get_number_of_layers()/2+1
        self.encoder2D.discard_layers(-8)
        data = self.numpy.read_sliced_spectrograms('Spectrograms')
        self.numpy.save_spectrogram_representations(data,self.encoder2D,self.output_folder)
        self.googleDrive.upload_file_to_folder('Test',self.output_folder+'/Test/representations.npy',self.input_id)
        self.googleDrive.upload_file_to_folder('Train',self.output_folder+'/Train/representations.npy',self.input_id)

    def generate_data_fusion_and_upload(self):
        data = self.numpy.read_sliced_spectrograms('Spectrograms')
        data.representations_test = self.numpy.read_numpy_file(self.output_folder+'/Test','representations.npy')
        data.representations_train = self.numpy.read_numpy_file(self.output_folder+'/Train','representations.npy')
        self.numpy.save_data_fusion(data,self.output_folder)
        self.googleDrive.upload_file_to_folder('Test',self.output_folder+'/Test/fusion.npy',self.input_id)
        self.googleDrive.upload_file_to_folder('Train',self.output_folder+'/Train/fusion.npy',self.input_id)

    def save_sliced_spectrograms_and_upload(self):
        self.numpy.save_dataset_to_numpy_files("Spectrograms",self.output_folder)
        self.googleDrive.upload_directory_recursively(self.output_folder,self.input_id,True)
