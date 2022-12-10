from Numpy.NumpyArray import NumpyArray
from Managers.ModelManager import ModelManager
from Managers.GDManager import GDManager
from Database.Database import Database
from Audio.Audio import Audio
from Recommendation.Recommendation_V2 import Recommendation_V2
class Controller:

    def __init__(self) -> None:
        self.numpy = NumpyArray()
        self.encoder2D = ModelManager()
        self.encoder1D = ModelManager()
        self.googleDrive = GDManager()
        self.SQL_DB = Database()
        self.audio = Audio()
        self.rec = Recommendation_V2()
        self.input_id = '1egJkNjgZOqZ3NNLVx_8TKuf-r5JFfPSY'
        self.output_folder = 'Train_Data'

    def generate_and_upload_representations(self):
        self.encoder2D.load_trained_model('Autoencoder_Saved/autoencoder.h5')
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

    def generate_final_representations_and_upload(self):
        self.encoder1D.load_trained_model('Autoencoder_Saved/autoencoder_secondary.h5')
        self.encoder1D.discard_layers(-9)
        fusion_test = self.numpy.read_numpy_file(self.output_folder+'/Test','fusion.npy')
        titles = self.numpy.read_numpy_file(self.output_folder+'/Test/slices','title.npy')
        fusion_test = self.numpy.expand_and_normalize(fusion_test,2)
        representations = [self.encoder1D.model_predict(self.numpy.expand(arr,0)) for arr in fusion_test]
        self.SQL_DB.drop_database()
        self.SQL_DB.connect_to_database()
        self.SQL_DB.create_table()
        for i in range(len(titles)):
            self.SQL_DB.insert_into_tables(titles[i],representations[i])
        self.SQL_DB.connection.commit()
        self.SQL_DB.connection.close()
        
    def generate_recommendations(self,new_music_file_path):
        return self.rec.generate_recommendation(new_music_file_path)