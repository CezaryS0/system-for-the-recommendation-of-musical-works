from Autoencoder.Encode import Encode
from Numpy.NumpyArray import NumpyArray
from Database.Database import Database
from Managers.ModelManager import ModelManager
from Numpy.NumpyArray import np
from sklearn.metrics.pairwise import cosine_similarity

class Recommendation_V2:

    def __init__(self) -> None:
        self.encode = Encode()
        self.numpy = NumpyArray()
        self.database = Database()
        self.model = ModelManager()
        self.model.load_trained_model('Autoencoder_Saved/autoencoder_secondary.h5')

    def cosine_similarity(self,prediction_anchor,predictions_song,counts):
        distance_array = []
        for i in range(len(predictions_song)):
            predictions_song[i] = predictions_song[i] / counts[i]
            distance_array.append(cosine_similarity(prediction_anchor,predictions_song[i]))
        return np.array(distance_array)

    def predict_songs(self,prediction_anchor,fusion_test):
        predictions_song = []
        predictions_title = []
        counts = []
        for i in range(len(fusion_test)):
            if fusion_test[i][0] not in predictions_title:
                predictions_title.append(fusion_test[i][0])
                predictions_song.append(fusion_test[i][1])
                counts.append(1)
            elif fusion_test[i][0] in predictions_title:
                index = predictions_title.index(fusion_test[i][0])
                predictions_song[index] = predictions_song[index] + fusion_test[i][1]
                counts[index] = counts[index] + 1
        distance_array = self.cosine_similarity(prediction_anchor,predictions_song,counts)
        return distance_array, predictions_title

    def create_prediction_anchor(self,fusion):
        matrix_size = self.model.get_model_shape()[1]
        prediction_anchor = np.zeros((1,matrix_size))
        for title,spectrogram in fusion:
            prediction_anchor = prediction_anchor+spectrogram
        prediction_anchor/=len(fusion)
        return prediction_anchor

    def print_predictions(self,name,distance_array,predictions_title):
        print("\nFor a song: ",name," I would recommend\n")
        for i in range(2):
            index = np.argmax(distance_array)
            value = distance_array[index]
            print(i+1,". ",predictions_title[index], ", similarity = ",value)
            distance_array[index] = -np.inf
        print('\n')


    def generate_recommendation(self,music_file_path):
        fusion = self.encode.encode(music_file_path)
        fusion_test = self.database.read_database()
        prediction_anchor = self.create_prediction_anchor(fusion)
        distance_array, predictions_title = self.predict_songs(prediction_anchor,fusion_test)
        self.print_predictions(fusion[0][0],distance_array,predictions_title)
        