from Autoencoder.Encode import Encode
from Numpy.NumpyArray import NumpyArray
from Database.Database import Database
from Managers.ModelManager import ModelManager
from Numpy.NumpyArray import np
from Managers.DirectoryManager import DirectoryManager
from sklearn.metrics.pairwise import cosine_similarity

class Recommendation_V2:

    def __init__(self) -> None:
        self.encode = Encode()
        self.numpy = NumpyArray()
        self.database = Database()
        self.model = ModelManager()
        self.dm = DirectoryManager()
        self.model.load_trained_model('Autoencoder_Saved/autoencoder_secondary.h5')

    def cosine_similarity(self,prediction_anchor,predictions_song,counts):
        distance_array = []
        for i in range(len(predictions_song)):
            predictions_song[i] = predictions_song[i] / counts[i]
            pred_song = predictions_song[i]
            similarity = cosine_similarity(prediction_anchor.flatten().reshape(1, -1) ,pred_song.flatten().reshape(1, -1))   
            distance_array.append(similarity)
        return np.array(distance_array)

    def predict_songs(self,prediction_anchor,title_array,representations):
        predictions_song = []
        predictions_id = []
        counts = []
        for i in range(int(len(title_array))):
            if i not in predictions_id:
                predictions_id.append(i)
                predictions_song.append(representations[i])
                counts.append(1)
            elif i in predictions_id:
                index = predictions_id.index(i)
                predictions_song[index] = predictions_song[index] + representations[i]
                counts[index] = counts[index] + 1
        distance_array = self.cosine_similarity(prediction_anchor,predictions_song,counts)
        return distance_array,predictions_id

    def create_prediction_anchor(self,fusion):
        prediction_anchor = np.zeros(np.shape(fusion[0]))
        for spectrogram in fusion:
            prediction_anchor = prediction_anchor+spectrogram
        prediction_anchor/=len(fusion)
        return prediction_anchor

    def recommendations_to_list(self,distance_array,title_array,predictions_id):
        rec = list()
        for i in range(2):
            index = np.argmax(distance_array)
            value = distance_array[index][0][0]
            id = predictions_id[index]
            title = title_array[id]
            rec.append((id,title,value))
            distance_array[index] = -np.inf
        return rec

    def generate_recommendation(self,music_file_path):
        name = self.dm.get_file_name(music_file_path)[0]
        fusion = self.encode.encode(music_file_path)
        self.database.connect_to_database()
        title_array,representaions = self.database.read_database()
        prediction_anchor = self.create_prediction_anchor(fusion)
        distance_array, predictions_id = self.predict_songs(prediction_anchor,title_array,representaions)
        return self.recommendations_to_list(distance_array,title_array,predictions_id)
        