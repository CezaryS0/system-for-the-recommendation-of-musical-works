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
        predictions_title = []
        counts = []
        for i in range(int(len(title_array))):
            if title_array[i] not in predictions_title:
                predictions_title.append(title_array[i])
                predictions_song.append(representations[i])
                counts.append(1)
            elif title_array[i] in predictions_title:
                index = predictions_title.index(title_array[i])
                predictions_song[index] = predictions_song[index] + representations[i]
                counts[index] = counts[index] + 1
        distance_array = self.cosine_similarity(prediction_anchor,predictions_song,counts)
        return distance_array, predictions_title

    def create_prediction_anchor(self,fusion):
        prediction_anchor = np.zeros(np.shape(fusion[0]))
        for spectrogram in fusion:
            prediction_anchor = prediction_anchor+spectrogram
        prediction_anchor/=len(fusion)
        return prediction_anchor

    def print_predictions(self,name,distance_array,predictions_title):
        print("\nFor a song: ",name," I would recommend\n")
        print(distance_array)
        for i in range(2):
            index = np.argmax(distance_array)
            value = distance_array[index]
            print(i+1,". ",predictions_title[index], ", similarity = ",value)
            distance_array[index] = -np.inf
        print('\n')

    def recommendations_to_list(self,distance_array,predictions_title):
        rec = list()
        for i in range(2):
            index = np.argmax(distance_array)
            value = distance_array[index]
            rec.append((predictions_title[index],value))
            distance_array[index] = -np.inf
        return rec

    def generate_recommendation(self,music_file_path):
        name = self.dm.get_file_name(music_file_path)[0]
        fusion = self.encode.encode(music_file_path)
        self.database.connect_to_database()
        title_array,representaions = self.database.read_database()
        prediction_anchor = self.create_prediction_anchor(fusion)
        distance_array, predictions_title = self.predict_songs(prediction_anchor,title_array,representaions)
        #self.print_predictions(name,distance_array,predictions_title)
        return self.recommendations_to_list(distance_array,predictions_title)
        