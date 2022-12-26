from Model.Autoencoder.Encode import Encode
from Model.Numpy.NumpyArray import NumpyArray
from Model.Database.Database import Database
from Model.Numpy.NumpyArray import np
from Model.Utilities.Timer import Timer
from Model.Managers.DirectoryManager import DirectoryManager
from sklearn.metrics.pairwise import cosine_similarity

class Recommendation_V2:

    def __init__(self) -> None:
        self.encode = Encode()
        self.numpy = NumpyArray()
        self.database = Database()
        self.dm = DirectoryManager()
        self.timer = Timer()

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
        for i in range(len(title_array)):
            if title_array[i] not in predictions_title:
                predictions_title.append(title_array[i])
                predictions_song.append(representations[i])
                counts.append(1)
            elif title_array[i] in predictions_title:
                index = predictions_title.index(title_array[i] )
                predictions_song[index] = predictions_song[index] + representations[i]
                counts[index] = counts[index] + 1
        distance_array = self.cosine_similarity(prediction_anchor,predictions_song,counts)
        return distance_array,predictions_title

    def create_prediction_anchor(self,fusion):
        prediction_anchor = np.zeros(np.shape(fusion[0]))
        for spectrogram in fusion:
            prediction_anchor = prediction_anchor+spectrogram
        prediction_anchor/=len(fusion)
        return prediction_anchor

    def recommendations_to_list(self,distance_array,predictions_title):
        rec = list()
        for _ in range(3):
            index = np.argmax(distance_array)
            value = distance_array[index][0][0]
            rec.append((predictions_title[index],value))
            distance_array[index] = -np.inf
        return rec

    def encode_spectrograms(self,music_file_path):
        self.timer.startTimer()
        fusion = self.encode.encode(music_file_path)
        self.timer.endTimer()
        self.timer.saveResults("Full encoding",'results.txt',False)
        return fusion

    def calculate_similarities(self,fusion,title_array,representations):
        self.timer.startTimer()
        prediction_anchor = self.create_prediction_anchor(fusion)
        distance_array, predictions_title = self.predict_songs(prediction_anchor,title_array,representations)
        self.timer.endTimer()
        self.timer.saveResults("Cosine similarity for every song",'results.txt',False)
        return distance_array, predictions_title

    def generate_recommendation(self,music_file_path):
        fusion = self.encode_spectrograms(music_file_path)
        self.database.connect_to_database()
        title_array,representations = self.database.read_database()
        distance_array, predictions_title = self.calculate_similarities(fusion,title_array,representations)
        return self.recommendations_to_list(distance_array,predictions_title)
        