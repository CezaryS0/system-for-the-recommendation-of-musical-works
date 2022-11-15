from Managers.DataManager import DataManager
from Managers.ModelManager import ModelManager
from Numpy.NumpyArray import NumpyArray
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class Recommendation:

    def __init__(self) -> None:
        self.dataManager = DataManager()
        self.model = ModelManager()
        self.numpy = NumpyArray()

    def make_prediction(self,test_image):
        test_image = np.expand_dims(test_image,axis=0)
        prediction = self.model.model_predict(test_image)
        return prediction

    def cosine_similarity(self,prediction_anchor,predictions_song,counts):
        distance_array = []
        for i in range(len(predictions_song)):
            predictions_song[i] = predictions_song[i] / counts[i]
            distance_array.append(cosine_similarity(prediction_anchor,predictions_song[i]))
        return np.array(distance_array)
        
    def generate_recommendations(self,model_path,test_data,name):
        songs,titles = self.dataManager.get_test_spectrograms_slices_and_titles(test_data)
        songs = songs.astype(np.float32)
        songs = np.expand_dims(songs,axis=3)
        songs = songs/255
        self.model.load_trained_model(model_path)
        matrix_size = self.model.get_model_shape()[1]
        prediction_anchor = np.zeros((1,matrix_size))
        count = 0
        predictions_song = []
        predictions_title = []
        counts = []

        for i in range(len(titles)):
            if titles[i] == name:
                prediction = self.make_prediction(songs[i])
                prediction_anchor = prediction_anchor+prediction
                count+=1
            elif titles[i] not in predictions_title:
                predictions_title.append(titles[i])
                prediction = self.make_prediction(songs[i])
                predictions_song.append(prediction)
                counts.append(1)
            elif titles[i] in predictions_title:
                index = predictions_title.index(titles[i])
                prediction = self.make_prediction(songs[i])
                predictions_song[index] = predictions_song[index] + prediction
                counts[index] = counts[index] + 1
        prediction_anchor/=count
        distance_array = self.cosine_similarity(prediction_anchor,predictions_song,counts)
        return distance_array, predictions_title
        
    def print_predictions(self,distance_array,predictions_title):
        for i in range(2):
            index = np.argmax(distance_array)
            value = distance_array[index]
            print("Song name: ", predictions_title[index], "similarity = ",value)
            distance_array[index] = -np.inf

