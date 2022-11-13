from Managers.DataManager import DataManager
from Managers.ModelManager import ModelManager
import numpy as np

class Recommendation:

    def __init__(self) -> None:
        self.dataManager = DataManager()
        self.model = ModelManager()

    def make_prediction(self,test_image):
        test_image = np.expand_dims(test_image,axis=0)
        prediction = self.model.model_predict(test_image)
        return prediction

    def generate_recommendations(self,model_path,test_data,name):
        songs,titles = self.dataManager.get_spectrograms_and_titles(test_data)
        songs = np.expand_dims(songs,axis=3)
        print(np.unique(titles))
        self.model.load_trained_model(model_path)
        matrix_size = self.model.get_model_shape()[1]

        prediction_anchor = np.zeros((1,matrix_size))
        count = 0
        predictions_song = []
        predictions_title = []
        counts = []
        distance_array = []

        for i in range(len(titles)):
            if titles[i] == name:
                prediction = self.make_prediction(songs[i])
                prediction_anchor = prediction_anchor+prediction
                count+=1
            elif titles[i] not in predictions_title:
                predictions_song.append(titles[i])
                prediction = self.make_prediction(songs[i])
                predictions_song.append(prediction)
                counts.append(1)
            elif titles[i] in predictions_title:
                index = predictions_title.index(titles[i])
                predictions_song[index] = predictions_song[index] + prediction
                counts[index] = counts[index] + 1

