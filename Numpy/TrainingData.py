from sklearn.cluster import KMeans
import numpy as np
class TrainingData:

    def __init__(self) -> None:

        self.fields =list()
        self.dataDict = dict()
        self.mutlilabel_fields_train = list()
        self.mutlilabel_fields_test = list()
        self.spectrograms = list()
        self.data_fusion = list()


    def clear(self):
        self.fields.clear()
        self.dataDict.clear()
        self.mutlilabel_fields.clear()

    def split_data(self):
        buf_array = []
        buf_array.append(self.split_the_dataset_from_array(self.spectrograms,0.9))
        buf_array.append(self.split_the_dataset_from_array(self.clusterize_kmeans('rolloff_freq'),0.9))
        buf_array.append(self.split_the_dataset_from_array(self.clusterize_kmeans('tempo_bmp'),0.9))
        buf_array.append(self.add_array_to_multilabel_from_key('key_signature'))
        return buf_array

    def create_data_fusion(self):
        buf_array = self.split_data()
        

    def split_the_dataset_from_key(self,key,percentage):
        train_array = []
        test_array = []
        train_size = int(len(self.dataDict[key])*percentage)
        test_size = len(self.dataDict[key])-train_size
        for i in range(train_size):
            train_array.append(self.dataDict[key][i])
        for i in range(train_size,train_size+test_size):
            test_array.append(self.dataDict[key][i])
        return train_array,test_array

    def split_the_dataset_from_array(self,array,percentage):
        train_array = []
        test_array = []
        train_size = int(len(array)*percentage)
        test_size = len(array)-train_size
        for i in range(train_size):
            train_array.append(array[i])
        for i in range(train_size,train_size+test_size):
            test_array.append(array[i])
        return train_array,test_array

    def add_array_to_multilabel_from_key(self,key):
        train_array,test_array = self.split_the_dataset_from_key(key,0.9)
        self.mutlilabel_fields_train.append(train_array)
        self.mutlilabel_fields_test.append(test_array)

    def add_array_to_multilabel_from_array(self,array):
        train_array,test_array = self.split_the_dataset_from_array(array,0.9)
        self.mutlilabel_fields_train.append(train_array)
        self.mutlilabel_fields_test.append(test_array)

    def clusterize_kmeans(self,key):
        km_model = KMeans(n_clusters=8)
        buf = np.reshape(self.dataDict[key],(-1,1))
        km_result = km_model.fit(buf)
        return km_result.labels_