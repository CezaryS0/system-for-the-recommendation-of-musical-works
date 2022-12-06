from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
class TrainingData:

    def __init__(self) -> None:

        self.fields =list()
        self.dataDict = dict()
        self.spectrograms = list()
        self.fusion = [np.array([]),np.array([])]
        self.representations_train = list()
        self.representations_test = list()

    def clear(self):
        self.fields.clear()
        self.dataDict.clear()

    def split_data(self):
        buf_array = []
        buf_array.append((self.representations_train,self.representations_test))
        buf_array.append(self.split_the_dataset_from_array(self.clusterize_kmeans('rolloff_freq',8),0.9))
        buf_array.append(self.split_the_dataset_from_array(self.clusterize_kmeans('tempo_bpm',8),0.9))
        buf_array.append(self.split_the_dataset_from_array(self.encode_labels('key_signature'),0.9))
        return buf_array

    def create_data_fusion(self):
        splitted_data = self.split_data()
        buf_array = []
        for n in range(2):
            for i in range(len(splitted_data[0][n])):
                buf_array = np.concatenate((buf_array,splitted_data[0][n][i].flatten()))
                for j in range(1,len(splitted_data)):
                    buf_array = np.concatenate((buf_array,[splitted_data[j][n][i]]))
                if self.fusion[n].size==0:
                    self.fusion[n] = buf_array
                else:
                    self.fusion[n] = np.vstack((self.fusion[n],buf_array))
                buf_array = []
        return self.fusion[0],self.fusion[1]

    def fuse_single_image(self,representations,details:dict):
        buf_array = []
        fusion = []
        for r in representations:
            buf_array = np.concatenate((buf_array,r.flatten()))
            buf_array = np.concatenate((buf_array,self.clusterize_kmeans_array(details['rolloff_freq'],8)))
            buf_array = np.concatenate((buf_array,self.clusterize_kmeans_array(details['tempo_bpm'],8)))
            buf_array = np.concatenate((buf_array,self.encode_labels_from_array(details['key_signature'])))
            if fusion.size==0:
                fusion = buf_array
            else:
                fusion = np.vstack((fusion,buf_array))
            buf_array = []
        return fusion

    def encode_labels_from_array(self,arr):
        unique_values = list(set(arr))
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform(unique_values)
        values_dict = dict(zip(unique_values,encoded_labels))
        buf_array = []
        for val in arr:
            buf_array.append(values_dict[val])
        return buf_array

    def encode_labels(self,key):
        unique_values = list(set(self.dataDict[key]))
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform(unique_values)
        values_dict = dict(zip(unique_values,encoded_labels))
        buf_array = []
        for val in self.dataDict[key]:
            buf_array.append(values_dict[val])
        return buf_array

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

    def clusterize_kmeans_array(self,arr,clusters):
        km_model = KMeans(n_clusters=clusters)
        buf = np.reshape(arr,(-1,1))
        km_result = km_model.fit(buf)
        return km_result.labels_

    def clusterize_kmeans(self,key,clusters):
        km_model = KMeans(n_clusters=clusters)
        buf = np.reshape(self.dataDict[key],(-1,1))
        km_result = km_model.fit(buf)
        return km_result.labels_