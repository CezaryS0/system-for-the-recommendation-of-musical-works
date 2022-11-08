from statistics import mode
from Numpy.NumpyArray import *
import pandas as pd
import numpy as np
from keras import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization
from Managers.DirectoryManager import DirectoryManager
from sklearn.model_selection import train_test_split

class Encoder:

    def __init__(self):
        self.dm = DirectoryManager()
        self.numpy = NumpyArray()
        self.encoder = None

    def reshape_data(self,train_x):
        list = []
        
        for elem in train_x:
            shape = np.shape(elem)
            list.append(np.reshape(elem,(shape[0],shape[1],1)))
        return list

    def prepare_data_for_training(self,spectrograms_array,labels):
        train_x, test_x, train_y, test_y = train_test_split(spectrograms_array, labels, test_size=0.05, shuffle=True)
        #train_x = np.reshape(train_x,(len(train_x),shape[0],shape[1],1))
        return train_x, test_x, train_y, test_y

    def encode(self,shape,classes) -> Model:
        input_img = Input(shape=shape)

        #encoded = Conv2D(filters=16,kernel_size=(3,3), input_shape=(128,1291,1),activation='relu',padding='same')(input_img)
        
        encoded = Sequential()
        encoded = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu')(input_img)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,4))(encoded)
        encoded = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(encoded)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,4))(encoded)
        encoded = Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu')(encoded)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,4))(encoded)
        encoded = Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu')(encoded)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((16,19))(encoded)
        encoded = Dense(4096,activation='relu')(encoded)
        encoded = Dense(classes,activation='softmax')(encoded)
        encoder = Model(inputs=input_img,outputs=encoded)
        return encoder

    def train_model(self,dataset_path,feature_file):
        self.dm.create_main_dir("Saved_Model")
        spectrograms_array = self.numpy.read_spectrograms_file(dataset_path)
        
        labels = self.numpy.read_numpy_file(dataset_path,feature_file)
        train_x, test_x, train_y, test_y = self.prepare_data_for_training(spectrograms_array,labels)
        shape = np.shape(train_x[0])
        shape = (shape[0],shape[1]+5,1)

        self.encoder = self.encode(shape,len(labels))
        self.encoder.compile(loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy'])
        self.encoder.fit(train_x, train_y, epochs=10, validation_split=0.1)
        #pd.DataFrame(self.encoder.fit(train_x, train_y, epochs=10, validation_split=0.1).history).to_csv("Saved_Model/training_history.csv")
        #self.encoder.evaluate(test_x,test_y)